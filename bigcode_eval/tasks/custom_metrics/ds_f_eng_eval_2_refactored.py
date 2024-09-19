# Base Model Class: Defines the interface for models (e.g., encode, fit, predict).
# Model Implementations: Separate classes for TabPFN and XGBoost models.
# DataProcessor Class: Handles data loading, preprocessing, and splitting.
# Evaluator Class: Evaluates model performance, can be extended to handle different models.
from abc import ABC, abstractmethod
import os
from collections import defaultdict, namedtuple
from typing import List, Dict, Set, Union, Tuple
from copy import deepcopy
from itertools import combinations

import bigcode_eval.tasks.custom_metrics.ds_f_eng_kernel_studies
import pandas as pd
import numpy as np
from bigcode_eval.tasks.custom_metrics.ds_f_eng_kernel_studies import transform
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from tabpfn import TabPFNClassifier
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from .execute import unsafe_execute

_WARNING = """
# WARNING: Untrusted code execution. Read disclaimer and set the appropriate environment variable to continue.
"""

DataSplit = namedtuple('DataSplit', ['test_x', 'test_target', 'train_x', 'train_target'])


class BaseModel(ABC):
    model_slug: str = "base"

    def __init__(self, enable_categorical: bool = False) -> None:
        self.enable_categorical = enable_categorical

    @abstractmethod
    def baseline_encode(self, data_split: DataSplit) -> DataSplit:
        pass

    @abstractmethod
    def fit(self, train_x: pd.DataFrame, train_target: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        pass

    def handle_missing(self, categorical_columns: List[str], test_x: pd.DataFrame, train_x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for col in train_x.columns:
            self._handle_missing_both(test_x[col], train_x[col], is_categorical=col in categorical_columns)
        return test_x, train_x

    @staticmethod
    def _handle_missing_both(test_x: pd.DataFrame, train_x:pd.DataFrame, is_categorical=True) -> None:
        if is_categorical:
            train_x.fillna('missing', inplace=True)
            test_x.fillna('missing', inplace=True)
        else:
            train_x.fillna(train_x.mean(), inplace=True)
            test_x.fillna(test_x.mean(), inplace=True)


class XGBoostModel(BaseModel):
    model_slug: str = "xgboost"

    def __init__(self, enable_categorical: bool = False, is_regression=False) -> None:
        super().__init__(enable_categorical)
        self.model = None
        self.is_regression = is_regression

    def baseline_encode(self, data_split: DataSplit) -> DataSplit:
        test_x = XGBoostModel.convert_objects_to_category(data_split.test_x)
        train_x = XGBoostModel.convert_objects_to_category(data_split.train_x)
        train_x = DataProcessor.clean_column_names(train_x)
        test_x = DataProcessor.clean_column_names(test_x)
        return DataSplit(test_x, data_split.test_target, train_x, data_split.train_target)

    def fit(self, train_x: pd.DataFrame, train_target: pd.Series):
        if self.is_regression:
            self.model = xgb.XGBRegressor(enable_categorical=self.enable_categorical)
            self.model.fit(train_x, train_target)
        else:
            self.model = xgb.XGBModel(enable_categorical=self.enable_categorical,
                                  num_class=len(train_target.unique()))
            self.model.fit(train_x, train_target.astype('category').cat.codes)

    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(test_x)

    @staticmethod
    def convert_objects_to_category(df: pd.DataFrame) -> pd.DataFrame:
        # Identify all object columns
        object_cols = df.select_dtypes(include=['object']).columns
        # Convert each object column to category type
        df[object_cols] = df[object_cols].astype('category')
        return df


class TabPFNModel(BaseModel):
    model_slug: str = "tabpfn"

    def __init__(self, enable_categorical=False) -> None:
        super().__init__(enable_categorical)
        self.model = None

    def baseline_encode(self, data_split: DataSplit) -> DataSplit:      # extract to a base
        train_df, test_df = data_split.train_x, data_split.test_x
        categorical_columns = DataProcessor.identify_categorical_columns(train_df, unique_val_threshold=5)
        test_df, train_df = self.handle_missing(categorical_columns, test_df, train_df)
        train_df[categorical_columns] = train_df[categorical_columns].astype(str)
        test_df[categorical_columns] = test_df[categorical_columns].astype(str)
        train_x_oh, test_x_oh = DataProcessor.one_hot_encode(train_df, test_df, categorical_columns)
        train_x = train_x_oh.join(train_df.drop(columns=categorical_columns))
        test_x = test_x_oh.join(test_df.drop(columns=categorical_columns))
        train_x = DataProcessor.clean_column_names(train_x)
        test_x = DataProcessor.clean_column_names(test_x)
        return DataSplit(test_x, data_split.test_target, train_x, data_split.train_target)

    def fit(self, train_x: pd.DataFrame, train_target: pd.Series) -> None:
        self.model = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
        self.model.fit(train_x, train_target, overwrite_warning=True)

    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        return self._process_long_test(test_x, self.model)

    @staticmethod
    def _process_long_test(test_x: pd.DataFrame, classifier: BaseModel, max_test_size=1000, max_workers=4) -> np.ndarray:
        def process_batch(start_p: int) -> np.ndarray:
            batch_test = test_x[start_p:start_p + max_test_size]
            return classifier.predict_proba(batch_test).argmax(1)

        predictions = np.array([], dtype=int)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(process_batch, start_pos): start_pos for start_pos in
                                   range(0, len(test_x), max_test_size)}
                predictions = np.concatenate([future.result() for future in as_completed(future_to_batch)])
        else:
            for start_position in range(0, len(test_x), max_test_size):
                predictions = np.append(predictions, process_batch(start_position))
        return predictions


class DataProcessor:
    def __init__(self, dataframe_global_path: Path, timeout: int = 60, allow_no_transform: bool = False) -> None:
        self.timeout = timeout
        self.allow_no_transform = allow_no_transform
        self.dataframe_global_path = dataframe_global_path
        self.split_column = "ds_f_eng__split"
        self.train_split = "train"
        self.test_split = "test"
        self.target_column = "ds_f_eng__target__response"
        self.unnamed = 'Unnamed: 0'

    def load_dataframe(self, dataframe_name: Path) -> DataSplit:
        dataframe_path = self.dataframe_global_path / dataframe_name
        df = pd.read_csv(dataframe_path) if dataframe_path.suffix == ".csv" else pd.read_csv(str(dataframe_path) + ".csv")
        train_dataframe = df[df[self.split_column] == self.train_split]
        test_dataframe = df[df[self.split_column] == self.test_split]
        return self.prepare_train_test_split(test_dataframe, train_dataframe)

    def prepare_train_test_split(self, test_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame) -> DataSplit:
        test_dataframe.dropna(subset=[self.target_column], inplace=True)        # FIXME:  is it baseline-related processing or not? if generic run it through all of the datasets
        train_dataframe.dropna(axis=1, how='all', inplace=True)
        test_dataframe = test_dataframe[train_dataframe.columns]
        train_dataframe = train_dataframe.dropna(subset=[self.target_column])
        ##
        train_target = train_dataframe[self.target_column]
        test_target = test_dataframe[self.target_column]
        train_x = self._remove_columns(train_dataframe, columns_to_remove={self.target_column, self.split_column})
        test_x = self._remove_columns(test_dataframe, columns_to_remove={self.target_column, self.split_column})
        return DataSplit(test_x=test_x, test_target=test_target, train_x=train_x, train_target=train_target)

    def _remove_columns(self, dataframe: pd.DataFrame, columns_to_remove: Set[str]) -> pd.DataFrame:
        if self.unnamed in dataframe.columns:
            columns_to_remove.add(self.unnamed)
        return dataframe.drop(columns=columns_to_remove, errors='ignore')

    @staticmethod
    def identify_categorical_columns(df: pd.DataFrame, unique_val_threshold: int) -> List[str]:
        cat_cols_by_dtype = df.select_dtypes(include=['object', 'category']).columns
        cat_cols_by_uniq = [col for col in df.columns if df[col].nunique() < unique_val_threshold]
        return list(set(cat_cols_by_dtype) | set(cat_cols_by_uniq))

    @staticmethod
    def one_hot_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_columns: List[str], fit_on_combined: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        combined_df = pd.concat([train_df[categorical_columns], test_df[categorical_columns]]) if fit_on_combined else train_df[categorical_columns]
        max_categories = DataProcessor.tune_max_categories_per_feature(test_df, categorical_columns)
        onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='infrequent_if_exist', max_categories=max_categories).fit(combined_df.astype(str))
        train_oh = pd.DataFrame(onehot_encoder.transform(train_df[categorical_columns]), columns=onehot_encoder.get_feature_names_out())
        test_oh = pd.DataFrame(onehot_encoder.transform(test_df[categorical_columns]), columns=onehot_encoder.get_feature_names_out())
        return train_oh, test_oh

    @staticmethod
    def tune_max_categories_per_feature(test_df: pd.DataFrame, categorical_columns: List[str]) -> int:
        num_categorical_columns = max(1, len(categorical_columns))
        max_cat_per_one = int((100 - test_df.shape[1]) / num_categorical_columns + 2)
        return max_cat_per_one

    def _transform_dataframe_inplace(self, prediction: str, dataframe: pd.DataFrame) -> Union[pd.DataFrame, None]:
        if len(prediction) == 0 and self.allow_no_transform:
            return dataframe
        if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
            raise ValueError(_WARNING)
        # TODO(ajedrosz): header constant
        df_transformation_with_assignment = f"""{prediction}
    dataframe_transformed = transform(dataframe)"""
        # TODO(ajedrosz): need to copy?
        exec_globals = {"dataframe": dataframe}
        result = []
        unsafe_execute(
            check_program=df_transformation_with_assignment,
            result=result,
            timeout=self.timeout,
            exec_globals=exec_globals,
        )
        if result.pop() != "passed":
            return None
        else:
            return exec_globals["dataframe_transformed"]

    def _transform_dataframe_inplace_three_tuple(self, prediction: str, data_split: DataSplit) -> Union[DataSplit, None]:
        if len(prediction) == 0 and self.allow_no_transform:
            return data_split
        if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
            raise ValueError(_WARNING)
        # TODO(ajedrosz): header constant
        df_transformation_with_assignment = f"""{prediction}
        \nnew_train_x, new_train_target, new_test_x  = transform(train_x, train_target, test_x)"""
        # TODO(ajedrosz): need to copy?
        exec_globals = {"test_x": data_split.test_x,
                        "train_x": data_split.train_x,
                        "train_target": data_split.train_target}
        result = []
        unsafe_execute(
            check_program=df_transformation_with_assignment,
            result=result,
            timeout=self.timeout,
            exec_globals=exec_globals,
        )
        if result.pop() != "passed":
            return None
        else:
            new_data_split = DataSplit(exec_globals["new_test_x"], data_split.test_target, exec_globals["new_train_x"], exec_globals["new_train_target"])
            return new_data_split

    def transform_dataframe(self, prediction: str, data_split: DataSplit) -> DataSplit:
        # new_train_x = self._transform_dataframe_inplace(prediction, data_split.train_x)
        # new_test_x = self._transform_dataframe_inplace(prediction, data_split.test_x)
        transformed_ds = self._transform_dataframe_inplace_three_tuple(prediction, data_split)
        if transformed_ds is None:
            return data_split
        else:
            return transformed_ds

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.replace('[', '(', regex=False)
        df.columns = df.columns.str.replace(']', ')', regex=False)
        df.columns = df.columns.str.replace('<', 'lesser', regex=False)
        df.columns = df.columns.str.replace('>', 'greater', regex=False)
        return df


class Evaluator:
    def __init__(self, data_processor: DataProcessor, logging_path: Path, metadata: pd.DataFrame) -> None:
        self.data_processor = data_processor
        self.logging_path = logging_path
        self.models_classification = XGBoostModel(enable_categorical=True, is_regression=False)
        self.models_regression = XGBoostModel(enable_categorical=True, is_regression=True)
        self.metadata = metadata

    @staticmethod
    def _evaluate(model: BaseModel, data_split: DataSplit, is_regr: bool = False) -> float:
        train_x, test_x = data_split.train_x, data_split.test_x
        try:
            model.fit(train_x, data_split.train_target)
            predictions = model.predict(test_x)
            if is_regr:
                score = mean_absolute_error(data_split.test_target, predictions)
                # from sklearn import metrics
                # score = metrics.r2_score(data_split.test_target, predictions)
            else:
                score = accuracy_score(data_split.test_target.astype('category').cat.codes, np.round(predictions))
        except Exception as e:
            print(e)
            score = 0
        return score

    def evaluate(self,
                 predictions: List[str], dataframe_names: List[Path],
                 do_baseline: bool = False
                 ) -> Dict[str, Dict[str, float]]:
        if len(predictions) != len(dataframe_names):
            raise ValueError(f"Expected num. predictions and num. dataframes to be the same, {len(predictions)} != {len(dataframe_names)}")

        accuracies = defaultdict(dict)
        for dataframe_predictions, dataframe_name in zip(predictions, dataframe_names):
            df_meta = self.metadata[(self.metadata['ref'].str.replace("/", "__") + '.csv') == dataframe_name]#[0]
            try:
                is_regr = (df_meta['task_type'] == 'regression').item()
            except:
                pass
            if is_regr:
                model = self.models_regression
            else:
                model = self.models_classification
            # for model in self.models:
            # if dataframe_name in ['dhanasekarjaisankar__correlation-between-posture-personality-trait.csv',
            #                       'sanskar457__fraud-transaction-detection.csv'
            #                       ]:
            #     continue
            # if dataframe_name in ['shubhamgupta012__titanic-dataset.csv']:
            #     break
            if dataframe_name not in ["arunavakrchakraborty__australia-weather-data.csv"]:  #["iabhishekofficial__mobile-price-classification.csv"]:
                 continue
            data_split = self.data_processor.load_dataframe(dataframe_name)
            print(f"{dataframe_name}: train_x = {data_split.train_x.shape}  test_x = {data_split.test_x.shape}")
            # if data_split.train_x.shape[0] > 5000 or data_split.train_x.shape[1] > 10000:
            #     continue
            prediction = dataframe_predictions[0]   # FIXME: why like this
            if do_baseline:
                data_split = model.baseline_encode(data_split)
            else:
                from .kernels import kernels
                try:
                    prediction = kernels[dataframe_name]
                except:
                    pass

                # from .concrete_strenght_regression import final_kernel

                # train_x, train_y, test_x = final_kernel(data_split.train_x, data_split.train_target, data_split.test_x)
                # data_split = DataSplit(test_x, data_split.test_target, train_x, data_split.train_target)

                # data_split = do_concrete_strength_regression_kernel_studies(data_split, self.data_processor, prediction, model)
                data_split = do_kernel_studies_weather(data_split, self.data_processor, prediction, model)

                # data_split = self.data_processor.transform_dataframe(prediction, data_split)
                # from copy import deepcopy
                # bckp_data_split = deepcopy(data_split)
                #
                # for i in range(7):
                #     data_split = bckp_data_split
                #     train_x, train_y, test_x = data_split.train_x, data_split.train_target, data_split.test_x
                #     do_these = tuple((i,))
                #     try:
                #         train_x, train_y, test_x = transform(train_x, train_y, test_x, do_these=do_these)
                #     except Exception as e:
                #         print(e)
                #     new_ds = DataSplit(test_x, data_split.test_target, train_x, train_y)
                #     accuracy = Evaluator._evaluate(model, new_ds)
                #     print(f"Do {do_these} => {accuracy}")

            accuracy = Evaluator._evaluate(model, data_split, is_regr)
            accuracies[dataframe_name][model.model_slug + ("_MAE" if is_regr else '_ACC')] = accuracy
        self.log_result(f"{dataframe_name}: {accuracies}")
        if do_baseline:
            # Saving the defaultdict to a file
            import pickle
            with open(
                    "/Users/mpietruszka/Repos/ds-f-eng/auto-feature-engineering/tmp_jsonlines/baselines_scores3.pickle",
                    'wb') as handle:
                pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return accuracies

    def log_result(self, log_result: str):
        print(log_result)
        with open(self.logging_path, 'at') as fh:
            fh.write(log_result + '\n')


#######   #######   #######   #######   #######   #######
#######   #######   #######   #######   #######   #######
def do_concrete_strength_regression_kernel_studies(data_split_pre, data_processor, prediction, model, do_study=True):
    from .concrete_strenght_regression import transform_data, transform_data_products, transform_data_best_features
    bckp_data_split = deepcopy(data_split_pre)
    is_regr=True
    prods_that_help = [(0), (4, 31), (4, 30, 33), (4, 20, 33, 47), (4, 25, 30, 31), (4, 30, 31, 33), (4, 30, 33, 44), (4, 31, 33, 44)]
    if do_study:
        results = {}
        improved = []
        improved_prods = []
        for prod_j in prods_that_help:
            for i in range(80):
                data_split = deepcopy(bckp_data_split)
                prod_j = (4, 30, 33, 44)
                do_these = (9, 43)#, 63)
                train_x, train_y, test_x = data_split.train_x, data_split.train_target, data_split.test_x
                try:
                    train_x, train_y, test_x = transform_data(data_split.train_x, data_split.train_target,
                                                              data_split.test_x, do_these=do_these)
                    train_x, train_y, test_x = transform_data_products(data_split.train_x, data_split.train_target,
                                                              data_split.test_x, do_these=prod_j)
                    # train_x, train_y, test_x = transform(train_x, train_y, test_x, do_these=(i,))
                except Exception as e:
                    print(e)
                new_ds = DataSplit(test_x, data_split.test_target, train_x, train_y)
                accuracy = Evaluator._evaluate(model, new_ds, is_regr=is_regr)
                print(f"Do {i}, prod {prod_j} => {accuracy}")
                results[i] = accuracy
                if accuracy > results[0]:       # xhange sign according to is_regr
                    improved.append(i)
                    improved_prods.append(prod_j)

        # Assuming `improved` is already populated
        all_combinations = generate_all_combinations(improved)
        for prod_j in prods_that_help[1:]:
            for combo in all_combinations:
                if len(combo)<7:
                    continue
                data_split = deepcopy(bckp_data_split)
                train_x, train_y, test_x = data_split.train_x, data_split.train_target, data_split.test_x
                do_these = combo
                try:
                    train_x, train_y, test_x = transform_data(data_split.train_x, data_split.train_target,
                                                              data_split.test_x, do_these=do_these)
                    train_x, train_y, test_x = transform_data_products(data_split.train_x, data_split.train_target,
                                                              data_split.test_x, do_these=prod_j)
                except Exception as e:
                    print(e)
                new_ds = DataSplit(test_x, data_split.test_target, train_x, train_y)
                accuracy = Evaluator._evaluate(model, new_ds, is_regr=is_regr)
                print(f"Do {do_these}, prod {prod_j} => {accuracy}")
                results[(do_these, prod_j)] = accuracy

                if accuracy > results[0]:
                    improved.append(do_these)

    # data_split_tfed = DataSplit(test_x, data_split.test_target, train_x, train_y)
    # accuracy = Evaluator._evaluate(model, data_split_tfed)
    print(f"accuracy: {accuracy}")


def do_kernel_studies_weather(data_split_pre, data_processor, prediction, model, do_study=True):
    from .kernel_weather_forecast import transform_data, transform_data_products, transform_data_divs
    bckp_data_split = deepcopy(data_split_pre)

    if do_study:
        results = {}
        improved = []
        for i in range(0*22):
            data_split = deepcopy(bckp_data_split)
            train_x, train_y, test_x = data_split.train_x, data_split.train_target, data_split.test_x
            try:
                # transform data: 1,15,16,21: 0.8574612475897128
                train_x, train_y, test_x = transform_data(data_split.train_x, data_split.train_target,
                                                          data_split.test_x, do_these=(1,15,16,21,))
                train_x, train_y, test_x = transform_data_products(train_x, train_y,
                                                          test_x, do_these=(i,))
                # train_x, train_y, test_x = transform(train_x, train_y, test_x, do_these=(i,))
            except Exception as e:
                print(e)
            new_ds = DataSplit(test_x, data_split.test_target, train_x, train_y)
            accuracy = Evaluator._evaluate(model, new_ds)
            print(f"Do {i} => {accuracy}")
            results[i] = accuracy
            if accuracy > results[0]:
                improved.append(i)
        results[0] = 0.8574612475897128
        improved = [25, 60, 70, 77, 185, 251, 260, 283, 284, 299, 324, 327, 329, 377, 426, 429, 437, 445]
        best_cands = [(60, 185), (70, 77), (77, 445), (25, 251, 284), (25, 260, 283), (25, 284, 426)]
        new_cands = []
        for i in range(len(best_cands) - 1):
            for j in range(i + 1, len(best_cands)):
                new_cands.append(tuple(set(best_cands[i]).union(set(best_cands[j]))))
                # new_cand = set(best_cands[i]) + set(best_cands[j])
        # Assuming `improved` is already populated
        top_new_cand = (77, 70, 445)
        for i in range(1000):
            data_split = deepcopy(bckp_data_split)
            train_x, train_y, test_x = data_split.train_x, data_split.train_target, data_split.test_x
            try:
                # train_x, train_y, test_x = transform_data(train_x, train_y, test_x, do_these=do_these)
                train_x, train_y, test_x = transform_data(data_split.train_x, data_split.train_target,
                                                          data_split.test_x, do_these=(1, 15, 16, 21,))
                train_x, train_y, test_x = transform_data_products(train_x, train_y,
                                                                   test_x, do_these=top_new_cand)
                train_x, train_y, test_x = transform_data_divs(train_x, train_y,
                                                               test_x, do_these=(i,))
            except Exception as e:
                print(e)
            new_ds = DataSplit(test_x, data_split.test_target, train_x, train_y)
            accuracy = Evaluator._evaluate(model, new_ds)
            print(f"Do {top_new_cand}, div({i}) => {accuracy}")
            results[i] = accuracy
            if accuracy > results[0]:
                improved.append(i)

    # data_split_tfed = DataSplit(test_x, data_split.test_target, train_x, train_y)
    # accuracy = Evaluator._evaluate(model, data_split_tfed)
    print(f"accuracy: {accuracy}")


class ScoreNormalizer:
    def __init__(self, baseline_results):
        self.baseline_results = baseline_results
        self.non_answer_penalty = -100

    def __call__(self, predictions):
        normalized_scores = []
        for dataset_name, baseline_result in self.baseline_results.items():
            metric_name, baseline_score = list(baseline_result.items())[0]
            try:
                predicted_score = predictions[dataset_name][metric_name]
            except KeyError:
                print(f"Missing scores for {dataset_name}")
                normalized_scores.append(self.non_answer_penalty)
                continue
            # _, predicted_score = list(prediction_v.items())[0]
            is_regr = metric_name.endswith("MAE")
            if is_regr:
                normalized_score = error_rate_normalizer_mae(baseline_score, predicted_score)
            else:
                normalized_score = error_rate_normalizer_acc(baseline_score, predicted_score)
            normalized_scores.append(normalized_score)
        return sum(normalized_scores)/len(normalized_scores)


def error_rate_normalizer_mae(baseline_score: float, predicted_score: float) -> float:
    # 2.0 -> 0.5 , should be 0.75
    # 1.0 -> 0.25 , should be 0.75
    error_rate_reduction = 1.0 - (predicted_score/baseline_score)
    return error_rate_reduction


def error_rate_normalizer_acc(baseline_score: float, predicted_score: float) -> float:
    # 0.6 -> 0.9 , should be 0.75
    # 0.9 -> 0.975 , should be 0.75
    baseline_error_rate = 1.0 - baseline_score
    predicted_error_rate = 1.0 - predicted_score
    error_rate_reduction = (baseline_error_rate-predicted_error_rate)/baseline_error_rate
    return error_rate_reduction


##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####


def do_kernel_studies_personality(data_split_pre, data_processor, prediction, model, do_study=False):
    data_split_new = data_processor.transform_dataframe(prediction, data_split_pre)
    bckp_data_split = deepcopy(data_split_new)
    if do_study:
        results = {}
        improved = []
        for i in range(23):
            data_split = deepcopy(bckp_data_split)
            train_x, train_y, test_x = data_split.train_x, data_split.train_target, data_split.test_x
            try:
                train_x, train_y, test_x = transform(train_x, train_y, test_x, do_these=(i,))
            except Exception as e:
                print(e)
            new_ds = DataSplit(test_x, data_split.test_target, train_x, train_y)
            accuracy = Evaluator._evaluate(model, new_ds)
            print(f"Do {i} => {accuracy}")
            results[i] = accuracy
            if accuracy > results[0]:
                improved.append(i)

        # Assuming `improved` is already populated
        all_combinations = generate_all_combinations(improved)
        for combo in all_combinations:
            data_split = deepcopy(bckp_data_split)
            train_x, train_y, test_x = data_split.train_x, data_split.train_target, data_split.test_x
            do_these = combo
            try:
                train_x, train_y, test_x = transform(train_x, train_y, test_x, do_these=do_these)
            except Exception as e:
                print(e)
            new_ds = DataSplit(test_x, data_split.test_target, train_x, train_y)
            accuracy = Evaluator._evaluate(model, new_ds)
            print(f"Do {do_these} => {accuracy}")
            results[do_these] = accuracy
            if accuracy > results[0]:
                improved.append(do_these)
    return data_split_new


def generate_all_combinations(improved_list):
    all_combinations = []
    # Generate combinations for all lengths from 1 to the length of the list
    for r in range(2, len(improved_list) + 1):
        all_combinations.extend(combinations(improved_list, r))
    return all_combinations
#######   #######   #######   #######   #######   #######   #######   #######   #######   #######   #######   #######
#######   #######   #######   #######   #######   #######   #######   #######   #######   #######   #######   #######


import pandas as pd
import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelEncoder

## 6-


# Example usage:
# Assuming train_x, train_y, and test_x are properly defined pandas DataFrame and Series
# train_x_transformed, train_y_transformed, test_x_transformed = transform(train_x, train_y, test_x)
