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
from tqdm import tqdm

import bigcode_eval.tasks.custom_metrics.ds_f_eng_kernel_studies
import pandas as pd
import numpy as np
from bigcode_eval.tasks.custom_metrics.ds_f_eng_kernel_studies import transform
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from tabpfn import TabPFNClassifier
from pathlib import Path

pd.options.mode.copy_on_write = True

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
        # Convert each object column to category type
        try:
            object_cols = df.select_dtypes(include=['object']).columns
            df[object_cols] = df[object_cols].astype('category')
        except:
            pass
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
    def __init__(self, dataframe_global_path: Path, timeout: int = 60, allow_no_transform: bool = False, target_column: str = None) -> None:
        self.timeout = timeout
        self.allow_no_transform = allow_no_transform
        self.dataframe_global_path = dataframe_global_path
        self.split_column = "ds_f_eng__split"
        self.train_split = "train"
        self.test_split = "test"
        self.target_column = target_column
        self.unnamed = 'Unnamed: 0'

    def load_dataframe(self, dataframe_name: Path, df_meta) -> DataSplit:
        dataframe_path = self.dataframe_global_path / dataframe_name
        df = pd.read_csv(dataframe_path) if dataframe_path.suffix == ".csv" else pd.read_csv(str(dataframe_path) + ".csv")
        train_dataframe = df[df[self.split_column] == self.train_split]
        test_dataframe = df[df[self.split_column] == self.test_split]
        target_col = df_meta.target_col.values[0] if not isinstance(df_meta.target_col, str) else df_meta.target_col
        return self.prepare_train_test_split(test_dataframe, train_dataframe, target_col)

    def prepare_train_test_split(self, test_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame, target_col_name) -> DataSplit:
        test_dataframe.dropna(subset=[target_col_name], inplace=True)
        train_dataframe.dropna(axis=1, how='all', inplace=True)
        test_dataframe = test_dataframe[train_dataframe.columns]
        train_dataframe = train_dataframe.dropna(subset=[target_col_name])
        ##
        train_target = train_dataframe[target_col_name]
        test_target = test_dataframe[target_col_name]
        train_x = self._remove_columns(train_dataframe, columns_to_remove={target_col_name, self.split_column})
        test_x = self._remove_columns(test_dataframe, columns_to_remove={target_col_name, self.split_column})
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
        transformed_ds = self._transform_dataframe_inplace_three_tuple(prediction, data_split)
        if transformed_ds is None:
            return data_split
        else:
            return transformed_ds

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df.columns = df.columns.str.replace('[', '(', regex=False)
        except:
            pass
        try:
            df.columns = df.columns.str.replace(']', ')', regex=False)
        except:
            pass
        try:
            df.columns = df.columns.str.replace('<', 'lesser', regex=False)
        except:
            pass
        try:
            df.columns = df.columns.str.replace('>', 'greater', regex=False)
        except:
            pass
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
        try:
            train_x, test_x = data_split.train_x, data_split.test_x
            model.fit(train_x, data_split.train_target)
            predictions = model.predict(test_x)
            if is_regr:
                score = mean_absolute_error(data_split.test_target, predictions)
            else:
                score = accuracy_score(data_split.test_target.astype('category').cat.codes, np.round(predictions))
        except Exception as e:
            print('Exception', e)
            score = 0 if not is_regr else float("inf")
        return score

    def evaluate(self,
                 predictions: List[str], dataframe_names: List[Path],
                 do_baseline: bool = False
                 ) -> Dict[str, Dict[str, float]]:
        if len(predictions) != len(dataframe_names):
            raise ValueError(f"Expected num. predictions and num. dataframes to be the same, {len(predictions)} != {len(dataframe_names)}")

        accuracies = defaultdict(dict)
        for dataframe_predictions, dataframe_name in tqdm(zip(predictions, dataframe_names)):
            df_meta = self.metadata[(self.metadata['ref'].str.replace("/", "__") + '.csv') == dataframe_name]

            is_regr = (df_meta['task_type'] == 'regression').item()
            if is_regr:
                model = self.models_regression
            else:
                model = self.models_classification

            data_split = self.data_processor.load_dataframe(dataframe_name, df_meta)
            # print(f"{dataframe_name}: train_x = {data_split.train_x.shape}  test_x = {data_split.test_x.shape}")
            prediction = dataframe_predictions[0]   # FIXME: why like this

            if do_baseline:
                data_split_transformed = model.baseline_encode(data_split)
            else:
                data_split_transformed = self.data_processor.transform_dataframe(
                    prediction=prediction,
                    data_split=data_split
                )
                data_split_transformed = model.baseline_encode(data_split_transformed)
                
                # if not all(data_split_transformed.train_target == data_split.train_target):
                #     import pdb
                #     pdb.set_trace()

                # if data_split.test_target.shape == data_split_transformed.test_target.shape:
                #     data_split_transformed = DataSplit(
                #         test_x=data_split_transformed.test_x,
                #         test_target=data_split.test_target,
                #         train_x=data_split_transformed.train_x,
                #         train_target=data_split.train_target
                #     )
                # else:
                #     import pdb
                #     pdb.set_trace()
                # data_split_transformed.train_target = data_split.train_target
            accuracy = Evaluator._evaluate(model, data_split_transformed, is_regr)
            accuracies[dataframe_name][model.model_slug + ("_MAE" if is_regr else '_ACC')] = accuracy
            self.log_result(f"{dataframe_name}: {accuracy}")

        return accuracies

    def log_result(self, log_result: str):
        with open(self.logging_path, 'at') as fh:
            fh.write(log_result + '\n')


class ScoreNormalizer:
    def __init__(self, baseline_results):
        self.baseline_results = baseline_results
        self.non_answer_penalty = -1

    def __call__(self, predictions):
        normalized_scores = []
        self.baseline_results = {k: v for k, v in self.baseline_results.items() if k in predictions.keys()}
        for dataset_name, baseline_result in self.baseline_results.items():
            metric_name, baseline_score = list(baseline_result.items())[0]
            try:
                predicted_score = predictions[dataset_name][metric_name]
            except KeyError:
                print(f"Missing scores for {dataset_name}")
                normalized_scores.append(self.non_answer_penalty)
                continue

            is_regr = metric_name.endswith("MAE")
            if is_regr:
                normalized_score = error_rate_normalizer_mae(baseline_score, predicted_score)
            else:
                normalized_score = error_rate_normalizer_acc(baseline_score, predicted_score)

            normalized_scores.append(normalized_score)

        return sum(normalized_scores)/len(normalized_scores)


def error_rate_normalizer_mae(baseline_score: float, predicted_score: float) -> float:
    # 2.0 -> 0.5, should be 0.75
    # 1.0 -> 0.25, should be 0.75
    # 1.0 -> 2.1, should be -1
    # 1.0 -> 1.1, should be -0.1
    error_rate_reduction = 1.0 - predicted_score/(baseline_score + 1e-16)
    return max(error_rate_reduction, 0)


def error_rate_normalizer_acc(baseline_score: float, predicted_score: float) -> float:
    # 0.6 -> 0.9 , should be 0.75
    # 0.9 -> 0.975 , should be 0.75
    baseline_error_rate = 1.0 - baseline_score
    predicted_error_rate = 1.0 - predicted_score
    error_rate_reduction = (baseline_error_rate-predicted_error_rate)/(baseline_error_rate + 1e-16)
    return max(error_rate_reduction, 0)

