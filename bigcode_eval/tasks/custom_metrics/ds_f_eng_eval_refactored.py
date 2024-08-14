import os
from collections import defaultdict
from typing import List, Dict, Set, Union
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from tabpfn import TabPFNClassifier
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from .execute import unsafe_execute

_WARNING = """
# WARNING: Untrusted code execution. Read disclaimer and set the appropriate environment variable to continue.
"""

class DataProcessor:
    def __init__(self, dataframe_global_path: Path):
        self.dataframe_global_path = dataframe_global_path
        self.split_column = "ds_f_eng__split"
        self.train_split = "train"
        self.test_split = "test"
        self.target_column = "ds_f_eng__target__response"
        self.unnamed = 'Unnamed: 0'

    def load_dataframe(self, dataframe_name: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        dataframe_path = self.dataframe_global_path / dataframe_name
        if dataframe_path.suffix == ".csv":
            df = pd.read_csv(dataframe_path)
        else:
            df =  pd.read_csv(str(dataframe_path) + ".csv")
        train_dataframe = df[df[self.split_column] == self.train_split]
        test_dataframe = df[df[self.split_column] == self.test_split]
        test_x, test_target, train_x, train_target = self.prepare_train_test_split(test_dataframe, train_dataframe)

        return test_x, test_target, train_x, train_target

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
    def identify_categorical_columns_new(df: pd.DataFrame, unique_val_threshold: int) -> List[str]:
        # Try to convert columns to numeric, track the ones that can't be converted
        non_numeric_columns = []
        for col in df.columns:
            # Attempt to convert to numeric
            try:
                pd.to_numeric(df[col])
            except ValueError:
                non_numeric_columns.append(col)

        # Now apply the existing logic for identifying categorical columns
        cat_cols_by_dtype = df[non_numeric_columns].select_dtypes(include=['object', 'category']).columns
        cat_cols_by_uniq = [col for col in df[non_numeric_columns].columns if df[col].nunique() < unique_val_threshold]

        # Combine both criteria
        return list(set(cat_cols_by_uniq) | set(non_numeric_columns))
        # return cat_cols_by_uniq, non_numeric_columns   #list(set(cat_cols_by_dtype) | set(cat_cols_by_uniq))

    @staticmethod
    def one_hot_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_columns: List[str], fit_on_combined=True):
        if fit_on_combined:
            combined_df = pd.concat([train_df[categorical_columns], test_df[categorical_columns]])
            onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='infrequent_if_exist').fit(combined_df.astype(str))
        else:
            onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='infrequent_if_exist').fit(train_df[categorical_columns].astype(str))
        train_oh = pd.DataFrame(onehot_encoder.transform(train_df[categorical_columns]), columns=onehot_encoder.get_feature_names_out())
        test_oh = pd.DataFrame(onehot_encoder.transform(test_df[categorical_columns]), columns=onehot_encoder.get_feature_names_out())
        return train_oh, test_oh

    def prepare_train_test_split(self, test_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame):
        test_dataframe = test_dataframe.dropna(subset=[self.target_column])  # , inplace=True)
        # Remove columns where all values are NaN
        train_dataframe.dropna(axis=1, how='all', inplace=True)
        test_dataframe.dropna(axis=1, how='all', inplace=True)
        # fill missing values
        # train_dataframe.fillna('missing', inplace=True)
        # test_dataframe.fillna('missing', inplace=True)
        # missing
        train_target = train_dataframe[self.target_column]
        test_target = test_dataframe[self.target_column]
        train_features = self._remove_columns(train_dataframe, columns_to_remove={self.target_column, self.split_column})
        test_features = self._remove_columns(test_dataframe, columns_to_remove={self.target_column, self.split_column})
        # Remove rows where a specific column has NaN values
        return test_features, test_target, train_features, train_target

    @staticmethod
    def _transform_dataframe_inplace(prediction: str, dataframe: pd.DataFrame, timeout: float,
                                     allow_no_transform: bool) -> Union[pd.DataFrame, None]:
        if len(prediction) == 0 and allow_no_transform:
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
            timeout=timeout,
            exec_globals=exec_globals,
        )
        if result.pop() != "passed":
            return None
        else:
            return exec_globals["dataframe_transformed"]


class LongProcessor:

    def __init__(self):
        pass
    def _process_long_test(self, test_x, classifier, max_test_size=10, max_workers=1) -> List[str]:
        def process_batch(start_position):
            batch_test = test_x[start_position:start_position + max_test_size]
            return classifier.predict(batch_test)

        predictions = []
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(process_batch, start_pos): start_pos for start_pos in
                                   range(0, len(test_x), max_test_size)}
                for future in as_completed(future_to_batch):
                    predictions.extend(future.result())
        else:
            for start_position in range(0, len(test_x), max_test_size):
                batch_test = test_x[start_position:start_position + max_test_size]
                predictions.extend(classifier.predict(batch_test))
        return predictions


class ModelTrainer(LongProcessor):
    def __init__(self, enable_categorical=False):
        super().__init__()
        self.enable_categorical = enable_categorical

    def xgb_fit_predict(self, test_df: pd.DataFrame, train_df: pd.DataFrame, train_target: pd.Series, categorical_columns=None):
        if self.enable_categorical and categorical_columns:
            train_df[categorical_columns] = train_df[categorical_columns].apply(lambda col: col.astype('category'))
            test_df[categorical_columns] = test_df[categorical_columns].apply(lambda col: col.astype('category'))
        xgb_classifier = xgb.XGBModel(enable_categorical=self.enable_categorical, num_class=len(train_target.unique()))
        xgb_classifier.fit(train_df, train_target.astype('category').cat.codes)
        return xgb_classifier.predict(test_df)

    def tabpfn_fit_predict(self, train_features, train_target, test_features, test_target, max_test_size=10):
        classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
        classifier.fit(train_features, train_target)
        return self._process_long_test(test_features, classifier, max_test_size)

    def get_accuracy(self, train_features: pd.DataFrame, test_features: pd.DataFrame, train_target: pd.Series,
                     test_target: pd.Series, model_type=("tabpfn", "xgboost")
                     ):
        categorical_columns = None
        if self.enable_categorical:
            # data_processor = DataProcessor(None)  # No global path needed
            categorical_columns = DataProcessor.identify_categorical_columns(train_features, unique_val_threshold=5)
            train_features, test_features = DataProcessor.one_hot_encode(train_features, test_features, categorical_columns)

        scores = {}
        if "tabpfn" in model_type:
            scores["tabpfn"] = accuracy_score(test_target,
                                              self.tabpfn_fit_predict(train_features, train_target, test_features,
                                                                      test_target))
        if "xgboost" in model_type:
            test_target_hat = self.xgb_fit_predict(test_features, train_features, train_target, categorical_columns=categorical_columns)
            scores["xgboost"] = accuracy_score(test_target.astype('category').cat.codes, np.round(test_target_hat))
        return scores


class BaselineEvaluator(LongProcessor):
    def __init__(self, enable_categorical=True):
        super().__init__()
        self.enable_categorical = enable_categorical
        self.model_trainer = ModelTrainer(enable_categorical)
        self.max_test_size = 100
        self.min_val_size = 100
        self.cross_validation_train_size = 900  # int((len(train_features) - min_val_size)/num_train_seeds)
        self.tabpfn_train_len = 900

    def evaluate(self, test_df: pd.DataFrame, test_target: pd.Series, train_df: pd.DataFrame, train_target: pd.Series)\
            -> dict[str, float]:
        scores = {}
        try:
            tabpfn_outs = self.model_trainer.tabpfn_fit_predict(train_df, train_target, test_df, test_target, max_test_size=100)
            scores['tabpfn'] = accuracy_score(test_target, tabpfn_outs)
            #accuracy_score(test_target.astype('category').cat.codes, tabpfn_outs)
        except Exception as e:
            scores['tabpfn'] = -1.0


        return scores

    def xgb_evaluate(self, test_df, test_target, train_df, train_target) -> float:
        try:
            xgboost_out = self.model_trainer.xgb_fit_predict(test_df, train_df, train_target)
            xgboost_score = accuracy_score(test_target.astype('category').cat.codes, np.round(xgboost_out))
        except Exception as e:
            xgboost_score = -1.0
        return xgboost_score

    # def get_baseline_accuracy(self, test_x, test_target, train_x, train_target) -> dict[str, float]:
    #     data_processor = DataProcessor(None)  # No global path needed
    #     if self.enable_categorical:
    #         categorical_columns = DataProcessor.identify_categorical_columns(train_x, unique_val_threshold=5)
    #         train_x_oh, test_x_oh = DataProcessor.one_hot_encode(train_x, test_x, categorical_columns)
    #
    #     return self.evaluate(test_x, test_target, train_x, train_target)

    def preprocess_categorical_columns_old(self, test_x, train_x):
        if self.enable_categorical:
            # Identify categorical columns
            categorical_columns, non_numeric_columns = DataProcessor.identify_categorical_columns_new(train_x, unique_val_threshold=5)

            # One-hot encode the categorical columns
            train_x_oh, test_x_oh = DataProcessor.one_hot_encode(train_x, test_x, categorical_columns)

            # Drop the original categorical columns from the numerical dataset
            columns = categorical_columns
            train_x_numerical = train_x.drop(columns=non_numeric_columns)
            test_x_numerical = test_x.drop(columns=non_numeric_columns)

            # Concatenate the numerical and one-hot encoded features
            train_x_combined = pd.concat([train_x_numerical.reset_index(drop=True), train_x_oh.reset_index(drop=True)], axis=1)
            test_x_combined = pd.concat([test_x_numerical.reset_index(drop=True), test_x_oh.reset_index(drop=True)], axis=1)
            return test_x_combined, train_x_combined
        else:
            return test_x, train_x

    def preprocess_categorical_columns(self, test_x, train_x):
        if self.enable_categorical:
            # Identify categorical columns
            categorical_columns, non_numeric_columns = DataProcessor.identify_categorical_columns_new(train_x,
                                                                                                      unique_val_threshold=5)

            # Handle missing values
            for col in train_x.columns:
                if col in non_numeric_columns:
                    train_x[col].fillna('missing', inplace=True)
                    test_x[col].fillna('missing', inplace=True)
                else:
                    train_x[col].fillna(train_x[col].mean(), inplace=True)
                    test_x[col].fillna(test_x[col].mean(), inplace=True)

            # One-hot encode the categorical columns
            train_x_oh, test_x_oh = DataProcessor.one_hot_encode(train_x, test_x, categorical_columns)

            # Drop the original categorical columns from the numerical dataset
            train_x_numerical = train_x.drop(columns=non_numeric_columns)
            test_x_numerical = test_x.drop(columns=non_numeric_columns)

            # Concatenate the numerical and one-hot encoded features
            train_x_combined = pd.concat([train_x_numerical.reset_index(drop=True), train_x_oh.reset_index(drop=True)],
                                         axis=1)
            test_x_combined = pd.concat([test_x_numerical.reset_index(drop=True), test_x_oh.reset_index(drop=True)],
                                        axis=1)
            return test_x_combined, train_x_combined
        else:
            return test_x, train_x

    def get_baseline_accuracy(self, test_x, test_target, train_x, train_target) -> dict[str, float]:
        test_x, train_x = self.preprocess_categorical_columns(test_x, train_x)

        tabpfn_score = self.tabpfn_cross_validation(test_x, test_target, train_x, train_target, max_train_seeds=3)
        xgboost_score = self.xgb_evaluate(test_x, test_target, train_x, train_target)
        scores = {"tabpfn_score": tabpfn_score,
                  "xgboost_score": xgboost_score}
        return scores       # = self.evaluate(test_x, test_target, train_x, train_target)

    def tabpfn_cross_validation(self, test_x, test_target, train_x, train_target, max_train_seeds=3) -> float:
        num_train_seeds = min(max_train_seeds, int(np.ceil(len(train_x) / self.tabpfn_train_len)))
        seed_i = 0
        if num_train_seeds <= 2:
            # Initialize and train the classifier
            classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
            classifier.fit(train_x[:self.tabpfn_train_len], train_target[:self.tabpfn_train_len])
        else:
            best_cv_score = -1.0
            best_classifier = None
            val_subset = train_x[:self.min_val_size]
            val_subset_target = train_target[:self.min_val_size]


            # Loop over the number of calculated training subsets, and validate on rest
            num_classifiers_fitted = 0
            while seed_i < num_train_seeds:
                print(f"{seed_i} / {num_train_seeds}")

                # for seed_i in range(num_train_seeds):
                started_pos = seed_i * self.cross_validation_train_size + self.min_val_size
                stop_pos = min(started_pos + self.cross_validation_train_size,
                               len(train_x))  # Ensure we do not go out of bounds
                # Slice the DataFrame to get the subset for training
                sampled_df = train_x.iloc[started_pos:stop_pos]
                sampled_y = train_target.iloc[started_pos:stop_pos]
                # Initialize and train the classifier
                seed_i += 1
                if num_train_seeds > 10000 and num_classifiers_fitted > 0:
                    break
                try:
                    classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
                    classifier.fit(sampled_df, sampled_y)
                    # Assume process_long_test is a function that processes the test data through the classifier
                    cv_target_hat = self._process_long_test(val_subset, classifier, max_test_size=self.max_test_size)
                    cv_acc = accuracy_score(val_subset_target, cv_target_hat)
                    num_classifiers_fitted += 1
                except:  # FIXME: try sampling randomly
                    cv_acc = -1.0
                    num_train_seeds += 1

                if cv_acc > best_cv_score:
                    best_cv_score = cv_acc
                    best_classifier = classifier
                # increment
            classifier = best_classifier

        test_target_hat = self._process_long_test(test_x, classifier, max_test_size=self.max_test_size)
        acc_max = accuracy_score(test_target, test_target_hat)
        return acc_max  # Or return all metrics as needed


class ScoreNormalizer:
    def __init__(self):
        self.baseline_results = {'mirlei__hcc-survival-data-set.csv': 0.6, 'rohit265__breast-cancer-uci-machine-learning.csv': 0.7, 'imsparsh__churn-risk-rate-hackerearth-ml.csv': 0.4}

    def __call__(self, predictions: Dict[str, float]):
        normalized_scores = []
        for prediction_k, prediction_v in predictions.items():
            normalized_score = self.error_rate_normalizer(self.baseline_results.get(prediction_k, 0.0), prediction_v)
            normalized_scores.append(normalized_score)
        return sum(normalized_scores) / len(normalized_scores)

    @staticmethod
    def error_rate_normalizer(baseline_score, test_score):
        baseline_error_rate = 1.0 - baseline_score
        test_error_rate = 1.0 - test_score
        return (baseline_error_rate - test_error_rate) / baseline_error_rate


class AccuracyEvaluator:
    def __init__(self, data_processor: DataProcessor, model_trainer: ModelTrainer, baseline_evaluator: BaselineEvaluator):
        self.data_processor = data_processor
        self.model_trainer = model_trainer
        self.baseline_evaluator = baseline_evaluator

    def evaluate(self,
                 predictions: List[str], dataframe_names: List[Path],
                 timeout: float, log_file: str, do_baseline: bool = False) -> Dict[str, float]:
        if len(predictions) != len(dataframe_names):
            raise ValueError(f"Expected num. predictions and num. dataframes to be the same, {len(predictions)} != {len(dataframe_names)}")

        dataframe_path_to_predictions = defaultdict(list)
        for prediction, dataframe_name in zip(predictions, dataframe_names):
            dataframe_path_to_predictions[dataframe_name].append(prediction)

        df_i = -1
        accuracies = {}
        for dataframe_name, dataframe_predictions in dataframe_path_to_predictions.items():
            df_i += 1
            if df_i < 41:
                continue

            test_x, test_target, train_x, train_target = self.data_processor.load_dataframe(dataframe_name)
            print(f"{dataframe_name}: train_x = {train_x.shape}  test_x = {test_x.shape}")

            # if train_x.shape[0] > 100000 or  train_x.shape[1] > 10000:
            #     continue
            if train_x.shape[0] <= 100000:
                continue
            for prediction in dataframe_predictions:
                if do_baseline:
                    scores_dict = self.baseline_evaluator.get_baseline_accuracy(test_x, test_target, train_x, train_target)
                else:
                    train_features_transformed = DataProcessor._transform_dataframe_inplace(
                        prediction=prediction,
                        dataframe=train_features,
                        timeout=timeout,
                        allow_no_transform=allow_no_transform,
                    )
                    test_features_transformed = _transform_dataframe_inplace(
                        prediction=prediction,
                        dataframe=test_features,
                        timeout=timeout,
                        allow_no_transform=allow_no_transform,
                    )
                    scores_dict = self.model_trainer.get_accuracy(
                        train_features=train_features_transformed,
                        test_features=test_features_transformed,
                        train_target=train_dataframe[target_column],
                        test_target=test_dataframe[target_column],
                        model_type=("tabpfn", "xgboost")
                    )
                accuracies[dataframe_name] = scores_dict
                self.log_result(log_file, f"{dataframe_name}: {scores_dict}")

        return accuracies

    @staticmethod
    def log_result(log_file: str, log_result: str):
        print(log_result)
        with open(log_file, 'at') as fh:
            fh.write(log_result + '\n')
