"""Desc"""

import itertools
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Set, List, Optional, Union
import pickle
from pathlib import Path
from copy import deepcopy
import statistics
import xgboost as xgb

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from concurrent.futures import ThreadPoolExecutor, as_completed

from .execute import unsafe_execute


_CITATION = """"""

_DESCRIPTION = """"""

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
This metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network.

Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:

>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"

################################################################################\
"""

_LICENSE = """"""


def tabpfn_average_accuracy(
    predictions: List[List[str]], 
    dataframe_names: List[Path],
    dataframe_global_path: Path,
    split_column: str,
    train_split: str,
    test_split: str,
    target_column: str, 
    timeout: float,
    log_file: str,
    do_baseline: bool = False,
) -> Dict[str, float]:
    if len(predictions) != len(dataframe_names):
        raise ValueError(f"Expected num. predictions and num. dataframes to be the same, {len(predictions)} != {len(dataframe_names)}")
    dataframe_path_to_predictions = defaultdict(list)
    for prediction, dataframe_name in zip(predictions, dataframe_names):
        if len(prediction) != 1:
            raise ValueError
        dataframe_path_to_predictions[dataframe_name].append(prediction.pop())
    accuracies = {}
    for dataframe_name, dataframe_predictions in dataframe_path_to_predictions.items():
        dataframe = load_dataframe(dataframe_path=dataframe_global_path/dataframe_name)
        train_dataframe = dataframe[dataframe[split_column] == train_split]
        test_dataframe = dataframe[dataframe[split_column] == test_split]
        for prediction in dataframe_predictions:
            if do_baseline:
                scores_dict = get_baseline_accuracy(target_column, test_dataframe, train_dataframe, enable_categorical=True)
                log_result = f"{dataframe_name}: {scores_dict}"
                print(log_result)
                with open(log_file, 'at') as fh:
                    fh.write(log_result + '\n')
            else:
                scores_dict = \
                    get_accuracy(
                        prediction=prediction,
                        train_dataframe=remove_dataframe_columns(train_dataframe, columns_to_remove={split_column}),
                        test_dataframe=remove_dataframe_columns(test_dataframe, columns_to_remove={split_column}),
                        target_column=target_column,
                        timeout=timeout,
                    )
                accuracies[dataframe_name] = scores_dict
            log_result = f"{dataframe_name}: {scores_dict}"
            print(log_result)
            with open(log_file, 'at') as fh:
                fh.write(log_result + '\n')

    return accuracies


def load_dataframe(dataframe_path: Path) -> pd.DataFrame:
    if dataframe_path.suffix == ".csv":
        return pd.read_csv(dataframe_path)
    else:
        return pd.read_csv(dataframe_path + ".csv")


def remove_dataframe_columns(dataframe: pd.DataFrame, columns_to_remove: Set[str]) -> pd.DataFrame:
    return dataframe[[column for column in dataframe.columns if column not in columns_to_remove]]


def process_long_test(test_features_transformed, classifier, max_test_size=10):
    predictions = []
    for start_position in range(0, len(test_features_transformed), max_test_size):
        batch_test = test_features_transformed[start_position:start_position + max_test_size]
        test_target_hat = classifier.predict(batch_test)
        predictions.extend(test_target_hat)
    return predictions


def identify_categorical_columns(df, unique_val_threshold):
    # Using both data type and unique value count criteria
    cat_cols_by_dtype = df.select_dtypes(include=['object', 'category']).columns
    cat_cols_by_uniq = [col for col in df.columns if df[col].nunique() < unique_val_threshold]
    categorical_cols = set(cat_cols_by_dtype) | set(cat_cols_by_uniq)  # Union of both criteria
    return list(categorical_cols)


def get_baseline_score(test_df, test_target, train_df, train_target, enable_categorical=True):
    # get tabpfn
    categorical_columns = identify_categorical_columns(train_df, unique_val_threshold=5)
    try:
        if enable_categorical:
            train_df_oh = train_df
            test_df_oh = test_df
            train_df_oh, test_df_oh = one_hot_encode(train_df_oh, test_df_oh, categorical_columns)
        tabpfn_score = run_baseline_tabpfn_with_cross_validation(test_df_oh, test_target, train_df_oh, train_target)
    except:
        tabpfn_score = 0.0

    try:
        test_target_hat = xgb_fit_predict(categorical_columns, test_df, train_df, train_target)
        xgboost_score = accuracy_score(test_target.astype('category').cat.codes, np.round(test_target_hat))
    except:
        xgboost_score = 0.0
    scores = {"tabpfn": tabpfn_score, "xgboost": xgboost_score}
    if tabpfn_score != 0.0 or xgboost_score != 0.0:
        best_model = 'tabpfn' if tabpfn_score > xgboost_score else 'xgboost'
        return max(tabpfn_score, xgboost_score), best_model, scores
    else:
        return 0.0, -1, {}


def xgb_fit_predict(test_df, train_df, train_target, enable_categorical=False, categorical_columns=None):
    if enable_categorical:
        train_df[categorical_columns] = train_df[categorical_columns].apply(lambda col: col.astype('category'))
        test_df[categorical_columns] = test_df[categorical_columns].apply(lambda col: col.astype('category'))
    xgb_classifier = xgb.XGBModel(enable_categorical=enable_categorical, num_class=len(train_target.unique()))
    xgb_classifier.fit(train_df, train_target.astype('category').cat.codes)
    test_target_hat = xgb_classifier.predict(test_df)
    return test_target_hat


def get_baseline_accuracy(target_column, test_dataframe, train_dataframe, enable_categorical=True):
    test_features, test_target, train_features, train_target = prepare_train_test_split(target_column, test_dataframe, train_dataframe)
    baseline_score, baseline_type, scores_dict = get_baseline_score(test_features, test_target, train_features, train_target, enable_categorical)
    return scores_dict


def get_accuracy(
    prediction: str, 
    train_dataframe: pd.DataFrame, 
    test_dataframe: pd.DataFrame, 
    target_column: str, 
    timeout: float,
    allow_no_transform: bool = True,
    N_ensemble_configurations: int = 32,
    preffered_model = ("tabpfn", "xgboost"),
    enable_categorical=False,
) -> dict[str, float]:
    test_features, test_target, train_features, train_target = prepare_train_test_split(target_column, test_dataframe, train_dataframe)

    # TODO(ajedrosz): handle potentially modified order of samples, e.g. id column that's not given in prompt header
    train_features_transformed = _transform_dataframe_inplace(
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
    # TODO(ajedrosz): what with hparams
    scores = {}
    if enable_categorical:
        # FIXME: this categoricals below should not be enabled as they are to be provided by the model
        categorical_columns = identify_categorical_columns(train_features, unique_val_threshold=5)
        train_features_transformed, test_features_transformed = one_hot_encode(train_features, test_features, categorical_columns)
    else:
        categorical_columns = None

    if "tabpfn" in preffered_model:
        try:
            classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
            classifier.fit(train_features_transformed, train_target)
            test_target_hat = process_long_test(test_features_transformed, classifier, max_test_size=100)
            scores["tabpfn"] = accuracy_score(test_target, test_target_hat)
        except Exception as e:
            print("Error fitting tabpfn on model's output.")
            scores["tabpfn"] = 0.0
    if "xgboost" in preffered_model:
        try:
            test_target_hat = xgb_fit_predict(test_features_transformed, train_features_transformed, train_target,
                                              categorical_columns=categorical_columns, enable_categorical=enable_categorical)
            xgboost_score = accuracy_score(test_target.astype('category').cat.codes, np.round(test_target_hat))
            scores["xgboost"] = xgboost_score
        except:
            print("Error fitting xgboost on model's output.")
            scores["xgboost"] = 0.0
    return scores


def prepare_train_test_split(target_column, test_dataframe, train_dataframe):
    train_target = train_dataframe[target_column]
    test_target = test_dataframe[target_column]
    train_features = remove_dataframe_columns(train_dataframe, columns_to_remove={target_column})
    test_features = remove_dataframe_columns(test_dataframe, columns_to_remove={target_column})

    return test_features, test_target, train_features, train_target


def one_hot_encode(train_df, test_df, categorical_columns):
    # Build one-hot encoder
    fitted_onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='infrequent_if_exist').fit(train_df[categorical_columns])

    # transform data
    train_oh = fitted_onehot_encoder.transform(train_df[categorical_columns])
    test_oh = fitted_onehot_encoder.transform(test_df[categorical_columns])
    # turn back to pandas
    train_oh_df = pd.DataFrame(train_oh, columns=fitted_onehot_encoder.get_feature_names_out())
    test_oh_df = pd.DataFrame(test_oh, columns=fitted_onehot_encoder.get_feature_names_out())

    return train_oh_df, test_oh_df


def baseline_run(test_features_transformed, test_target, train_features, train_target, tabpfn_train_len=900):
    num_train_seeds = np.ceil(len(train_features)/tabpfn_train_len)
    # num_seeds = 10
    # MAX_TRAIN_SAMPLES = 10
    accuracies = []
    for seed_i in range(num_train_seeds):
        started_pos = seed_i * tabpfn_train_len
        stop_pos = (seed_i+1) * tabpfn_train_len
        sampled_df = train_features[started_pos:stop_pos]
        # sampled_df = train_features.sample(MAX_TRAIN_SAMPLES, random_state=42 + seed_i)
        sampled_y = train_target.loc[sampled_df.index]
        classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
        classifier.fit(sampled_df, sampled_y)
        test_target_hat = process_long_test(test_features_transformed, classifier, max_test_size=10)
        accuracies.append(accuracy_score(test_target, test_target_hat))
    accuracies = np.array(accuracies)
    acc_std = accuracies.std()
    acc_max = accuracies.max()
    acc_mean = accuracies.mean()
    return acc_max


def select_classifier_with_cross_validation(test_features_transformed, test_target, train_features_transformed, train_target):
    pass

def run_baseline_tabpfn_with_cross_validation(test_features_transformed, test_target, train_features, train_target,
                                              tabpfn_train_len=900,
                                              max_train_seeds=3
                                              ):
    num_train_seeds = min(max_train_seeds, int(np.ceil(len(train_features) / tabpfn_train_len)))
    max_test_size = 100
    min_val_size = 100
    if num_train_seeds <= 2:
        # Initialize and train the classifier
        classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
        classifier.fit(train_features, train_target)
    else:
        val_subset = train_features[:min_val_size]
        val_subset_target = train_target[:min_val_size]
        cross_validation_train_size = 900  # int((len(train_features) - min_val_size)/num_train_seeds)
        best_cv_score = -1.0
        best_classifier = None

        # Loop over the number of calculated training subsets, and validate on rest
        for seed_i in range(num_train_seeds):
            started_pos = seed_i * cross_validation_train_size + min_val_size
            stop_pos = min(started_pos+cross_validation_train_size, len(train_features))  # Ensure we do not go out of bounds

            # Slice the DataFrame to get the subset for training
            sampled_df = train_features.iloc[started_pos:stop_pos]
            sampled_y = train_target.iloc[started_pos:stop_pos]

            # Initialize and train the classifier
            classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
            classifier.fit(sampled_df, sampled_y)

            # Assume process_long_test is a function that processes the test data through the classifier
            cv_target_hat = process_long_test(val_subset, classifier, max_test_size=max_test_size)
            cv_acc = accuracy_score(val_subset_target, cv_target_hat)
            if cv_acc > best_cv_score:
                best_cv_score = cv_acc
                best_classifier = classifier
        classifier = best_classifier

    test_target_hat = process_long_test(test_features_transformed, classifier, max_test_size=max_test_size)
    acc_max = accuracy_score(test_target, test_target_hat)
    return acc_max  # Or return all metrics as needed


def _transform_dataframe_inplace(prediction: str, dataframe: pd.DataFrame, timeout: float, allow_no_transform: bool) -> Union[pd.DataFrame, None]:
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


class ScoreNormalizer:
    def __init__(self):
        self.normalizer = error_rate_normalizer
        self.baseline_results = {'dhanasekarjaisankar__correlation-between-posture-personality-trait.csv': {'tabpfn_score': 0.45, 'xgboost_score': 0.35}, 'arashnic__taxi-pricing-with-mobility-analytics.csv': {'tabpfn_score': 0.4829048452118354, 'xgboost_score': 0.4965771658933597}, 'gauravtopre__bank-customer-churn-dataset.csv': {'tabpfn_score': 0.8384138785625774, 'xgboost_score': 0.8453531598513011}, 'shubhamgupta012__titanic-dataset.csv': {'tabpfn_score': 0.8123167155425219, 'xgboost_score': 0.7800586510263929}, 'shivamb__vehicle-claim-fraud-detection.csv': {'tabpfn_score': 0.9391444713478612, 'xgboost_score': 0.9389830508474576}, 'vicsuperman__prediction-of-music-genre.csv': {'tabpfn_score': 0.09790634166076666, 'xgboost_score': 0.13689693536967734}, 'mansoordaku__ckdisease.csv': {'tabpfn_score': 0.9867549668874173, 'xgboost_score': 0.6556291390728477}, 'imsparsh__churn-risk-rate-hackerearth-ml.csv': {'tabpfn_score': 0.3052887866901123, 'xgboost_score': 0.3150953604761261}, 'youssefaboelwafa__hotel-booking-cancellation-prediction.csv': {'tabpfn_score': 0.7977852672123255, 'xgboost_score': 0.8678038379530917}, 'fedesoriano__hepatitis-c-dataset.csv': {'tabpfn_score': 0.9042145593869731, 'xgboost_score': 0.9042145593869731}, 'danushkumarv__glass-identification-data-set.csv': {'tabpfn_score': 0.9178082191780822, 'xgboost_score': 0.958904109589041}, 'shibumohapatra__book-my-show.csv': {'tabpfn_score': 0.8702137132680321, 'xgboost_score': 0.967720391807658}, 'debasisdotcom__parkinson-disease-detection.csv': {'tabpfn_score': 0.9117647058823529, 'xgboost_score': 0.8970588235294118}, 'abisheksudarshan__customer-segmentation.csv': {'tabpfn_score': 0.47399199753770394, 'xgboost_score': 0.4838411819021237}, 'robikscube__eye-state-classification-eeg-dataset.csv': {'tabpfn_score': 0.5694259012016022, 'xgboost_score': 0.9237316421895861}, 'gauravduttakiit__bar-passage-qualification.csv': {'tabpfn_score': 0.9473420260782347, 'xgboost_score': 0.9393179538615848}, 'reihanenamdari__breast-cancer.csv': {'tabpfn_score': 0.8955676988463873, 'xgboost_score': 0.891317547055252}, 'mysarahmadbhat__lung-cancer.csv': {'tabpfn_score': 0.9126984126984127, 'xgboost_score': 0.9047619047619048}, 'imsparsh__jobathon-analytics-vidhya.csv': {'tabpfn_score': 0.7602658788774003, 'xgboost_score': 0.7492368291482029}, 'harish24__music-genre-classification.csv': {'tabpfn_score': 0.645320197044335, 'xgboost_score': 0.6231527093596059}, 'mitesh58__bollywood-movie-dataset.csv': {'tabpfn_score': 0.58203125, 'xgboost_score': 0.58203125}, 'jillanisofttech__brain-stroke-dataset.csv': {'tabpfn_score': 0.9500249875062469, 'xgboost_score': 0.9405297351324338}, 'olcaybolat1__dermatology-dataset-classification.csv': {'tabpfn_score': 0.9664429530201343, 'xgboost_score': 0.9463087248322147}, 'snassimr__data-for-investing-type-prediction.csv': {'tabpfn_score': 0.7123359580052493, 'xgboost_score': 0.7706036745406825}, 'creepyghost__uci-ionosphere.csv': {'tabpfn_score': 0.9206349206349206, 'xgboost_score': 0.9126984126984127}, 'divyansh22__crystal-system-properties-for-liion-batteries.csv': {'tabpfn_score': 0.6214285714285714, 'xgboost_score': 0.6285714285714286}, 'cherngs__heart-disease-cleveland-uci.csv': {'tabpfn_score': 0.8308823529411765, 'xgboost_score': 0.7573529411764706}, 'jeevannagaraj__indian-liver-patient-dataset.csv': {'tabpfn_score': 0.6864406779661016, 'xgboost_score': 0.635593220338983}, 'rashikrahmanpritom__heart-attack-analysis-prediction-dataset.csv': {'tabpfn_score': 0.8046875, 'xgboost_score': 0.734375}, 'adhoppin__customer-cellular-data.csv': {'tabpfn_score': 0.898635477582846, 'xgboost_score': 0.9161793372319688}, 'captainozlem__framingham-chd-preprocessed-data.csv': {'tabpfn_score': 0.8465736810187993, 'xgboost_score': 0.8265615524560339}, 'saadmansakib__smartphone-human-fall-dataset.csv': {'tabpfn_score': 0.9691011235955056, 'xgboost_score': 0.9719101123595506}, 'arunavakrchakraborty__australia-weather-data.csv': {'tabpfn_score': 0.7742468634963564, 'xgboost_score': 0.7738461923722235}, 'sammy123__lower-back-pain-symptoms-dataset.csv': {'tabpfn_score': 0.8728813559322034, 'xgboost_score': 0.8050847457627118}, 'uciml__default-of-credit-card-clients-dataset.csv': {'tabpfn_score': 0.8144216999410328, 'xgboost_score': 0.8144216999410328}, 'juliasuzuki__waze-dataset-to-predict-user-churn.csv': {'tabpfn_score': 0.8266970706893527, 'xgboost_score': 0.027188212594281704}, 'caesarlupum__betsstrategy.csv': {'tabpfn_score': 1.0, 'xgboost_score': 1.0}, 'henrysue__online-shoppers-intention.csv': {'tabpfn_score': 0.890536149471974, 'xgboost_score': 0.8883021933387489}, 'ulrikthygepedersen__tic-tac-toe.csv': {'tabpfn_score': 1.0, 'xgboost_score': 0.989100817438692}, 'pavansubhasht__ibm-hr-analytics-attrition-dataset.csv': {'tabpfn_score': 0.8894736842105263, 'xgboost_score': 0.8771929824561403}, 'antimoni__metabolic-syndrome.csv': {'tabpfn_score': 0.8203285420944558, 'xgboost_score': 0.8213552361396304}, 'shrushtijoshi__asteroid-impacts.csv': {'tabpfn_score': 0.6825015994881638, 'xgboost_score': 0.9584133077415227}, 'rajgupta2019__qsar-bioconcentration-classes-dataset.csv': {'tabpfn_score': 0.7353846153846154, 'xgboost_score': -1.0}, 'iammustafatz__diabetes-prediction-dataset.csv': {'tabpfn_score': 0.9598883155086879, 'xgboost_score': 0.9714057786752425}, 'yasserh__wine-quality-dataset.csv': {'tabpfn_score': 0.6334056399132321, 'xgboost_score': 0.6268980477223427}, 'mdmahmudulhasansuzan__students-adaptability-level-in-online-education.csv': {'tabpfn_score': 0.8707070707070707, 'xgboost_score': 0.8565656565656565}, 'gauravduttakiit__smoker-status-prediction-using-biosignals.csv': {'tabpfn_score': 0.7278468899521531, 'xgboost_score': 0.7641467304625199}, 'raniahelmy__no-show-investigate-dataset.csv': {'tabpfn_score': 0.7963009134386545, 'xgboost_score': 0.7950542850019267}, 'ayushtankha__70k-job-applicants-data-human-resource.csv': {'tabpfn_score': 0.7783236106846791, 'xgboost_score': 0.780541056868966}, 'sahasourav17__child-sexual-abuse-awareness-knowledge-level.csv': {'tabpfn_score': 0.9157986111111112, 'xgboost_score': 0.9184027777777778}, 'devzohaib__eligibility-prediction-for-loan.csv': {'tabpfn_score': 0.6816326530612244, 'xgboost_score': 0.5959183673469388}, 'rohit265__breast-cancer-uci-machine-learning.csv': {'tabpfn_score': 0.7235772357723578, 'xgboost_score': 0.8048780487804879}, 'gdabhishek__fertilizer-prediction.csv': {'tabpfn_score': 0.9523809523809523, 'xgboost_score': 0.8571428571428571}, 'devvret__congressional-voting-records.csv': {'tabpfn_score': 0.9649122807017544, 'xgboost_score': 0.9590643274853801}, 'kukuroo3__body-performance-data.csv': {'tabpfn_score': 0.6731230400295148, 'xgboost_score': 0.7410071942446043}, 'mnassrib__telecom-churn-datasets.csv': {'tabpfn_score': 0.9250374812593704, 'xgboost_score': 0.9580209895052474}, 'sheikhsohelmoon__harry-potters-second-wizard-war-dataset.csv': {'tabpfn_score': 0.6, 'xgboost_score': 0.1}, 'sakshigoyal7__credit-card-customers.csv': {'tabpfn_score': 1.0, 'xgboost_score': 1.0}, 'architsharma01__loan-approval-prediction-dataset.csv': {'tabpfn_score': 0.9417360285374554, 'xgboost_score': 0.9815695600475625}, 'anaghakp__adult-income-census.csv': {'tabpfn_score': 0.7634945397815912, 'xgboost_score': 0.8}, 'yasserh__loan-default-dataset.csv': {'tabpfn_score': 0.8584507749472768, 'xgboost_score': 0.8696314397616577}, 'ulrikthygepedersen__mushroom-attributes.csv': {'tabpfn_score': 0.5383906633906634, 'xgboost_score': 0.9837223587223587}, 'adityakadiwal__water-potability.csv': {'tabpfn_score': 0.6133434420015162, 'xgboost_score': 0.5534495830174374}, 'mirlei__hcc-survival-data-set.csv': {'tabpfn_score': 0.639344262295082, 'xgboost_score': 0.6065573770491803}, 'sadeghjalalian__wine-customer-segmentation.csv': {'tabpfn_score': 1.0, 'xgboost_score': 0.9285714285714286}, 'janiobachmann__bank-marketing-dataset.csv': {'tabpfn_score': 0.7955714605233728, 'xgboost_score': 0.8163721762469246}, 'uciml__pima-indians-diabetes-database.csv': {'tabpfn_score': 0.7491638795986622, 'xgboost_score': 0.7190635451505016}, 'hiimanshuagarwal__advertising-ef.csv': {'tabpfn_score': 0.5279187817258884, 'xgboost_score': 0.5279187817258884}, 'manikantasanjayv__crop-recommender-dataset-with-soil-nutrients.csv': {'tabpfn_score': 0.9721115537848606, 'xgboost_score': 0.9760956175298805}}
        # 'sanskar457__fraud-transaction-detection.csv' = {dict: 2}
        # {'tabpfn_score': 1.0, 'xgboost_score': 1.0}
        # 'teejmahal20__airline-passenger-satisfaction.csv' = {dict: 2}
        # {'tabpfn_score': 0.9157684016014783, 'xgboost_score': 0.9650061595318756}
        # 'jinxzed__av-janatahack-crosssell-prediction.csv' = {dict: 2}
        # {'tabpfn_score': 0.8768809040171969, 'xgboost_score': -1.0}
        # 'mohamedkhaledidris__ofhddd.csv' = {dict: 2}
        # {'tabpfn_score': 0.9327409554318257, 'xgboost_score': 0.9326646835990136}
        # 'sajidhussain3__jobathon-may-2021-credit-card-lead-prediction.csv' = {dict: 2}
        # {'tabpfn_score': 0.8531411425098754, 'xgboost_score': 0.8321368884837436}
        # 'mlg-ulb__creditcardfraud.csv' = {dict: 2}
        # {'tabpfn_score': 0.998947442284752, 'xgboost_score': 0.9995175777138446}
        # for line in logs:
        #     dataset_name, results = line[:line.find(":")], line[line.find(":") + 2:]
        #     try:
        #         all_results[dataset_name] = eval(results)
        #     except:
        #         print(F" Not parsed {results}")

    def __call__(self, predictions):
        normalized_scores = []
        for prediction_k, prediction_v in predictions.items():
            normalized_score = self.normalizer(self.baseline_results[prediction_k], prediction_v)
            normalized_scores.append(normalized_score)
        return sum(normalized_scores)/len(normalized_scores)

def error_rate_normalizer(baseline_score, test_score):
    # 0.6 -> 0.9 , should be 0.75
    # 0.9 -> 0.975 , should be 0.75

    baseline_error_rate = 1.0 - baseline_score
    test_error_rate = 1.0 - test_score
    error_rate_reduction = (baseline_error_rate-test_error_rate)/baseline_error_rate
    return error_rate_reduction