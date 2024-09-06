import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def encode_categorical(df, columns, one_hot=False):
    """
    Encodes categorical columns in the DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to be processed.
    columns (list of str): List of column names to encode.
    one_hot (bool): If True, performs one-hot encoding; otherwise, performs label encoding.

    Returns:
    pd.DataFrame: The DataFrame with encoded columns.
    """
    for column in columns:
        if one_hot:
            # Perform one-hot encoding
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
        else:
            # Perform label encoding
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].fillna('Missing'))
    return df


def calculate_bmi(df):
    df['HEIGHT_m'] = df['HEIGHT'] * 0.0254
    df['WEIGHT_kg'] = df['WEIGHT'] * 0.453592
    df['BMI'] = df['WEIGHT_kg'] / (df['HEIGHT_m'] ** 2)
    return df


def aggregate_pain(df):
    df['PAIN_MEAN'] = df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']].mean(axis=1)
    df['PAIN_MAX'] = df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']].max(axis=1)
    return df


def encode_activity(df):
    activity_map = {'Low': 1, 'Moderate': 2, 'High': 3}
    df['ACTIVITY_LEVEL_ENC'] = df['ACTIVITY LEVEL'].map(activity_map)
    return df


def pain_mbti_interaction(df):
    df['PAIN_MEAN'] = df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']].mean(axis=1)
    df['PAINxE'] = df['PAIN_MEAN'] * df['E']
    df['PAINxI'] = df['PAIN_MEAN'] * df['I']
    return df


def encode_sex(df):
    df['SEX'] = df['SEX'].map({'Male': 1, 'Female': 0})
    return df


def weight_height_ratio(df):
    df['WeightHeightRatio'] = df['WEIGHT'] / df['HEIGHT']
    return df


def categorize_age(df):
    bins = [0, 18, 35, 50, 65, 100]
    labels = ['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly']
    df['AgeGroup'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
    return encode_categorical(df, ['AgeGroup'], one_hot=True)


def expand_mbti(df):
    df['Introvert'] = df['MBTI'].apply(lambda x: 1 if 'I' in x else 0)
    df['Intuitive'] = df['MBTI'].apply(lambda x: 1 if 'N' in x else 0)
    df['Feeling'] = df['MBTI'].apply(lambda x: 1 if 'F' in x else 0)
    df['Judging'] = df['MBTI'].apply(lambda x: 1 if 'J' in x else 0)
    return df


def activity_age_interaction(df):
    df = encode_activity(df)
    df['ActivityxAge'] = df['ACTIVITY_LEVEL_ENC'] * df['AGE']
    return df

def total_pain(df):
    df['PAIN_TOTAL'] = df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']].sum(axis=1)
    return df

def activity_weight_interaction(df):
    df = encode_activity(df)
    df['Activity_Weight_Interaction'] = df['ACTIVITY_LEVEL_ENC'] * df['WEIGHT']
    return df

def encode_mbti_frequency(df):
    frequency = df['MBTI'].value_counts().to_dict()
    df['MBTI_Frequency_Enc'] = df['MBTI'].map(frequency)
    return df

def log_transform_weight(df):
    df['Log_Weight'] = np.log(df['WEIGHT'] + 1)  # Adding 1 to handle zero weights just in case
    return df
def relative_height(df):
    max_height = df['HEIGHT'].max()
    df['Relative_Height'] = df['HEIGHT'] / max_height
    return df
def mbti_dichotomy_count(df):
    df['Extroverted_Traits'] = df['MBTI'].apply(lambda x: sum([1 for char in x if char in ['E', 'N', 'F', 'P']]))
    df['Introverted_Traits'] = df['MBTI'].apply(lambda x: sum([1 for char in x if char in ['I', 'S', 'T', 'J']]))
    df.drop('MBTI', axis=1, inplace=True)
    return df
def high_pain_binary(df, threshold=7):
    df['HIGH_PAIN'] = (df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']] > threshold).any(axis=1).astype(int)
    return df
def pain_sum_by_category(df):
    # Assuming pain categories are 1, 2, 3, 4 and defined in columns
    df['TOTAL_PAIN'] = df['PAIN 1'] + df['PAIN 2'] + df['PAIN 3'] + df['PAIN 4']
    return df
def age_bmi_interaction(df):
    calculate_bmi(df)
    df['Age_BMI_Interaction'] = df['AGE'] * df['BMI']
    return df
def pain_std(df):
    df['PAIN_STD'] = df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']].std(axis=1)
    return df

def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, do_these) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    # Process training data
    if 8 not in do_these or 17 not in do_these:
        train_x = encode_categorical(train_x, ['MBTI'], one_hot=False)
        test_x = encode_categorical(test_x, ['MBTI'], one_hot=False)

    if 1 in do_these:
        train_x = encode_sex(train_x)
        test_x = encode_sex(test_x)
    if 2 in do_these:
        train_x = calculate_bmi(train_x)
        test_x = calculate_bmi(test_x)
    if 3 in do_these:
        train_x = aggregate_pain(train_x)
        test_x = aggregate_pain(test_x)
    if 4 in do_these:
        train_x = encode_activity(train_x)
        test_x = encode_activity(test_x)
    if 5 in do_these:
        train_x = pain_mbti_interaction(train_x)
        test_x = pain_mbti_interaction(test_x)
    if 6 in do_these:
        train_x = weight_height_ratio(train_x)
        test_x = weight_height_ratio(test_x)
    if 7 in do_these:
        train_x = categorize_age(train_x)
        test_x = categorize_age(test_x)
    if 8 in do_these:
        train_x = expand_mbti(train_x)
        test_x = expand_mbti(test_x)
        train_x.drop('MBTI', axis=1, inplace=True)
        test_x.drop('MBTI', axis=1, inplace=True)
    if 9 in do_these:
        train_x = activity_age_interaction(train_x)
        test_x = activity_age_interaction(test_x)
    if 10 in do_these:
        train_x = total_pain(train_x)
        test_x = total_pain(test_x)
    if 11 in do_these:
        train_x = activity_age_interaction(train_x)
        test_x = activity_age_interaction(test_x)
    if 12 in do_these:
        train_x = total_pain(train_x)
        test_x = total_pain(test_x)
    if 13 in do_these:
        train_x = activity_weight_interaction(train_x)
        test_x = activity_weight_interaction(test_x)
    if 14 in do_these:
        train_x = encode_mbti_frequency(train_x)
        test_x = encode_mbti_frequency(test_x)
    if 15 in do_these:
        train_x = log_transform_weight(train_x)
        test_x = log_transform_weight(test_x)
    if 16 in do_these:
        train_x = relative_height(train_x)
        test_x = relative_height(test_x)
    if 17 in do_these:
        train_x = mbti_dichotomy_count(train_x)
        test_x = mbti_dichotomy_count(test_x)
    if 18 in do_these:
        train_x = high_pain_binary(train_x)
        test_x = high_pain_binary(test_x)
    if 19 in do_these:
        train_x = pain_sum_by_category(train_x)
        test_x = pain_sum_by_category(test_x)
    if 20 in do_these:
        train_x = age_bmi_interaction(train_x)
        test_x = age_bmi_interaction(test_x)
    if 21 in do_these:
        train_x = pain_std(train_x)
        test_x = pain_std(test_x)
    if 22 in do_these:
        test_x, train_x = get_top_correlated_features(test_x, train_x, n=13)
    # if do_these[-1]>22:
    #     test_x, train_x = get_top_correlated_features(test_x, train_x, n=5+(do_these[-1]-22))
    return train_x, train_y, test_x


