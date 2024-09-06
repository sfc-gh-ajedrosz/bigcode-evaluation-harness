# personality traits
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    def encode_categorical(df, columns):
        for column in columns:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].fillna('Missing'))
        return df

    def fill_nans(df, strategy='mean'):
        for column in df.select_dtypes(include=[np.number]).columns:
            if strategy == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif strategy == 'zero':
                df[column] = df[column].fillna(0)
        return df

    def standardize_data(df, numerical_columns):
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        return df

    def pain_std(df):
        df['PAIN_STD'] = df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']].std(axis=1)
        return df

    def encode_sex(df):
        df['SEX'] = df['SEX'].map({'Male': 1, 'Female': 0})
        return df

    def weight_height_ratio(df):
        df['WeightHeightRatio'] = df['WEIGHT'] / df['HEIGHT']
        return df

    def pain_mbti_interaction(df):
        df['PAINxE'] = df['PAIN_MEAN'] * df['E']
        df['PAINxI'] = df['PAIN_MEAN'] * df['I']
        return df

    def aggregate_pain(df):
        df['PAIN_MEAN'] = df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']].mean(axis=1)
        df['PAIN_MAX'] = df[['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']].max(axis=1)
        return df

    def get_top_correlated_features(test_x, train_x, n=18):
        selected_columns = []
        columns = []
        correlation = train_x.corr()
        # print(correlation)
        top_corr = correlation.abs().unstack().sort_values(ascending=False).drop_duplicates()[:n]
        for i, j in top_corr.items():
            columns.append(i[0])
        for i in columns:
            if i not in selected_columns:
                selected_columns.append(i)
        train_x = train_x[selected_columns]
        test_x = test_x[selected_columns]
        return test_x, train_x

    #
    categorical_cols = ['SEX', 'ACTIVITY LEVEL', 'MBTI']  # Define categorical columns
    numerical_cols = ['AGE', 'HEIGHT', 'WEIGHT', 'E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']  # Define numerical columns

    # Encode categorical data and fill missing values
    train_x = encode_categorical(train_x, categorical_cols)
    test_x = encode_categorical(test_x, categorical_cols)

    # Fill NaNs for numerical columns
    train_x = fill_nans(train_x)
    test_x = fill_nans(test_x)

    # Standardize numerical data
    train_x = standardize_data(train_x, numerical_cols)
    test_x = standardize_data(test_x, numerical_cols)
    #
    train_x = encode_sex(train_x)
    test_x = encode_sex(test_x)
    train_x = aggregate_pain(train_x)
    test_x = aggregate_pain(test_x)
    train_x = pain_mbti_interaction(train_x)
    test_x = pain_mbti_interaction(test_x)
    train_x = weight_height_ratio(train_x)
    test_x = weight_height_ratio(test_x)
    train_x = pain_std(train_x)
    test_x = pain_std(test_x)
    test_x, train_x = get_top_correlated_features(test_x, train_x, n=20)
    return train_x, train_y, test_x
