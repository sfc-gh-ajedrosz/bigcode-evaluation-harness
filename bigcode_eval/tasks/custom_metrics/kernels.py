kernels = {
    'shubhamgupta012__titanic-dataset.csv' : """
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    # Combine train and test data temporarily for consistent feature encoding
    combined = pd.concat([train_x, test_x], axis=0)

    # Encoding categorical features
    encoders = {
        'Sex': LabelEncoder(),
        'Embarked': LabelEncoder()
    }

    for column, encoder in encoders.items():
        # Encode the column
        combined[column] = encoder.fit_transform(combined[column].astype(str))
        # Convert to int to ensure the dtype is acceptable by XGBoost
        combined[column] = combined[column].astype(int)

    # Split the combined back into train and test
    train_x = combined.iloc[:len(train_x)]
    test_x = combined.iloc[len(train_x):]

    # Handle missing values
    # Fill missing values with the median of the training set
    for column in ['Age', 'Fare']:
        median = train_x[column].median()
        train_x[column].fillna(median, inplace=True)
        test_x[column].fillna(median, inplace=True)

    # Drop non-informative features
    if 'PassengerId' in train_x.columns:
        train_x.drop('PassengerId', axis=1, inplace=True)
        test_x.drop('PassengerId', axis=1, inplace=True)

    return train_x, train_y, test_x
    """,
    'OLD_dhanasekarjaisankar__correlation-between-posture-personality-trait.csv' : """    
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    # categorical_cols = ['SEX', 'ACTIVITY LEVEL', 'MBTI']  # Define categorical columns
    categorical_cols = ['SEX', 'ACTIVITY LEVEL']  # Define categorical columns
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

    return train_x, train_y, test_x

# Example usage:
# Assuming train_x, train_y, and test_x are already defined pandas DataFrame and Series
# train_x_transformed, train_y_transformed, test_x_transformed = transform(train_x, train_y, test_x)
""",
    'BOTH__dhanasekarjaisankar__correlation-between-posture-personality-trait.csv': """
    
# personality traits
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    categorical_cols = ['SEX', 'ACTIVITY LEVEL']  # Define categorical columns
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
    test_x, train_x = get_top_correlated_features(test_x, train_x, n=18)
    return train_x, train_y, test_x
""",
    'dhanasekarjaisankar__correlation-between-posture-personality-trait.csv':
"""
# personality traits
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (
pd.DataFrame, pd.Series, pd.DataFrame):
    def encode_categorical(df, columns, one_hot=False):
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

    train_x = encode_categorical(train_x, ['MBTI', 'ACTIVITY LEVEL', 'SEX'], one_hot=False)
    test_x = encode_categorical(test_x, ['MBTI', 'ACTIVITY LEVEL', 'SEX'], one_hot=False)
    train_x = fill_nans(train_x)
    test_x = fill_nans(test_x)
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
    test_x, train_x = get_top_correlated_features(test_x, train_x, n=18)
    return train_x, train_y, test_x
""",
'GPT_dhanasekarjaisankar__correlation-between-posture-personality-trait.csv': """       # refactored by chat gpt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Define global helper functions
def encode_categorical(df, columns, one_hot=False):
    for column in columns:
        if one_hot:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
        else:
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


def transform(train_x, train_y, test_x):

    # Common columns and setups
    categorical_cols = ['MBTI', 'ACTIVITY LEVEL', 'SEX']
    pain_columns = ['PAIN 1', 'PAIN 2', 'PAIN 3', 'PAIN 4']

    # Encoding and cleaning
    for df in [train_x, test_x]:
        df = encode_categorical(df, categorical_cols, one_hot=False)
        df = fill_nans(df)
        # df['SEX'] = df['SEX'].map({'Male': 1, 'Female': 0})  # Encode sex directly
        df['WeightHeightRatio'] = df['WEIGHT'] / df['HEIGHT']
        df['PAIN_MEAN'] = df[pain_columns].mean(axis=1)
        df['PAIN_MAX'] = df[pain_columns].max(axis=1)
        df['PAIN_STD'] = df[pain_columns].std(axis=1)
        df['PAINxE'] = df['PAIN_MEAN'] * df['E']
        df['PAINxI'] = df['PAIN_MEAN'] * df['I']

    # Feature selection based on correlation in train data
    correlation = train_x.corr()
    test_x, train_x = get_top_correlated_features(test_x, train_x, n=18)
    # top_features = correlation.nlargest(18, 'target_column')['target_column'].index  # Adjust target column name accordingly
    # train_x = train_x[top_features]
    # test_x = test_x[top_features]

    return train_x, train_y, test_x

# Usage
# Assuming train_x, train_y, test_x are already defined
# train_x_transformed, train_y_transformed, test_x_transformed = transform(train_x, train_y, test_x)
"""
,
'arunavakrchakraborty__australia-weather-data.csv': """
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    # categorical_cols = ['SEX', 'ACTIVITY LEVEL', 'MBTI']  # Define categorical columns
    categorical_cols = ['SEX', 'ACTIVITY LEVEL']  # Define categorical columns
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

    return train_x, train_y, test_x"""
}




# TODO : wziąć selekcję top kolumn i zrobić ją wprost z jakimś komentarzem




ideas_to_check = {
'shubhamgupta012__titanic-dataset.csv': """

def transform(train_x, train_y, test_x, do_these=()):
    def extract_title(df):
        df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        return df

    def family_size(df):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        return df

    def cabin_features(df):
        df['Cabin'] = df['Cabin'].fillna('Unknown')
        df['Deck'] = df['Cabin'].apply(lambda x: x[0])
        return df

    def ticket_prefix(df):
        df['TicketPrefix'] = df['Ticket'].apply(
            lambda x: ''.join(filter(str.isalpha, x.split()[0])) if not x.split()[0].isdigit() else 'None')
        return df

    def age_class_interaction(df):
        df['Age*Class'] = df['Age'] * df['Pclass']
        return df

    def fare_per_person(df):
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
        return df

    if 1 in do_these:
        train_x = extract_title(train_x)
        test_x = extract_title(test_x)

    if 2 in do_these:
        train_x = family_size(train_x)
        test_x = family_size(test_x)

    if 3 in do_these:
        train_x = cabin_features(train_x)
        test_x = cabin_features(test_x)

    if 4 in do_these:
        train_x = ticket_prefix(train_x)
        test_x = ticket_prefix(test_x)

    if 5 in do_these:       # improves  0.8064516129032258
        train_x = age_class_interaction(train_x)
        test_x = age_class_interaction(test_x)

    if 6 in do_these:       # 0.7976539589442815
        train_x = fare_per_person(train_x)
        test_x = fare_per_person(test_x)

    # Continue with encoding and other preprocessing
    return train_x, train_y, test_x

""",

}


