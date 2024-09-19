kernels = {
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

'arunavakrchakraborty__australia-weather-data.csv':
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

def encode_categorical(df, columns):
    for column in columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].fillna('Missing'))
    return df

def apply_city_specific_normalization(train_x, test_x, numeric_cols):
    # Normalize numeric columns for each city in train_x and apply the same transformation to test_x.
    scalers = {}

    # Iterate over each city in the training data
    for city, group in train_x.groupby('Location'):
        if numeric_cols:  # Check if there are numeric columns to scale
            scaler = StandardScaler()
            # Fit the scaler on the training group
            train_x.loc[train_x['Location'] == city, numeric_cols] = scaler.fit_transform(group[numeric_cols])
            # Store the scaler to apply it to the test data
            scalers[city] = scaler

    # Apply the same scaler to corresponding cities in the test data
    for city, scaler in scalers.items():
        if city in test_x['Location'].values:
            test_x.loc[test_x['Location'] == city, numeric_cols] = scaler.transform(
                test_x[test_x['Location'] == city][numeric_cols])

    return train_x, test_x

def map_geo_to_climate_resilience(df):
    # Hypothetical scores based on broad characteristics
    climate_resilience = {
        'Albury': 0.6, 'BadgerysCreek': 0.5, 'Cobar': 0.4, 'CoffsHarbour': 0.8,
        'Moree': 0.5, 'Newcastle': 0.7, 'NorahHead': 0.75, 'NorfolkIsland': 0.85,
        'Penrith': 0.5, 'Richmond': 0.5, 'Sydney': 0.8, 'SydneyAirport': 0.8,
        'WaggaWagga': 0.55, 'Williamtown': 0.7, 'Wollongong': 0.8,
        'Canberra': 0.65, 'Tuggeranong': 0.65, 'MountGinini': 0.6,
        'Ballarat': 0.7, 'Bendigo': 0.65, 'Sale': 0.7, 'MelbourneAirport': 0.75,
        'Melbourne': 0.75, 'Mildura': 0.4, 'Nhil': 0.45, 'Portland': 0.8,
        'Watsonia': 0.75, 'Dartmoor': 0.75, 'Brisbane': 0.85, 'Cairns': 0.85,
        'GoldCoast': 0.85, 'Townsville': 0.8, 'Adelaide': 0.75,
        'MountGambier': 0.75, 'Nuriootpa': 0.6, 'Woomera': 0.4,
        'Albany': 0.75, 'Witchcliffe': 0.75, 'PearceRAAF': 0.7,
        'PerthAirport': 0.75, 'Perth': 0.75, 'SalmonGums': 0.5,
        'Walpole': 0.75, 'Hobart': 0.8, 'Launceston': 0.7,
        'AliceSprings': 0.35, 'Darwin': 0.9, 'Katherine': 0.7, 'Uluru': 0.3
    }
    df['ClimateResillience'] = df['Location'].map(lambda x: climate_resilience[x])
    return df

def map_geo_1(geo_features, df):
    # Map each geo feature to the DataFrame using the dictionary values
    df['Coast'] = df['Location'].map(lambda x: geo_features[x]['Coast'])
    return df

def map_geo_2(geo_features, df):
    # Map each geo feature to the DataFrame using the dictionary values
    df['Ocean'] = df['Location'].map(lambda x: geo_features[x]['Ocean'])
    return df

def map_geo_3(geo_features, df):
    # Map each geo feature to the DataFrame using the dictionary values
    df['Desert'] = df['Location'].map(lambda x: geo_features[x]['Desert'])
    return df

def map_geo_4(geo_features, df):
    # Map each geo feature to the DataFrame using the dictionary values
    df['Mountains'] = df['Location'].map(lambda x: geo_features[x]['Mountains'])
    return df

def weather_conditions_ratios(df):
    df['Temperature_Range_Ratio'] = (df['MaxTemp'] - df['MinTemp'] +100) / (df['MaxTemp'] +100)
    df['Pressure_Stability_Ratio'] = df['Pressure9am'] / df['Pressure3pm']
    return df

def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    numeric_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                    'Humidity3pm',
                    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
                    'Temp3pm']

    categorical_cols = ['WindGusDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    train_x, test_x = apply_city_specific_normalization(train_x, test_x, numeric_cols)

    geo_features = {
        'Albury': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'BadgerysCreek': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Cobar': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
        'CoffsHarbour': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Moree': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
        'Newcastle': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'NorahHead': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'NorfolkIsland': {'Coast': 'Island', 'Ocean': True, 'Desert': False, 'Mountains': False},
        'Penrith': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Richmond': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Sydney': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'SydneyAirport': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'WaggaWagga': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Williamtown': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Wollongong': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Canberra': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Tuggeranong': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'MountGinini': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Ballarat': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Bendigo': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Sale': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'MelbourneAirport': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Melbourne': {'Coast': 'South', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Mildura': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
        'Nhil': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
        'Portland': {'Coast': 'South', 'Ocean': True, 'Desert': False, 'Mountains': False},
        'Watsonia': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Dartmoor': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Brisbane': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Cairns': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'GoldCoast': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Townsville': {'Coast': 'East', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Adelaide': {'Coast': 'South', 'Ocean': True, 'Desert': True, 'Mountains': True},
        'MountGambier': {'Coast': 'South', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Nuriootpa': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'Woomera': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
        'Albany': {'Coast': 'South', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Witchcliffe': {'Coast': 'South', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'PearceRAAF': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'PerthAirport': {'Coast': 'West', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Perth': {'Coast': 'West', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'SalmonGums': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
        'Walpole': {'Coast': 'South', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Hobart': {'Coast': 'South', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Launceston': {'Coast': 'None', 'Ocean': False, 'Desert': False, 'Mountains': True},
        'AliceSprings': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
        'Darwin': {'Coast': 'North', 'Ocean': True, 'Desert': False, 'Mountains': True},
        'Katherine': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
        'Uluru': {'Coast': 'None', 'Ocean': False, 'Desert': True, 'Mountains': False},
    }

    train_x = map_geo_1(geo_features, train_x)
    test_x = map_geo_1(geo_features, test_x)
    train_x = map_geo_2(geo_features, train_x)
    test_x = map_geo_2(geo_features, test_x)
    train_x = map_geo_3(geo_features, train_x)
    test_x = map_geo_3(geo_features, test_x)
    train_x = map_geo_4(geo_features, train_x)
    test_x = map_geo_4(geo_features, test_x)

    # add map to climate resilience informaiton
    train_x = map_geo_to_climate_resilience(train_x)
    test_x = map_geo_to_climate_resilience(test_x)


    categorical = train_x.select_dtypes(include="object").columns
    cleaner = ColumnTransformer([
        ('categorical_transformer', SimpleImputer(strategy='most_frequent'), categorical)
    ])
    train_x[categorical] = cleaner.fit_transform(train_x[categorical])
    test_x[categorical] = cleaner.fit_transform(test_x[categorical])

    train_x = encode_categorical(train_x, categorical)
    test_x = encode_categorical(test_x, categorical)

    columns_to_remove = ['row ID', 'Evaporation', 'Sunshine']

    train_x = train_x.drop(columns=columns_to_remove)
    test_x = test_x.drop(columns=columns_to_remove)

    train_x = train_x.fillna(train_x.median())
    test_x = test_x.fillna(train_x.median())

    train_x = weather_conditions_ratios(train_x)
    test_x = weather_conditions_ratios(test_x)

    # Adding new feature: MaxTemp * Temperature_Range_Ratio
    train_x['MaxTemp_x_Temperature_Range_Ratio'] = train_x['MaxTemp'] * train_x['Temperature_Range_Ratio']
    test_x['MaxTemp_x_Temperature_Range_Ratio'] = test_x['MaxTemp'] * test_x['Temperature_Range_Ratio']
    # Adding new feature: MaxTemp * Temp3pm
    train_x['MaxTemp_x_Temp3pm'] = train_x['MaxTemp'] * train_x['Temp3pm']
    test_x['MaxTemp_x_Temp3pm'] = test_x['MaxTemp'] * test_x['Temp3pm']
    # # Adding new feature: MaxTemp * Temp3pm
    # train_x['MaxTemp_x_Temp3pm2'] = train_x['MaxTemp'] * train_x['Temp3pm']
    # test_x['MaxTemp_x_Temp3pm2'] = test_x['MaxTemp'] * test_x['Temp3pm']
    # Adding new feature: Location / Cloud3pm
    train_x['Location_div_Cloud3pm'] = train_x['Location'] / train_x['Cloud3pm']
    test_x['Location_div_Cloud3pm'] = test_x['Location'] / test_x['Cloud3pm']
    return train_x, train_y, test_x
""",

'maajdl__yeh-concret-data.csv':
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    # do_these=(4, 30, 33, 44)
    # Adding new feature: cement / water
    train_x[f'feat_cement_div_water'] = train_x["cement"] / train_x["water"]
    test_x[f'feat_cement_div_water'] = test_x["cement"] / test_x["water"]
    # Adding "new" feature: water / "coarseaggregat"
    train_x[f'feat_water_div_coarseaggregate'] = train_x["water"] / train_x["coarseaggregate"]
    test_x[f'feat_water_div_coarseaggregate'] = test_x["water"] / test_x["coarseaggregate"]
    # Adding "new" feature: superplasticizer / "cemen"
    train_x[f'feat_superplasticizer_div_cement'] = train_x["superplasticizer"] / train_x["cement"]
    test_x[f'feat_superplasticizer_div_cement'] = test_x["superplasticizer"] / test_x["cement"]
    # Adding "new" feature: coarseaggregate / "wate"
    train_x[f'feat_coarseaggregate_div_water'] = train_x["coarseaggregate"] / train_x["water"]
    test_x[f'feat_coarseaggregate_div_water'] = test_x["coarseaggregate"] / test_x["water"]
    # Adding "new" feature: slag * "flyas"
    train_x[f'feat_slag_div_flyash'] = train_x["slag"] * train_x["flyash"]
    test_x[f'feat_slag_div_flyash'] = test_x["slag"] * test_x["flyash"]

    # Ratio of slag to total components
    train_x['slag_ratio'] = train_x['slag'] / train_x[
        ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate']].sum(
        axis=1)
    test_x['slag_ratio'] = test_x['slag'] / test_x[
        ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate']].sum(
        axis=1)
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)
    test_x = pd.DataFrame(scaler.transform(test_x), columns=train_x.columns)
    return train_x, train_y, test_x"""
}