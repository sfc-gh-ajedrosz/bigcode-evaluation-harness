import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.utils import resample
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer, make_column_selector
import requests

def get_elevation(lat, lon):
    """Fetches elevation for given latitude and longitude using OpenElevation API."""
    base_url = "https://api.open-elevation.com/api/v1/lookup"
    parameters = {
        'locations': f"{lat},{lon}"
    }
    response = requests.get(base_url, params=parameters)
    data = response.json()
    if 'results' not in data:
        return None
    return data['results'][0]['elevation'] if data['results'] else None

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


def fill_missing_values(df):
    # Fill missing values with mean for numerical columns

    for col in numerical_features:
        df[col] = df[col].fillna(df[col].mean())  # Using mean to fill missing values

    return df


def encode_features(train_x, test_x, ordinal_features):
    # Label encoding for ordinal data
    label_encoder = LabelEncoder()
    for df in [train_x, test_x]:
        for col in ordinal_features:
            df[col] = label_encoder.fit_transform(
                df[col].fillna('Missing'))  # Handling missing values by replacing them with a placeholder

    # One-hot encoding for nominal data
    nominal_features = []  #'Location']  # Example of nominal feature
    train_x = pd.get_dummies(train_x, columns=nominal_features, drop_first=True)
    test_x = pd.get_dummies(test_x, columns=nominal_features, drop_first=True)
    return train_x, test_x


def transform_data_good_85p4(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (
pd.DataFrame, pd.Series, pd.DataFrame):
    numerical_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am',
                          'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                          'Cloud3pm', 'Temp9am', 'Temp3pm']
    ordinal_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Location']  # Example of ordinal features

    # Encode categorical data and fill missing values
    train_x = encode_categorical(train_x, ordinal_features)
    test_x = encode_categorical(test_x, ordinal_features)

    # Fill NaNs for numerical columns
    train_x = fill_nans(train_x)
    test_x = fill_nans(test_x)

    # Standardize numerical data
    train_x = standardize_data(train_x, numerical_features)
    test_x = standardize_data(test_x, numerical_features)

    # Continue from the previous encoded dataframe
    train_x, test_x = encode_features(train_x, test_x, ordinal_features)
    columns_to_remove = ['row ID', 'Evaporation', 'Sunshine']

    train_x = train_x.drop(columns=columns_to_remove)
    test_x = test_x.drop(columns=columns_to_remove)

    return train_x, train_y, test_x


def encode_weather_conditions(df, columns, one_hot=True):
    """Encodes weather condition columns using one-hot or label encoding."""
    for column in columns:
        if one_hot:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
        else:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].fillna('Missing'))
    return df


def calculate_temperature_indices(df):
    """Calculates additional temperature-related indices like wind chill and heat index."""
    # Wind chill calculation
    df['WindChillC'] = 13.12 + 0.6215 * df['TempC'] - 11.37 * (df['WindSpeedKmph'] ** 0.16) + 0.3965 * df['TempC'] * (df['WindSpeedKmph'] ** 0.16)
    # Heat index calculation
    df['HeatIndexC'] = -8.784695 + 1.61139411 * df['TempC'] + 2.338549 * df['Humidity'] - 0.14611605 * df['TempC'] * df['Humidity'] \
                     - 0.012308094 * (df['TempC'] ** 2) - 0.016424828 * (df['Humidity'] ** 2) + 0.002211732 * (df['TempC'] ** 2) * df['Humidity'] \
                     + 0.00072546 * df['TempC'] * (df['Humidity'] ** 2) - 0.000003582 * (df['TempC'] ** 2) * (df['Humidity'] ** 2)
    return df


def aggregate_rainfall(df):
    """Aggregates rainfall data into more useful statistics."""
    df['TotalRainfall'] = df[['Rainfall_mm']].sum(axis=1)
    df['RainyDays'] = (df['Rainfall_mm'] > 0).astype(int)
    return df

def encode_wind_direction(df):
    """One-hot encodes wind direction columns."""
    wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in wind_cols:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df

def calculate_temperature_range(df):
    """Calculates the daily temperature range."""
    df['TempRange'] = df['MaxTemp'] - df['MinTemp']
    return df

def process_rain_data(df):
    """Encodes 'RainToday' and aggregates rain data."""
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    df['Rainfall'] = df['Rainfall'].fillna(0)  # Assuming NaN implies no rainfall
    return df

def normalize_pressure(df):
    """Normalizes morning and afternoon pressure readings."""
    pressure_cols = ['Pressure9am', 'Pressure3pm']
    for col in pressure_cols:
        max_val = df[col].max()
        min_val = df[col].min()
        df[col + '_Normalized'] = (df[col] - min_val) / (max_val - min_val)
    return df

def calculate_humidity_difference(df):
    """Calculates the difference in humidity from morning to afternoon."""
    df['HumidityChange'] = df['Humidity9am'] - df['Humidity3pm']
    return df
def normalize_pressure(df):
    """Normalizes atmospheric pressure readings to a standard range."""
    max_pressure = df['Pressure9am', 'Pressure3pm'].max()
    min_pressure = df['Pressure'].min()
    df['NormalizedPressure'] = (df['Pressure'] - min_pressure) / (max_pressure - min_pressure)
    return df


def pressure_change(df):
    # Pressure Change Over the Day
    df['Pressure_Change'] = df['Pressure9am'] - df['Pressure3pm']
    return df


def wind_congruence(df):
    # Wind Speed and Wind Gust Direction Congruence
    df['Wind_Congruence'] = (df['WindDir9am'] == df['WindGustDir']).astype(int)
    return df


def humidity_rainfall(df):
    # Humidity and Rainfall Interaction
    df['Humidity_Rainfall'] = df['Humidity9am'] * df['Rainfall']
    return df

def temp_variability(df):
    # Temperature Variability Index
    df['Temp_Variability'] = df[['Temp9am', 'Temp3pm', 'MinTemp', 'MaxTemp']].std(axis=1)
    return df

def wind_gust(df):
    # Wind Gust Speed and Max Temperature Interaction
    df['WindGust_MaxTemp'] = df['WindGustSpeed'] * df['MaxTemp']
    return df

def rain_humidity(df):
    # Rainfall Binary Multiplied by Humidity
    # df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    df['Rain_Humidity'] = df['RainToday'] * df['Humidity3pm']
    return df

def pressure_humidity_index(df):
    # Pressure and Humidity Composite Index
    normalized_pressure = (df['Pressure9am'] + df['Pressure3pm']) / 2
    normalized_humidity = (df['Humidity9am'] + df['Humidity3pm']) / 2
    df['Pressure_Humidity_Index'] = (normalized_pressure + normalized_humidity) / 2
    return df

def apply_city_specific_normalization(train_x, test_x, numeric_cols):
    """
    Normalize numeric columns for each city in train_x and apply the same transformation to test_x.

    Args:
    train_x (pd.DataFrame): The training DataFrame.
    test_x (pd.DataFrame): The testing DataFrame.
    numeric_cols (list of str): List of names of numeric columns to be scaled.

    Returns:
    pd.DataFrame, pd.DataFrame: The modified train_x and test_x DataFrames with scaled numeric columns.
    """
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
def normalize_encode_by_city(train_x, test_x, numeric_cols, categorical_cols):
    """
    Normalize numeric columns and encode categorical columns separately for each city.

    Args:
    df (pd.DataFrame): The DataFrame containing weather data.
    numeric_cols (list of str): List of names of numeric columns to be scaled.
    categorical_cols (list of str): List of names of categorical columns to be encoded.

    Returns:
    pd.DataFrame: The DataFrame with normalized and encoded columns.
    """
    # Create an empty DataFrame to hold processed data
    processed_df = pd.DataFrame()

    # Iterate over each city
    for city, group in train_x.groupby('Location'):
        # Normalize numeric columns
        scaler = StandardScaler()
        if numeric_cols:  # Check if there are numeric columns to scale
            scaler = scaler.fit(group[numeric_cols])
            group[numeric_cols] = scaler.transform(group[numeric_cols])
            test_x[test_x['Location'] == city][numeric_cols] = scaler.transform(
                test_x[test_x['Location'] == city][numeric_cols])
        # # Encode categorical columns
        # for col in categorical_cols:
        #     if col in group.columns:
        #         encoder = LabelEncoder()
        #         group[col] = encoder.fit_transform(group[col].astype(str))

        # Append processed group to the main DataFrame
        processed_df = pd.concat([processed_df, group], axis=0)

    # Return the combined DataFrame with processed data
    return processed_df, test_x

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


def map_geo_to_climate_resilience(df):
    # Hypothetical scores based on broad characteristics\
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

def interaction_temp_humid(df):
    df['Temp_Humidity_Index'] = df['MaxTemp'] * df['Humidity3pm']
    df['Wind_Chill'] = 13.12 + 0.6215 * df['MinTemp'] - 11.37 * (df['WindSpeed9am']**0.16) + 0.3965 * df['MinTemp'] * (df['WindSpeed9am']**0.16)
    df['Humidity_WindSpeed_Index'] = df['Humidity3pm'] / (df['WindSpeed3pm'] + 1)  # +1 to avoid division by zero

    return df

def std_by_location(df):
    for col in ['MinTemp', 'MaxTemp', 'Pressure9am', 'Pressure3pm']:
        df[col + '_std_by_location'] = df.groupby('Location')[col].transform(lambda x: (x - x.mean()) / x.std())

def fire_risk_index(df):
    df['Fire_Risk_Index'] = (df['MaxTemp'] * df['WindSpeed3pm']) / (df['Humidity3pm'] + 1)
    return df

def weather_conditions_ratios(df):
    df['Temperature_Range_Ratio'] = (df['MaxTemp'] - df['MinTemp'] +100) / (df['MaxTemp'] +100)
    df['Pressure_Stability_Ratio'] = df['Pressure9am'] / df['Pressure3pm']
    return df


def diurnal_temperature_range(df):
    df['DTR'] = df['MaxTemp'] - df['MinTemp']
    return df
def weather_severity_index(df):
    df = calculate_temperature_range(df)
    df['Weather_Severity'] = (df['WindGustSpeed'] + df['Rainfall'] + df['TempRange']) / 3
    return df

def temperature_inversion(df):
    df['Temp_Inversion'] = (df['Temp3pm'] > df['Temp9am']).astype(int)
    return df

def cumulative_rainfall(df):
    df['Cumulative_Rainfall'] = df['Rainfall'].expanding().sum()
    return df

def normalize_weather_metrics(df, columns):
    for col in columns:
        df[col + '_Normalized'] = (df[col] - df[col].mean()) / df[col].std()
    return df
def temperature_humidity_index(df):
    df['THI'] = 0.8 * df['Temp9am'] + (df['Humidity9am']/100) * (df['Temp9am']-14.4) + 46.4
    return df
def advanced_climatic_features(df):
    # Placeholder for actual implementation
    df['UV_Index'] = np.random.normal(loc=5, scale=2, size=len(df))  # Hypothetical implementation
    return df

def transform_data(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, do_these=(0,)) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    numeric_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                    'Humidity3pm',
                    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
                    'Temp3pm']
    if 1 in do_these:

        categorical_cols = ['WindGusDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        train_x, test_x = apply_city_specific_normalization(train_x, test_x, numeric_cols)
        # train_x, test_x = normalize_encode_by_city(train_x, test_x,numeric_cols, categorical_cols)
        # test_x = normalize_encode_by_city(test_x, numeric_cols, categorical_cols)
    if 15 in do_these:
        train_x = map_geo_1(geo_features, train_x)
        test_x = map_geo_1(geo_features, test_x)
        train_x = map_geo_2(geo_features, train_x)
        test_x = map_geo_2(geo_features, test_x)
        train_x = map_geo_3(geo_features, train_x)
        test_x = map_geo_3(geo_features, test_x)
        train_x = map_geo_4(geo_features, train_x)
        test_x = map_geo_4(geo_features, test_x)
    if 16 in do_these:
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


    if 2 in do_these:
        train_x = calculate_temperature_range(train_x)
        test_x = calculate_temperature_range(test_x)
    if 3 in do_these:
        train_x = process_rain_data(train_x)
        test_x = process_rain_data(test_x)
    if 4 in do_these:
        train_x = normalize_pressure(train_x)
        test_x = normalize_pressure(test_x)
    if 5 in do_these:
        train_x = calculate_humidity_difference(train_x)
        test_x = calculate_humidity_difference(test_x)

    if 6 in do_these:
        train_x = add_interaction_features(train_x)
        test_x = add_interaction_features(test_x)

    if 7 in do_these:
        train_x = pressure_change(train_x)
        test_x = pressure_change(test_x)
    if 8 in do_these:
        train_x = wind_congruence(train_x)
        test_x = wind_congruence(test_x)
    if 9 in do_these:
        train_x = humidity_rainfall(train_x)
        test_x = humidity_rainfall(test_x)
    if 10 in do_these:
        train_x = temp_variability(train_x)
        test_x = temp_variability(test_x)
    if 11 in do_these:
        train_x = wind_gust(train_x)
        test_x = wind_gust(test_x)
    if 12 in do_these:
        train_x = rain_humidity(train_x)
        test_x = rain_humidity(test_x)
    if 13 in do_these:
        train_x = pressure_humidity_index(train_x)
        test_x = pressure_humidity_index(test_x)
    if 14 in do_these:
        train_x = encode_wind_direction(train_x)
        test_x = encode_wind_direction(test_x)

    if 17 in do_these:
        train_x = interaction_temp_humid(train_x)
        test_x = interaction_temp_humid(test_x)

    if 18 in do_these:
        train_x = std_by_location(train_x)
        test_x = std_by_location(test_x)

    if 19 in do_these:
        train_x = fire_risk_index(train_x)
        test_x = fire_risk_index(test_x)
    if 21 in do_these:
        train_x = weather_conditions_ratios(train_x)
        test_x = weather_conditions_ratios(test_x)
    if 22 in do_these:

        train_x = diurnal_temperature_range(train_x)
        test_x = diurnal_temperature_range(test_x)
    if 23 in do_these:

        train_x = weather_severity_index(train_x)
        test_x = weather_severity_index(test_x)
    if 24 in do_these:
        train_x = temperature_inversion(train_x)
        test_x = temperature_inversion(test_x)
    if 25 in do_these:
        train_x = cumulative_rainfall(train_x)
        test_x = cumulative_rainfall(test_x)
    if 26 in do_these:
        train_x = normalize_weather_metrics(train_x, numeric_cols)
        test_x = normalize_weather_metrics(test_x, numeric_cols)
    if 27 in do_these:
        train_x = temperature_humidity_index(train_x)
        test_x = temperature_humidity_index(test_x)
    if 28 in do_these:
        train_x = advanced_climatic_features(train_x)
        test_x = advanced_climatic_features(test_x)

    city_coordinates = {
        'Sydney': (-33.86785, 151.20732),
        'Melbourne': (-37.814, 144.96332),
        'Brisbane': (-27.46794, 153.02809),
        'Perth': (-31.95224, 115.8614),
        'Canberra': (-35.28346, 149.12807),
        'Adelaide': (-34.92866, 138.59863),
        'Gold Coast': (-28.00029, 153.43088),
        'Newcastle': (-32.92953, 151.7801),
        'Wollongong': (-34.424, 150.89345),
        'Townsville': (-19.26639, 146.80569),
        'Cairns': (-16.92366, 145.76613),
        'Darwin': (-12.46113, 130.84185),
        'Launceston': (-41.43876, 147.13467),
        'Ballarat': (-37.56622, 143.84957),
        'Hobart': (-42.87936, 147.32941)
    }
    # elevs = {}
    # import time
    #
    # for k, v in city_coordinates.items():
    #     elev = get_elevation(v[0], v[1])
    #     elevs[k] = elev
    #     # Pause for 5 seconds
    #     time.sleep(3)
    return train_x, train_y, test_x


# train_x, train_y, test_x = transform(data_split.train_x, data_split.train_target, data_split.test_x)
# data_split_tfed = DataSplit(test_x, data_split.test_target, train_x, train_y)

#
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# import category_encoders as ce
#
#
# def encode_categorical(df, categorical_columns):
#     """ Encodes categorical columns using Label Encoding """
#     encoder = LabelEncoder()
#     for column in categorical_columns:
#         df[column] = encoder.fit_transform(df[column].fillna('Missing'))
#     return df
#
#
# def one_hot_encode(df, nominal_features):
#     """ Applies one-hot encoding to specified nominal features """
#     return pd.get_dummies(df, columns=nominal_features, drop_first=True)
#
#
# def fill_missing_values(df, strategy='mean'):
#     """ Fills missing values in numerical columns """
#     for column in df.select_dtypes(include=[np.number]).columns:
#         df[column] = df[column].fillna(getattr(df[column], strategy)())
#     return df
#
#
# def standardize_data(df, numerical_columns):
#     """ Standardizes numerical columns """
#     scaler = StandardScaler()
#     df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
#     return df
#
#
# def transform_data(train_x, train_y, test_x):
#     """ Transforms training and testing datasets by applying encoding, filling missing values, and standardization """
#     ordinal_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Location']
#     numerical_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
#                           'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
#                           'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
#
#     # Encode categorical and ordinal features
#     train_x, test_x = [encode_categorical(df.copy(), ordinal_features) for df in [train_x, test_x]]
#
#     # Optional: Apply target encoding or one-hot encoding to 'Location'
#     # Example of target encoding
#     # target_encoder = ce.TargetEncoder()
#     # train_x['Location'] = target_encoder.fit_transform(train_x['Location'], train_y)
#     # test_x['Location'] = target_encoder.transform(test_x['Location'])
#
#     # Handle missing values
#     train_x, test_x = [fill_missing_values(df) for df in [train_x, test_x]]
#
#     # Standardize numerical data
#     train_x, test_x = [standardize_data(df, numerical_features) for df in [train_x, test_x]]
#
#     # Drop non-feature columns if necessary
#     train_x = train_x.drop(columns=['row ID'])
#     test_x = test_x.drop(columns=['row ID'])
#
#     return train_x, train_y, test_x
