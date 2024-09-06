import pandas as pd
# maajdl__yeh-concret-data.csv
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np


def transform_data_products(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, do_these=(0,)) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    features = train_x.columns.tolist()
    num_features = train_x.shape[1]
    for iter_i in do_these:
        if iter_i == 0:
            return train_x, train_y, test_x
        if iter_i in range(1, 1+num_features*num_features):
            iter_ci = iter_i-1
            from itertools import product
            product_cols = list(product(features, features))
            feat_1_name, feat_2_name = product_cols[iter_ci]
            train_x[f'feat_{iter_ci}'] = train_x[feat_1_name]/train_x[feat_2_name]
            test_x[f'feat_{iter_ci}'] = test_x[feat_1_name]/test_x[feat_2_name]
    return train_x, train_y, test_x
# (4, 31) --> 0.9308373407508865
# (4, 30, 33) --> 0.9317326812594511
# (4, 20, 33, 47) --> 0.930349728487479
# (4, 25, 30, 31) --> 0.9300087297116048
# (4, 30, 31, 33) --> 0.930953237220155
# (4, 30, 33, 44) --> 0.932013410530171
# (4, 31, 33, 44) --> 0.9316077145417071
# (4, 31) --> 0.9308373407508865
# (4, 30, 33) --> 0.9317326812594511
# (4, 20, 33, 47) --> 0.930349728487479
# (4, 25, 30, 31) --> 0.9300087297116048
# (4, 30, 31, 33) --> 0.930953237220155
# (4, 30, 33, 44) --> 0.932013410530171
# (4, 31, 33, 44) --> 0.9316077145417071
# Do (35, 40, 42), prod (4, 30, 33, 44) => 0.932013410530171
# Do (35, 40, 46), prod (4, 30, 33, 44) => 0.932013410530171
# Do (35, 40, 51), prod (4, 30, 33, 44) => 0.932013410530171
def transform_data_best_features(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, do_these=(0,)) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    do_these_multi = (7, 8, 14, 15, 23, 30, 31, 42, 44, 48, 49, 50, 51, 52, 57, 62) #(44, 57)
    do_these_div = []# (,)#(6, 44, 63)
    features = train_x.columns.tolist()
    num_features = train_x.shape[1]
    from itertools import product
    product_cols = list(product(features, features))
    for iter_i in do_these_multi:
        if iter_i == 0:
            return train_x, train_y, test_x
        if iter_i in range(1, 1+num_features*num_features):
            iter_ci = iter_i-1
            feat_1_name, feat_2_name = product_cols[iter_ci]
            train_x[f'feat_m_{iter_ci}'] = train_x[feat_1_name]*train_x[feat_2_name]
            test_x[f'feat_m_{iter_ci}'] = test_x[feat_1_name]*test_x[feat_2_name]

    for iter_i in do_these_div:
        if iter_i == 0:
            return train_x, train_y, test_x
        if iter_i in range(1, 1+num_features*num_features):
            iter_ci = iter_i-1
            feat_1_name, feat_2_name = product_cols[iter_ci]
            train_x[f'feat_{iter_ci}'] = train_x[feat_1_name]*train_x[feat_2_name]
            test_x[f'feat_{iter_ci}'] = test_x[feat_1_name]*test_x[feat_2_name]
    return train_x, train_y, test_x

#multiplication
#Do (7, 13, 56) => 3.079975420325003
# 2, 4, 6, 7, 11, 12, 13, 14, 15, 16, 21, 22, 25, 32, 33, 41, 42, 48, 49, 50, 56, 57
# [7, 8, 14, 15, 23, 30, 31, 42, 44, 48, 49, 50, 51, 52, 57, 62]
# Do (44, 57) => 0.9216724590164939
# divs:
# Do (6, 44, 63) => 0.9245711167687142
#

# below:
#Do (1, 5) => 0.923273198888533

def transform_data(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, do_these=(0,)) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    # numeric_cols = ['battery_power', 'clock_speed',  'fc',
    #    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
    #    'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time',
    #    ]
    # categorical_cols = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
    # Index(['cement', 'slag', 'flyash', 'water', 'superplasticizer',
    #        'coarseaggregate', 'fineaggregate', 'age'],
    #       dtype='object')
    # Feature Engineering
    import numpy as np

    # scaler = StandardScaler()
    # # pca = PCA(n_components=n_components)  # Adjust n_components based on your analysis needs
    # scaler.fit(train_x)
    # train_x = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)
    # test_x = pd.DataFrame(scaler.transform(test_x), columns=train_x.columns)

    if 1 in do_these:
        train_x['water_to_cement'] = train_x['water'] / train_x['cement']
        test_x['water_to_cement'] = test_x['water'] / test_x['cement']
    if 2 in do_these:
       train_x['coarse_to_fine_aggregate'] = train_x['coarseaggregate'] / train_x['fineaggregate']
       test_x['coarse_to_fine_aggregate'] = test_x['coarseaggregate'] / test_x['fineaggregate']
    if 3 in do_these:
        train_x['cement_percentage'] = train_x['cement'] / train_x.drop('Concrete compressive strength', axis=1).sum(axis=1)
        test_x['cement_percentage'] = test_x['cement'] / test_x.drop('Concrete compressive strength', axis=1).sum(axis=1)
    if 4 in do_these:
        train_x['cement_squared'] = train_x['cement'] ** 2
        test_x['cement_squared'] = test_x['cement'] ** 2
    if 5 in do_these:
         train_x['age_cement_interaction'] = train_x['cement'] * train_x['age']
         test_x['age_cement_interaction'] = test_x['cement'] * test_x['age']
    if 6 in do_these:
        train_x['log_cement'] = np.log1p(train_x['cement'])
        test_x['log_cement'] = np.log1p(test_x['cement'])

    if 7 in do_these:
        # Standardization Example (if using sklearn for modeling later)
        scaler = StandardScaler()
        scaler.fit(train_x[['cement', 'water', 'age']])
        train_x[['cement', 'water', 'age']] = scaler.transform(train_x[['cement', 'water', 'age']])
        test_x[['cement', 'water', 'age']] = scaler.transform(test_x[['cement', 'water', 'age']])
    if 8 in do_these:
        # Aggregate blend ratio
        train_x['aggregate_blend'] = (train_x['coarseaggregate'] + train_x['fineaggregate']) / train_x['cement']
        test_x['aggregate_blend'] = (test_x['coarseaggregate'] + test_x['fineaggregate']) / test_x['cement']

    if 9 in do_these:
        # Water to total aggregate ratio
        train_x['water_to_aggregate'] = train_x['water'] / (train_x['coarseaggregate'] + train_x['fineaggregate'])
        test_x['water_to_aggregate'] = test_x['water'] / (test_x['coarseaggregate'] + test_x['fineaggregate'])

    if 10 in do_these:
        # Age squared to capture non-linear aging effects
        train_x['age_squared'] = train_x['age'] ** 2
        test_x['age_squared'] = test_x['age'] ** 2

    if 11 in do_these:
        # Interaction between superplasticizer and water
        train_x['superplasticizer_water_interaction'] = train_x['superplasticizer'] * train_x['water']
        test_x['superplasticizer_water_interaction'] = test_x['superplasticizer'] * test_x['water']

    if 12 in do_these:
        # Ratio of slag to total components
        train_x['slag_ratio'] = train_x['slag'] / train_x[
            ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate']].sum(axis=1)
        test_x['slag_ratio'] = test_x['slag'] / test_x[
            ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate']].sum(axis=1)

    if 13 in do_these:
        # Fly ash contribution to the strength
        train_x['flyash_to_cement'] = train_x['flyash'] / train_x['cement']
        test_x['flyash_to_cement'] = test_x['flyash'] / test_x['cement']

    if 14 in do_these:
        import numpy as np
        # Log transformation of age to normalize distribution
        train_x['log_age'] = np.log1p(train_x['age'])
        test_x['log_age'] = np.log1p(test_x['age'])

    if 15 in do_these:
        # Inverse of superplasticizer to capture its effect when low
        train_x['inverse_superplasticizer'] = 1 / (
                    train_x['superplasticizer'] + 0.01)  # Adding small constant to avoid division by zero
        test_x['inverse_superplasticizer'] = 1 / (test_x['superplasticizer'] + 0.01)

    if 16 in do_these:
        # Detecting and removing outliers
        Q1 = train_x['Concrete compressive strength'].quantile(0.25)
        Q3 = train_x['Concrete compressive strength'].quantile(0.75)
        IQR = Q3 - Q1
        train_x = train_x[(train_x['Concrete compressive strength'] >= (Q1 - 1.5 * IQR)) & (
                    train_x['Concrete compressive strength'] <= (Q3 + 1.5 * IQR))]

    def remove_outliers_isoforest(df, contamination=0.05):
        iso = IsolationForest()  # Adjust contamination parameter as necessary
        yhat = iso.fit_predict(df.select_dtypes(include=[np.number]))
        mask = yhat != -1
        return mask

    def remove_outliers_zscore(df, z_threshold=3):
        import numpy as np
        z_scores = stats.zscore(df.select_dtypes(include=[np.number]))
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < z_threshold).all(axis=1)
        return filtered_entries.values

    def remove_outliers_iqr(df, outlier_threshold=1.5):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df < (Q1 - outlier_threshold * IQR)) | (df > (Q3 +outlier_threshold * IQR))).any(axis=1)
        return mask.values

    def remove_outliers_dbscan(df, eps=3):
        import numpy as np
        outlier_detection = DBSCAN(min_samples=2, eps=eps)
        clusters = outlier_detection.fit_predict(df.select_dtypes(include=[np.number]))
        mask = clusters != -1
        return mask

    if 17 in do_these:
        mask = remove_outliers_isoforest(train_x, contamination=0.05)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 18 in do_these:
        mask = remove_outliers_zscore(train_x, z_threshold=3.5)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 19 in do_these:
        mask = remove_outliers_zscore(train_x, z_threshold=4.5)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 20 in do_these:
        mask = remove_outliers_zscore(train_x, z_threshold=5)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 21 in do_these:
        mask = remove_outliers_zscore(train_x, z_threshold=5.5)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 22 in do_these:
        mask = remove_outliers_zscore(train_x, z_threshold=6.0)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 23 in do_these:
        mask = remove_outliers_zscore(train_x, z_threshold=6.5)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 24 in do_these:
        mask = remove_outliers_iqr(train_x, outlier_threshold=1.75)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 25 in do_these:
        mask = remove_outliers_iqr(train_x, outlier_threshold=2.0)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 26 in do_these:
        mask = remove_outliers_iqr(train_x, outlier_threshold=2.25)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 27 in do_these:
        mask = remove_outliers_iqr(train_x, outlier_threshold=2.5)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 28 in do_these:
        mask = remove_outliers_iqr(train_x, outlier_threshold=3.0)
        train_x = train_x[mask]
        train_y = train_y[mask]
    if 29 in do_these:
        mask = remove_outliers_iqr(train_x, outlier_threshold=3.5)
        train_x = train_x[mask]
        train_y = train_y[mask]

    if 30 in do_these:
        mask = remove_outliers_dbscan(train_x)
        train_x = train_x[mask]
        train_y = train_y[mask]

    def apply_smote(train_x, train_y):
        smote = SMOTE(random_state=42)
        train_x_resampled, train_y_resampled = smote.fit_resample(train_x, train_y)
        return train_x_resampled, train_y_resampled

    if 31 in do_these:
        train_x, train_y = apply_smote(train_x, train_y)

    import numpy as np

    def augment_data(train_x, train_y, augment_size=0.1, augment_strength=0.01, augment_target=False):
        additional_data = []
        additional_target = []
        np.random.seed(42)
        try:
            for i in range(int(len(train_x) * augment_size)):
                # Choose a random sample
                sample = train_x.iloc[i]     #.sample(n=1)
                # Add Gaussian noise with mean 0 and standard deviation that is 1% of the range of each feature
                noise = np.random.normal(0, augment_strength * (train_x.max() - train_x.min()), size=train_x.shape[1])
                new_sample = sample + noise
                additional_data.append(new_sample)
                if augment_target:
                    tgt = train_y.iloc[i] + (augment_strength*(train_y.max() - train_y.min()))
                else:
                    tgt = train_y.iloc[i]
                additional_target.append(tgt)
        except:
            pass
        additional_data = pd.DataFrame(additional_data)
        additional_target = pd.Series(additional_target)
        train_x_augmented = pd.concat([train_x, additional_data])
        train_y_augmented = pd.concat([train_y, additional_target])

        return train_x_augmented, train_y_augmented

    if 32 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.02, augment_strength=0.01)
    if 33 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.1, augment_strength=0.01)
    if 34 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.2, augment_strength=0.01)
    if 35 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.1, augment_strength=0.05)
    if 36 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.1, augment_strength=0.1)
    if 37 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.2, augment_strength=0.01)
    if 38 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.2, augment_strength=0.05)
    if 39 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.2, augment_strength=0.1)
    if 40 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.4, augment_strength=0.1)
    if 41 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.4, augment_strength=0.05)
    if 42 in do_these:
       train_x, train_y = augment_data(train_x, train_y, augment_size=0.02, augment_strength=0.01, augment_target=True)
    if 43 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.1, augment_strength=0.01, augment_target=True)
    if 44 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.2, augment_strength=0.01, augment_target=True)
    if 45 in do_these:
       train_x, train_y = augment_data(train_x, train_y, augment_size=0.1, augment_strength=0.05, augment_target=True)
    if 46 in do_these:
       train_x, train_y = augment_data(train_x, train_y, augment_size=0.1, augment_strength=0.1, augment_target=True)
    if 47 in do_these:
       train_x, train_y = augment_data(train_x, train_y, augment_size=0.2, augment_strength=0.01, augment_target=True)
    if 48 in do_these:
       train_x, train_y = augment_data(train_x, train_y, augment_size=0.2, augment_strength=0.05, augment_target=True)
    if 49 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.2, augment_strength=0.1, augment_target=True)
    if 50 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.4, augment_strength=0.1, augment_target=True)
    if 51 in do_these:
        train_x, train_y = augment_data(train_x, train_y, augment_size=0.4, augment_strength=0.05, augment_target=True)

    def do_pca(train_x, test_x, n_components=2):
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)  # Adjust n_components based on your analysis needs
        # scaler.fit(train_x)
        # train_x = scaler.transform(train_x)
        # test_x = scaler.transform(test_x)
        pca.fit(train_x)
        train_x = pca.transform(train_x)
        test_x = pca.transform(test_x)
        return train_x, test_x

    if 52 in do_these:
        train_x, test_x = do_pca(train_x, test_x, n_components=2)
    if 53 in do_these:
        train_x, test_x = do_pca(train_x, test_x, n_components=3)
    if 54 in do_these:
        train_x, test_x = do_pca(train_x, test_x, n_components=5)
    if 55 in do_these:
        train_x, test_x = do_pca(train_x, test_x, n_components=8)
    if 56 in do_these:  # based on https://www.kaggle.com/code/paulrogov/improved-1st-place-full-solution-3-50621
        def add_new_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
            admixtures = ['slag', 'flyash', 'coarseaggregate', 'fineaggregate', 'superplasticizer']
            train_df['admixtures'] = train_df[admixtures].sum(axis=1)  # kg in m3
            test_df['admixtures'] = test_df[admixtures].sum(axis=1)  # kg in m3

            admixtures_lowering_strength = ['slag', 'flyash', 'coarseaggregate', 'fineaggregate']
            train_df['admixtures_lowering_strength'] = train_df[admixtures_lowering_strength].sum(axis=1)  #  superplasticizer
            test_df['admixtures_lowering_strength'] = test_df[admixtures_lowering_strength].sum(axis=1)

            train_df['admixtures_lowering_strength_proportion'] = train_df[admixtures_lowering_strength].sum(axis=1) / (
                        train_df['cement'] + train_df['superplasticizer'])
            test_df['admixtures_lowering_strength_proportion'] = test_df[admixtures_lowering_strength].sum(axis=1) / (
                        test_df['cement'] + test_df['superplasticizer'])

            # density = Mass (components without water, kg) / volume (m3)
            feature_columns = train_df.columns.tolist()
            components_excl_water = [column for column in feature_columns if column not in ["water", "age"]]
            train_df['density'] = (train_df[components_excl_water].sum(axis=1) - train_df['water']) / 1000
            test_df['density'] = (test_df[components_excl_water].sum(axis=1) - test_df['water']) / 1000

            return (train_df, test_df)

        train_x, test_x = add_new_features(train_x, test_x)

    admixtures_lowering_strength = ['slag', 'flyash', 'coarseaggregate', 'fineaggregate']
    admixtures = ['slag', 'flyash', 'coarseaggregate', 'fineaggregate', 'superplasticizer']

    if 57 in do_these:
        train_x['admixtures'] = train_x[admixtures].sum(axis=1)  # kg in m3
        test_x['admixtures'] = test_x[admixtures].sum(axis=1)  # kg in m3
    if 58 in do_these:
        train_x['admixtures_lowering_strength'] = train_x[admixtures_lowering_strength].sum(
            axis=1)  # superplasticizer
        test_x['admixtures_lowering_strength'] = test_x[admixtures_lowering_strength].sum(axis=1)
    if 59 in do_these:
        train_x['admixtures_lowering_strength_proportion'] = train_x[admixtures_lowering_strength].sum(axis=1) / (
                train_x['cement'] + train_x['superplasticizer'])
        test_x['admixtures_lowering_strength_proportion'] = test_x[admixtures_lowering_strength].sum(axis=1) / (
                test_x['cement'] + test_x['superplasticizer'])
    if 60 in do_these:
        feature_columns = train_x.columns.tolist()
        components_excl_water = [column for column in feature_columns if column not in ["water", "age"]]
        train_x['density'] = (train_x[components_excl_water].sum(axis=1) - train_x['water']) / 1000
        test_x['density'] = (test_x[components_excl_water].sum(axis=1) - test_x['water']) / 1000
    if 61 in do_these:
        train_x["water_per_cement"] = train_x['water'] / train_x['cement']  # add this feature before csMPa column
        test_x["water_per_cement"] = test_x['water'] / test_x['cement']  # add this feature before csMPa column
    if 62 in do_these:
        import numpy as np
        # train_x['age'] = np.log10(train_x['age'])
        train_x['age'] = train_x['age'] ** 0.5
        # test_x['age'] = np.log10(test_x['age'])
        test_x['age'] = test_x['age'] ** 0.5
    if 63 in do_these:
        import numpy as np
        train_x["water_per_cement"] = train_x['water'] / train_x['cement']  # add this feature before csMPa column
        test_x["water_per_cement"] = test_x['water'] / test_x['cement']
        train_x['water_per_cement'] = np.log(train_x['water_per_cement'])
        test_x['water_per_cement'] = np.log(test_x['water_per_cement'])
    if 64 in do_these:
        train_x['superplasticizer'] = train_x['superplasticizer'] ** 0.5  # arcsinh also may be used
        test_x['superplasticizer'] = test_x['superplasticizer'] ** 0.5  # arcsinh also may be used
    if 65 in do_these:
        train_x['slag'] = train_x['slag'] ** 0.2
        test_x['slag'] = test_x['slag'] ** 0.2



    # if 42 in do_these:
    #     train_x, train_y = augment_data(train_x, train_y, augment_size=0.4, augment_strength=0.2)
        # Since this is an unsupervised augmentation, we assume target values (train_y) do not change and simply repeat them.
        # train_y = pd.concat([train_y] * (1 + 0.1))  # Assuming augmentation size is 10%

    # if 19 in do_these:
    # if 19 in do_these:
    # if 19 in do_these:
    # if 11 in do_these:
    # if 12 in do_these:


        # # Outlier detection - Example using Z-score for 'ram'
        # mean_ram = train_x['ram'].mean()
        # std_ram = train_x['ram'].std()
        # ram_z_score = (train_x['ram'] - mean_ram) / std_ram
        # # Filter out extreme outliers, for example, using a threshold of Z-score > 3 or < -3
        # train_x = train_x[np.abs(ram_z_score) < 3]
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)
    test_x = pd.DataFrame(scaler.transform(test_x), columns=train_x.columns)
    return train_x, train_y, test_x
# R2, seeded
# Do (1, 35, 42) => 0.9305606291309698
# Do (2, 8, 32, 40, 46) => 0.933905798433382


# Do (1, 2, 5, 12, 35) => 2.9943383123197824
# Do (1, 2, 5, 12, 32, 38) => 2.9949527609256825
# Do (1, 2, 5, 12, 32, 38) => 2.9949527609256825
# Do (1, 5, 9, 12, 32, 33, 38) => 2.9924046260804467
# Do (1, 2, 5, 9, 12, 32, 33, 35, 43) => 2.972135959049625


# Do (1, 2, 5, 19, 43) => 3.027962342625688
# Do (1, 2, 5, 12, 43) => 3.0171363278849954

# Do (9, 43, 63) => 2.9170729637633803
