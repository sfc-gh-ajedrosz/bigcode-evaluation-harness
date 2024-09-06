import numpy as np
import pandas as pd


def transform_data(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, do_these=(0,)) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    numeric_cols = ['battery_power', 'clock_speed',  'fc',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time',
       ]
    categorical_cols = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']

    # Feature Engineering
    if 1 in do_these:
         train_x['camera_quality_index'] = train_x['fc'] + train_x['pc']
         test_x['camera_quality_index'] = test_x['fc'] + test_x['pc']
    if 2 in do_these:
        train_x['pixel_area'] = train_x['px_height'] * train_x['px_width']
        test_x['pixel_area'] = test_x['px_height'] * test_x['px_width']
    if 3 in do_these:
        train_x['screen_dimension_ratio'] = train_x['sc_h'] / train_x['sc_w']
        test_x['screen_dimension_ratio'] = test_x['sc_h'] / test_x['sc_w']
    if 4 in do_these:
        train_x['weight_to_screen_area_ratio'] = train_x['mobile_wt'] / (train_x['sc_h'] * train_x['sc_w'])
        test_x['weight_to_screen_area_ratio'] = test_x['mobile_wt'] / (test_x['sc_h'] * test_x['sc_w'])
    if 5 in do_these:
        train_x['memory_efficiency_index'] = train_x['int_memory'] / train_x['ram']
        test_x['memory_efficiency_index'] = test_x['int_memory'] / test_x['ram']
    if 6 in do_these:
       train_x['battery_efficiency'] = train_x['battery_power'] / train_x['talk_time']
       test_x['battery_efficiency'] = test_x['battery_power'] / test_x['talk_time']
    if 7 in do_these:
        # Outlier detection - Example using Z-score for 'ram'
        mean_ram = train_x['ram'].mean()
        std_ram = train_x['ram'].std()
        ram_z_score = (train_x['ram'] - mean_ram) / std_ram
        # Filter out extreme outliers, for example, using a threshold of Z-score > 3 or < -3
        train_x = train_x[np.abs(ram_z_score) < 3]
    return train_x, train_y, test_x

