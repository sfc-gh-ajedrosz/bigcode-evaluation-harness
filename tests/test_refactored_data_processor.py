import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from your_module import DataProcessor  # replace with your actual module name

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to hold the CSV files
        self.temp_dir = TemporaryDirectory()
        self.data_processor = DataProcessor(Path(self.temp_dir.name))

        # Example DataFrames
        self.numeric_df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "ds_f_eng__split": ["train", "train", "test", "test", "train"],
            "ds_f_eng__target__response": [1, 0, 1, 0, 1]
        })

        self.mixed_df = pd.DataFrame({
            "A": [1, 2, "missing", 4, 5],
            "B": [5, np.nan, "missing", 2, "string"],
            "C": ["cat1", "cat2", "cat3", np.nan, "cat2"],
            "ds_f_eng__split": ["train", "train", "test", "test", "train"],
            "ds_f_eng__target__response": [1, 0, 1, 0, 1]
        })

        self.categorical_df = pd.DataFrame({
            "A": ["low", "medium", "high", "low", "medium"],
            "B": ["cat1", "cat2", "cat3", "cat1", np.nan],
            "ds_f_eng__split": ["train", "train", "test", "test", "train"],
            "ds_f_eng__target__response": [1, 0, 1, 0, 1]
        })

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_load_dataframe_numeric(self):
        # Test for purely numeric data
        self.numeric_df.to_csv(Path(self.temp_dir.name) / "numeric.csv", index=False)
        test_x, test_target, train_x, train_target = self.data_processor.load_dataframe(Path("numeric.csv"))
        self.assertEqual(len(train_x), 3)
        self.assertEqual(len(test_x), 2)
        self.assertTrue(train_x.isnumeric)
        self.assertTrue(test_x.isnumeric)

    def test_load_dataframe_mixed(self):
        # Test for mixed data types
        self.mixed_df.to_csv(Path(self.temp_dir.name) / "mixed.csv", index=False)
        test_x, test_target, train_x, train_target = self.data_processor.load_dataframe(Path("mixed.csv"))
        self.assertEqual(len(train_x), 3)
        self.assertEqual(len(test_x), 2)
        self.assertIn("missing", train_x.values)
        self.assertIn("string", test_x.values)

    def test_load_dataframe_categorical(self):
        # Test for categorical data
        self.categorical_df.to_csv(Path(self.temp_dir.name) / "categorical.csv", index=False)
        test_x, test_target, train_x, train_target = self.data_processor.load_dataframe(Path("categorical.csv"))
        self.assertEqual(len(train_x), 3)
        self.assertEqual(len(test_x), 2)
        self.assertTrue(train_x["A"].dtype == 'object')
        self.assertTrue(test_x["B"].dtype == 'object')

    def test_identify_categorical_columns(self):
        # Test identify_categorical_columns
        cat_columns = DataProcessor.identify_categorical_columns(self.mixed_df, unique_val_threshold=3)
        self.assertIn("C", cat_columns)
        self.assertNotIn("A", cat_columns)

    def test_identify_categorical_columns_new(self):
        # Test identify_categorical_columns_new
        cat_columns = DataProcessor.identify_categorical_columns_new(self.mixed_df, unique_val_threshold=3)
        self.assertIn("C", cat_columns)
        self.assertIn("B", cat_columns)

    def test_one_hot_encode(self):
        # Test one-hot encoding on the categorical dataframe
        train_x, test_x = self.data_processor.one_hot_encode(self.categorical_df[self.categorical_df['ds_f_eng__split'] == 'train'],
                                                             self.categorical_df[self.categorical_df['ds_f_eng__split'] == 'test'],
                                                             ["A", "B"], fit_on_combined=True)
        self.assertIn("A_low", train_x.columns)
        self.assertIn("A_medium", train_x.columns)
        self.assertNotIn("B_nan", test_x.columns)

    def test_prepare_train_test_split(self):
        # Test prepare_train_test_split on the mixed dataframe
        test_x, test_target, train_x, train_target = self.data_processor.prepare_train_test_split(
            self.mixed_df[self.mixed_df['ds_f_eng__split'] == 'test'],
            self.mixed_df[self.mixed_df['ds_f_eng__split'] == 'train']
        )
        self.assertEqual(train_x.shape[1], 2)
        self.assertEqual(test_x.shape[1], 2)

if __name__ == "__main__":
    unittest.main()
