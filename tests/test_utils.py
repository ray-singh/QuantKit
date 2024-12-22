import unittest
import pandas as pd
import numpy as np
from QuantKit.utils import (
    convert_to_returns,
    save_data_to_csv,
    load_data_from_csv,
    normalize_data,
    compute_rolling_metrics
)
import os


class TestUtilsFunctions(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.sample_data = pd.Series([100, 105, 110, 120, 115])
        self.sample_df = pd.DataFrame({'Close': [100, 105, 110, 120, 115]})
        self.test_csv = 'test_data.csv'

    def tearDown(self):
        # Remove test CSV file if it exists
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    # Test convert_to_returns
    def test_convert_to_returns(self):
        returns = convert_to_returns(self.sample_data)
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(self.sample_data) - 1)
        self.assertAlmostEqual(returns.iloc[0], 0.05)  # 5% return

    # Test save_data_to_csv
    def test_save_data_to_csv(self):
        save_data_to_csv(self.sample_df, self.test_csv)
        self.assertTrue(os.path.exists(self.test_csv))

    # Test load_data_from_csv
    def test_load_data_from_csv(self):
        save_data_to_csv(self.sample_df, self.test_csv)
        loaded_data = load_data_from_csv(self.test_csv)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertTrue('Close' in loaded_data.columns)

    # Test normalize_data
    def test_normalize_data(self):
        normalized = normalize_data(self.sample_df)
        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertAlmostEqual(normalized['Close'].iloc[0], 1.0)

    # Test compute_rolling_metrics
    def test_compute_rolling_metrics_mean(self):
        rolling_mean = compute_rolling_metrics(self.sample_data, metric='mean', window=2)
        self.assertIsInstance(rolling_mean, pd.Series)
        self.assertEqual(len(rolling_mean), len(self.sample_data))
        self.assertTrue(np.isnan(rolling_mean.iloc[0]))  # First value should be NaN
        self.assertAlmostEqual(rolling_mean.iloc[1], 102.5)

    def test_compute_rolling_metrics_std(self):
        rolling_std = compute_rolling_metrics(self.sample_data, metric='std', window=2)
        self.assertIsInstance(rolling_std, pd.Series)
        self.assertEqual(len(rolling_std), len(self.sample_data))

    def test_compute_rolling_metrics_invalid_metric(self):
        with self.assertRaises(ValueError):
            compute_rolling_metrics(self.sample_data, metric='invalid', window=2)


if __name__ == '__main__':
    unittest.main()
