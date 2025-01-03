import unittest
from Stockify.data_fetching import *
import pandas as pd

class TestDataFetching(unittest.TestCase):

    def test_fetch_data_valid(self):
        data = fetch_data("AAPL", "2024-01-01", "2024-06-01")
        self.assertFalse(data.empty)
        self.assertIn('Close', data.columns)

    def test_fetch_data_invalid_ticker(self):
        data = fetch_data("INVALID", "2024-01-01", "2024-06-01")
        self.assertTrue(data.empty)

    def test_fetch_data_invalid_date(self):
        data = fetch_data("AAPL", "2024-13-01", "2024-06-01")  # Invalid month
        self.assertTrue(data.empty)

    def test_fetch_company_info_valid(self):
        info = fetch_company_info("AAPL")
        self.assertIn('Name', info)
        self.assertNotEqual(info['Name'], "N/A")

    def test_fetch_company_info_invalid(self):
        info = fetch_company_info("INVALID")
        self.assertEqual(info['Name'], "N/A")

    def test_fetch_live_price_valid(self):
        price = fetch_live_price("AAPL")
        self.assertGreater(price, 0)

    def test_fetch_live_price_invalid(self):
        price = fetch_live_price("INVALID")
        self.assertEqual(price, 0.0)

    def test_fetch_multiple_tickers_data(self):
        tickers = ["AAPL", "MSFT"]
        data = fetch_multiple_tickers_data(tickers, "2024-01-01", "2024-06-01")
        self.assertEqual(len(data), 2)
        self.assertFalse(data['AAPL'].empty)
        self.assertFalse(data['MSFT'].empty)

    def test_fetch_financials_valid(self):
        data = fetch_financials("AAPL")
        self.assertIn('annualReports', data)

    def test_fetch_financials_invalid(self):
        data = fetch_financials("INVALID")
        self.assertEqual(data, {})

    def test_get_stock_values_valid(self):
        values = get_stock_values("AAPL", "2024-01-01", "2024-06-01")
        self.assertIsInstance(values, pd.Series)
        self.assertFalse(values.empty)

    def test_get_stock_values_invalid(self):
        with self.assertRaises(KeyError):
            get_stock_values("INVALID", "2024-01-01", "2024-06-01")

    def test_get_daily_returns_valid(self):
        values = pd.Series([100, 105, 110, 120])
        returns = get_daily_returns(values)
        self.assertEqual(len(returns), 3)
        self.assertAlmostEqual(returns.iloc[0], 0.05)

    def test_get_daily_returns_invalid(self):
        with self.assertRaises(ValueError):
            get_daily_returns([100, 105, 110, 120])  # Not a Series


if __name__ == '__main__':
    unittest.main()
