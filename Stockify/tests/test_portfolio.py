import unittest
import pandas as pd
from Stockify.Portfolio import Portfolio

class TestPortfolio(unittest.TestCase):

    def setUp(self):
        # Setup for tests
        self.symbols = ['AAPL', 'MSFT']
        self.start_date = '2020-01-01'
        self.end_date = '2024-01-01'
        self.weights = {'AAPL': 0.6, 'MSFT': 0.4}
        self.portfolio = Portfolio(self.symbols, self.start_date, self.end_date, self.weights)

    def test_initialization(self):
        # Test portfolio initialization
        self.assertEqual(self.portfolio.stock_symbols, self.symbols)
        self.assertEqual(self.portfolio.start_date, self.start_date)
        self.assertEqual(self.portfolio.end_date, self.end_date)
        self.assertEqual(self.portfolio.weights, self.weights)

    def test_fetch_stocks_data(self):
        # Test fetch_stocks_data output
        data = self.portfolio.fetch_stocks_data()
        self.assertIsInstance(data, dict)
        for symbol in self.symbols:
            self.assertIn(symbol, data)
            self.assertIsInstance(data[symbol], pd.DataFrame)

    def test_calculate_returns(self):
        # Test returns calculation
        returns = self.portfolio.calculate_returns()
        self.assertIsInstance(returns, pd.DataFrame)
        for symbol in self.symbols:
            self.assertIn(symbol, returns)
            self.assertIsInstance(returns, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
