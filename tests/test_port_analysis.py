import unittest
import pandas as pd
from QuantKit.Portfolio import *

class TestPortfolioAnalysis(unittest.TestCase):

    def setUp(self):
        # Mock data for Portfolio instance
        self.mock_symbols = ['AAPL', 'TSLA', 'GOOGL']
        self.mock_data = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.005, 0.015],
            'TSLA': [0.03, -0.01, 0.025, 0.005],
            'GOOGL': [0.02, 0.015, -0.01, 0.01]
        })
        self.portfolio = Portfolio(stock_symbols=self.mock_symbols)
        self.portfolio.calculate_returns = lambda: self.mock_data.mean()
        self.portfolio.calculate_volatility = lambda: self.mock_data.std()

    def test_recommend_stocks_to_sell(self):
        # Mock fetch_company_info function
        def mock_fetch_company_info(ticker):
            data = {
                'AAPL': {"trailingPE": 25, "dividendYield": 0.01},
                'TSLA': {"trailingPE": 10, "dividendYield": None},
                'GOOGL': {"trailingPE": 30, "dividendYield": 0.005}
            }
            return data.get(ticker, {})

        # Inject mock function
        fetch_company_info = mock_fetch_company_info

        # Call the function and verify results
        results = recommend_stocks_to_sell(self.portfolio, pe_ratio_threshold=20, div_yield_threshold=2.0)
        self.assertEqual(len(results), 0)

    def test_optimize_portfolio(self):
        mock_data = type('MockData', (object,), {
            'stocks': {'AAPL': 100, 'TSLA': 50, 'GOOGL': 75},
            'returns': self.mock_data
        })()

        # Optimize portfolio using Sharpe ratio
        weights = optimize_portfolio(mock_data, method='sharpe')
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        self.assertTrue(all(0 <= w <= 1 for w in weights.values()))

if __name__ == '__main__':
    unittest.main()
