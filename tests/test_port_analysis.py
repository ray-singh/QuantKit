import unittest
import pandas as pd
import numpy as np
from QuantKit.Portfolio.Portfolio import Portfolio
from QuantKit.Portfolio.portfolio_analysis import (
    recommend_stocks_to_sell,
    optimize_portfolio,
    calculate_annualized_return,
    sharpe_ratio,
    sortino_ratio
)

class TestPortfolioAnalysis(unittest.TestCase):

    def setUp(self):
        # Create mock portfolio
        self.data = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.005, 0.01, 0.015],
            'MSFT': [0.015, -0.01, 0.005, 0.02, 0.01]
        }, index=pd.date_range('2024-01-01', periods=5))
        self.portfolio = Portfolio(['AAPL', 'MSFT'], self.data)

    def test_recommend_stocks_to_sell(self):
        # Mock fetch_company_info to simulate API responses
        def mock_fetch_company_info(ticker):
            info = {
                'AAPL': {'trailingPE': 25, 'dividendYield': 0.01, 'Name': 'Apple', 'Sector': 'Tech', 'Industry': 'Hardware'},
                'MSFT': {'trailingPE': 15, 'dividendYield': 0.02, 'Name': 'Microsoft', 'Sector': 'Tech', 'Industry': 'Software'}
            }
            return info.get(ticker, {})

        # Replace the real function with mock
        from data_fetching import fetch_company_info
        fetch_company_info = mock_fetch_company_info

        # Test recommendation logic
        results = recommend_stocks_to_sell(self.portfolio, 20, 1.5)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)  # Only AAPL should be recommended
        self.assertEqual(results[0]['Ticker'], 'AAPL')

    def test_optimize_portfolio(self):
        optimized_weights = optimize_portfolio(self.portfolio, method='sharpe')
        self.assertIsInstance(optimized_weights, dict)
        self.assertAlmostEqual(sum(optimized_weights.values()), 1.0)  # Weights sum to 1

    def test_calculate_annualized_return(self):
        annual_return = calculate_annualized_return(self.portfolio)
        self.assertIsInstance(annual_return, pd.Series)
        self.assertGreaterEqual(annual_return, -1.0)  # Reasonable range
        self.assertLessEqual(annual_return, 1.0)

    def test_sharpe_ratio(self):
        sharpe = sharpe_ratio(self.portfolio, risk_free_rate=0.01)
        self.assertIsInstance(sharpe, float)
        self.assertGreaterEqual(sharpe, -10)
        self.assertLessEqual(sharpe, 10)

    def test_sortino_ratio(self):
        sortino = sortino_ratio(self.portfolio, risk_free_rate=0.01)
        self.assertIsInstance(sortino, float)
        self.assertGreaterEqual(sortino, -10)
        self.assertLessEqual(sortino, 10)


if __name__ == '__main__':
    unittest.main()
