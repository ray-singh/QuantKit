import unittest
import numpy as np
import pandas as pd
from QuantKit.Portfolio.Portfolio import Portfolio
from QuantKit.Portfolio.portfolio_analysis import (recommend_stocks_to_sell,
                                                   optimize_portfolio,
                                                   calculate_annualized_return,
                                                   sharpe_ratio, sortino_ratio)

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

    def test_calculate_annualized_return(self):
        # Mock the Portfolio class
        portfolio = Portfolio(stock_symbols=['AAPL', 'TSLA', 'GOOGL'])

        # Simulate the return values from the calculate_returns method
        portfolio.calculate_returns = lambda: pd.Series([0.01, 0.02, 0.015, -0.01])

        # Call the function to calculate annualized return
        annualized_return = calculate_annualized_return(portfolio)

        # Verify the result (252 trading days assumed)
        expected_return = 0.01 * 252

        # Since the result is a float and expected return is also a float, no need for Series comparison
        self.assertAlmostEqual(annualized_return, expected_return, places=2)

    def test_sharpe_ratio(self):
        result = sharpe_ratio(self.portfolio, risk_free_rate=0.01)
        expected_return = self.mock_data.mean() * 252
        volatility = self.mock_data.std()
        expected_sharpe = ((expected_return - 0.01) / volatility).mean()
        self.assertAlmostEqual(result, expected_sharpe, places=4)

    def test_sortino_ratio(self):
        result = sortino_ratio(self.portfolio, risk_free_rate=0.01)
        downside_returns = self.mock_data[self.mock_data < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        expected_return = self.mock_data.mean() * 252
        expected_sortino = ((expected_return - 0.01) / downside_deviation).mean()
        self.assertAlmostEqual(result, expected_sortino, places=4)

if __name__ == '__main__':
    unittest.main()
