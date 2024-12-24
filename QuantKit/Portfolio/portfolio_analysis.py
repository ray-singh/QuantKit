# portfolio_analysis.py
import numpy as np
import pandas as pd
from Portfolio import Portfolio
from QuantKit.data_fetching import fetch_data, fetch_company_info, fetch_live_price
from typing import List, Dict


def recommend_stocks_to_sell(self, pe_ratio_threshold: float, div_yield_threshold: float) -> List[Dict]:
    """
    Identify underperforming stocks in the portfolio that should be sold off.

    Args:
        pe_ratio_threshold (float): Maximum acceptable P/E ratio.
        div_yield_threshold (float): Minimum acceptable dividend yield.

    Returns:
        List[Dict]: List of dictionaries containing underperforming stock information.
    """
    underperforming_stocks = []

    for ticker in self.stock_symbols:
        info = fetch_company_info(ticker)
        if not info:
            continue  # Skip if unable to fetch info

        pe_ratio = info.get("trailingPE", None)
        div_yield = info.get("dividendYield", None)

        # Normalize dividend yield percentage if available
        div_yield = div_yield * 100 if div_yield is not None else None

        # Check if the stock is underperforming
        if (pe_ratio and pe_ratio > pe_ratio_threshold) or (div_yield and div_yield < div_yield_threshold):
            underperforming_stocks.append({
                "Ticker": ticker,
                "Name": info.get("Name", "N/A"),
                "P/E Ratio": pe_ratio,
                "Dividend Yield (%)": div_yield,
                "Sector": info.get("Sector", "N/A"),
                "Industry": info.get("Industry", "N/A")
            })
    return underperforming_stocks

def optimize_portfolio(data, method='sharpe'):
    """
    Optimize the portfolio using mean-variance optimization.

    Parameters:
    method (str, optional): The optimization method. 'sharpe' for maximizing the Sharpe ratio (default), or 'volatility' for minimizing volatility.

    Returns:
    dict: A dictionary of optimized portfolio weights.
    """
    from scipy.optimize import minimize

    num_assets = len(data.stocks)
    initial_weights = np.ones(num_assets) / num_assets  # Initial equal weights
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling

    def objective(weights):
        weighted_returns = data.returns.dot(weights)
        if method == 'sharpe':
            excess_returns = weighted_returns.mean() - 0.01  # Default risk-free rate
            return -excess_returns / weighted_returns.std()  # Minimize negative Sharpe ratio
        elif method == 'volatility':
            return np.sqrt(np.dot(weights.T, np.dot(data.returns.cov() * 252, weights)))  # Minimize volatility

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x
    return dict(zip(data.stocks.keys(), optimized_weights))


def calculate_annualized_return(portfolio: Portfolio) -> pd.Series:
    """
    Calculate the annualized return of the portfolio.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.

    Returns:
        pd.Series: Annualized return of the portfolio.
    """
    daily_returns = portfolio.calculate_returns()
    annualized_return = daily_returns.mean() * 252  # Assume 252 trading days in a year
    return annualized_return


def sharpe_ratio(portfolio: Portfolio, risk_free_rate=0.01) -> float:
    """
    Calculate the Sharpe Ratio for the portfolio, which measures risk-adjusted return.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.
        risk_free_rate (float): Risk-free rate for comparison (default is 1%).

    Returns:
        float: Sharpe Ratio for the portfolio.
    """
    daily_returns = portfolio.calculate_returns()
    annualized_return = calculate_annualized_return(portfolio)
    volatility = portfolio.calculate_volatility()

    # Sharpe Ratio formula: (mean return - risk-free rate) / volatility
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    return sharpe_ratio


def sortino_ratio(portfolio: Portfolio, risk_free_rate=0.01) -> float:
    """
    Calculate the Sortino Ratio for the portfolio, which focuses on downside risk.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.
        risk_free_rate (float): Risk-free rate for comparison (default is 1%).

    Returns:
        float: Sortino Ratio for the portfolio.
    """
    daily_returns = portfolio.calculate_returns()
    downside_returns = daily_returns[daily_returns < 0]
    annualized_return = calculate_annualized_return(portfolio)

    # Calculate downside deviation (standard deviation of negative returns)
    downside_deviation = downside_returns.std() * np.sqrt(252)

    # Sortino Ratio formula: (mean return - risk-free rate) / downside deviation
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
    return sortino_ratio


#--------------------Example Use Cases-----------------------------#

# Assuming we have a portfolio instance with a list of stock symbols
portfolio = Portfolio(stock_symbols=['AAPL', 'TSLA', 'GOOGL', 'AMZN'])

# Specify the P/E ratio and dividend yield thresholds
pe_ratio_threshold = 20.0  # Maximum acceptable P/E ratio
div_yield_threshold = 3.0  # Minimum acceptable dividend yield (%)

# Call the function to get the underperforming stocks
underperforming_stocks = recommend_stocks_to_sell(portfolio, pe_ratio_threshold, div_yield_threshold)

# Output the results
print("Stocks to consider selling:")
for stock in underperforming_stocks:
    print(stock)

# Call the function to optimize portfolio based on the Sharpe ratio
optimized_weights = optimize_portfolio(portfolio, method='sharpe')

# Output the optimized weights for each stock
print("Optimized Portfolio Weights (Sharpe Ratio):")
for stock, weight in optimized_weights.items():
    print(f"{stock}: {weight:.2f}")

# New portfolio instance
portfolio = Portfolio(stock_symbols=['AAPL', 'TSLA', 'GOOGL', 'AMZN'])

# Call the function to calculate the annualized return
annualized_return = calculate_annualized_return(portfolio)

# Output the result
print(f"Annualized Return of the Portfolio: {annualized_return}%")

# Call the function to calculate the Sharpe Ratio
sharpe_ratio_value = sharpe_ratio(portfolio, risk_free_rate=0.01)

# Output the result
print(f"Sharpe Ratio of the Portfolio: {sharpe_ratio_value}")

# Call the function to calculate the Sortino Ratio
sortino_ratio_value = sortino_ratio(portfolio, risk_free_rate=0.01)

# Output the result
print(f"Sortino Ratio of the Portfolio: {sortino_ratio_value}")