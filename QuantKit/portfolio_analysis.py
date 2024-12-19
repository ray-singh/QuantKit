# portfolio_analysis.py
import numpy as np
import pandas as pd
from Portfolio import Portfolio


def calculate_daily_returns(portfolio: Portfolio) -> pd.DataFrame:
    """
    Compute daily returns for the portfolio based on individual stock data.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.

    Returns:
        pd.DataFrame: DataFrame containing daily returns for each stock in the portfolio.
    """
    return portfolio.returns

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


def calculate_annualized_return(portfolio: Portfolio) -> float:
    """
    Calculate the annualized return of the portfolio.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.

    Returns:
        float: Annualized return of the portfolio.
    """
    daily_returns = portfolio.calculate_returns()
    annualized_return = daily_returns.mean() * 252  # Assume 252 trading days in a year
    return annualized_return


def calculate_volatility(portfolio: Portfolio) -> float:
    """
    Calculate the volatility (standard deviation) of the portfolio.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.

    Returns:
        float: Annualized volatility of the portfolio.
    """
    return portfolio.calculate_volatility()


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
    volatility = calculate_volatility(portfolio)

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


def calculate_portfolio_return(portfolio: Portfolio) -> float:
    """
    Calculate the total return of the portfolio based on its weights and stock returns.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.

    Returns:
        float: Total return of the portfolio.
    """
    return portfolio.calculate_portfolio_return()
