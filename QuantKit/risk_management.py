import numpy as np
import pandas as pd


# Calculate Value at Risk (VaR) at a given confidence level
def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns (pd.Series): A series of daily returns.
        confidence_level (float): Confidence level for the VaR calculation (default is 95%).

    Returns:
        float: The Value at Risk (VaR) at the given confidence level.
    """
    if returns.empty:
        raise ValueError("Returns data is empty")

    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var


# Calculate Conditional Value at Risk (CVaR) at a given confidence level
def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

    Args:
        returns (pd.Series): A series of daily returns.
        confidence_level (float): Confidence level for the CVaR calculation (default is 95%).

    Returns:
        float: The Conditional Value at Risk (CVaR) at the given confidence level.
    """
    if returns.empty:
        raise ValueError("Returns data is empty")

    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()  # Average of returns that are less than or equal to VaR
    return cvar


# Calculate Maximum Drawdown (MDD)
def calculate_max_drawdown(data: pd.Series) -> float:
    """
    Calculate Maximum Drawdown (MDD), the largest peak-to-trough drop.

    Args:
        data (pd.Series): Stock price or portfolio value data.

    Returns:
        float: The Maximum Drawdown.
    """
    if data.empty:
        raise ValueError("Data is empty")

    # Calculate the cumulative returns
    cumulative_returns = data / data.cummax() - 1
    max_drawdown = cumulative_returns.min()
    return max_drawdown


# Calculate Volatility (Standard Deviation of Returns)
def calculate_volatility(returns: pd.Series, window: int = 252) -> float:
    """
    Calculate the volatility (standard deviation of returns) over a specified window.

    Args:
        returns (pd.Series): A series of daily returns.
        window (int): The window over which to calculate volatility (default is 252, for one trading year).

    Returns:
        float: The volatility over the given window.
    """
    if returns.empty:
        raise ValueError("Returns data is empty")

    volatility = returns.rolling(window=window).std().iloc[-1]  # Latest volatility value
    return volatility


# Calculate Sharpe Ratio
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe Ratio, a measure of risk-adjusted return.

    Args:
        returns (pd.Series): A series of daily returns.
        risk_free_rate (float): The risk-free rate (default is 2% annualized).

    Returns:
        float: The Sharpe ratio.
    """
    if returns.empty:
        raise ValueError("Returns data is empty")

    excess_returns = returns.mean() - risk_free_rate / 252  # Convert annual risk-free rate to daily
    volatility = calculate_volatility(returns)
    sharpe_ratio = excess_returns / volatility
    return sharpe_ratio


if __name__ == "__main__":
    # Example usage with random data (replace with actual stock/portfolio returns)
    data = pd.Series([100, 102, 101, 105, 110, 107, 108, 109, 112, 115])

    returns = data.pct_change().dropna()  # Daily returns

    print("----- Value at Risk (VaR) -----")
    var = calculate_var(returns)
    print(f"VaR: {var:.2f}")

    print("\n----- Conditional Value at Risk (CVaR) -----")
    cvar = calculate_cvar(returns)
    print(f"CVaR: {cvar:.2f}")

    print("\n----- Maximum Drawdown (MDD) -----")
    max_drawdown = calculate_max_drawdown(data)
    print(f"Maximum Drawdown: {max_drawdown:.2f}")

    print("\n----- Volatility -----")
    volatility = calculate_volatility(returns)
    print(f"Volatility: {volatility:.2f}")

    print("\n----- Sharpe Ratio -----")
    sharpe_ratio = calculate_sharpe_ratio(returns)
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
