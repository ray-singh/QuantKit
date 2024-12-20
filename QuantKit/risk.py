import pandas as pd
import numpy as np

def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) for a stock or portfolio.

    Args:
        returns (pd.Series): Time series of stock or portfolio returns.
        confidence_level (float): Confidence level for VaR calculation (default: 0.95).

    Returns:
        float: Value at Risk (VaR).
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("Input 'returns' must be a pandas Series.")

    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def conditional_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Compute Conditional Value at Risk (CVaR) for tail risk (Expected Shortfall).

    Args:
        returns (pd.Series): Time series of stock or portfolio returns.
        confidence_level (float): Confidence level for CVaR calculation (default: 0.95).

    Returns:
        float: Conditional Value at Risk (CVaR).
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("Input 'returns' must be a pandas Series.")

    var = value_at_risk(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    return cvar

def maximum_drawdown(data: pd.Series) -> float:
    """
    Evaluate the largest peak-to-trough loss in stock price or portfolio value.

    Args:
        data (pd.Series): Time series of stock or portfolio values.

    Returns:
        float: Maximum drawdown as a percentage.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Input 'data' must be a pandas Series.")

    cumulative_max = data.cummax()
    drawdown = (data - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown
