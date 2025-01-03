import pandas as pd
import numpy as np
from typing import Union

def validate_input(data: Union[pd.Series, np.ndarray], name: str):
    """Validate and preprocess input data."""
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    elif not isinstance(data, pd.Series):
        raise ValueError(f"Input '{name}' must be a pandas Series or numpy array.")
    data = data.dropna()
    if len(data) < 2:
        raise ValueError(f"Input '{name}' must contain at least 2 valid data points.")
    return data


def value_at_risk(returns: Union[pd.Series, np.ndarray], confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) for a stock or portfolio.

    Args:
        returns (pd.Series or np.ndarray): Time series of stock or portfolio returns.
        confidence_level (float): Confidence level for VaR calculation (default: 0.95).

    Returns:
        float: Value at Risk (VaR).
    """
    returns = validate_input(returns, "returns")
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")

    var = np.quantile(returns, 1 - confidence_level)
    return round(var, 6)


def conditional_value_at_risk(returns: Union[pd.Series, np.ndarray], confidence_level: float = 0.95) -> float:
    """
    Compute Conditional Value at Risk (CVaR) for tail risk (Expected Shortfall).

    Args:
        returns (pd.Series or np.ndarray): Time series of stock or portfolio returns.
        confidence_level (float): Confidence level for CVaR calculation (default: 0.95).

    Returns:
        float: Conditional Value at Risk (CVaR).
    """
    returns = validate_input(returns, "returns")
    var = value_at_risk(returns, confidence_level)
    tail_returns = returns[returns <= var]
    if len(tail_returns) == 0:  # Handle cases where no tail losses exist
        return np.nan
    cvar = tail_returns.mean()
    return round(cvar, 6)


def maximum_drawdown(data: Union[pd.Series, np.ndarray], as_percentage: bool = True) -> float:
    """
    Evaluate the largest peak-to-trough loss in stock price or portfolio value.

    Args:
        data (pd.Series or np.ndarray): Time series of stock or portfolio values.
        as_percentage (bool): Whether to return drawdown as a percentage (default: True).

    Returns:
        float: Maximum drawdown as a percentage or decimal.
    """
    data = validate_input(data, "data")
    cumulative_max = data.cummax()
    drawdown = (data - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return round(max_drawdown * 100, 2) if as_percentage else round(max_drawdown, 6)
