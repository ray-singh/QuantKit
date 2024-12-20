import pandas as pd
import numpy as np

# Convert stock prices to percentage returns
def convert_to_returns(data: pd.Series) -> pd.Series:
    """
    Convert stock prices to percentage returns.

    Args:
        data (pd.Series): Series of stock prices.

    Returns:
        pd.Series: Series of percentage returns.
    """
    return data.pct_change().dropna()


# Save stock/portfolio data to CSV
def save_data_to_csv(data: pd.DataFrame, filename: str) -> None:
    """
    Save stock/portfolio data to a CSV file.

    Args:
        data (pd.DataFrame): Data to save.
        filename (str): Name of the CSV file.
    """
    data.to_csv(filename, index=True)
    print(f"Data saved to {filename}")


# Load stock/portfolio data from a CSV file
def load_data_from_csv(filename: str) -> pd.DataFrame:
    """
    Load stock/portfolio data from a CSV file.

    Args:
        filename (str): Name of the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    print(f"Data loaded from {filename}")
    return data


# Normalize stock prices for comparison
def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize stock prices to start at 1 for comparison.

    Args:
        data (pd.DataFrame): DataFrame with stock prices.

    Returns:
        pd.DataFrame: Normalized stock price data.
    """
    return data / data.iloc[0]


# Compute rolling metrics (e.g., mean, std deviation, etc.)
def compute_rolling_metrics(data: pd.Series, metric: str = 'mean', window: int = 30) -> pd.Series:
    """
    Compute rolling metrics such as mean or standard deviation.

    Args:
        data (pd.Series): Series of stock prices or returns.
        metric (str): Metric to compute ('mean' or 'std').
        window (int): Rolling window size.

    Returns:
        pd.Series: Rolling metric values.
    """
    if metric == 'mean':
        return data.rolling(window=window).mean()
    elif metric == 'std':
        return data.rolling(window=window).std()
    else:
        raise ValueError("Invalid metric. Supported metrics: 'mean', 'std'.")
