import pandas as pd
import numpy as np


# Calculate Simple Moving Average (SMA)
def calculate_sma(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA).

    Args:
        data (pd.DataFrame): Stock data with a 'Close' column.
        window (int): The period over which to calculate the average.

    Returns:
        pd.Series: A series of SMA values.
    """
    return data['Close'].rolling(window=window).mean()


# Calculate Exponential Moving Average (EMA)
def calculate_ema(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).

    Args:
        data (pd.DataFrame): Stock data with a 'Close' column.
        window (int): The period over which to calculate the EMA.

    Returns:
        pd.Series: A series of EMA values.
    """
    return data['Close'].ewm(span=window, adjust=False).mean()


# Calculate Relative Strength Index (RSI)
def calculate_rsi(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    Args:
        data (pd.DataFrame): Stock data with a 'Close' column.
        window (int): The period over which to calculate the RSI.

    Returns:
        pd.Series: A series of RSI values.
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(data: pd.DataFrame,
                   ticker: str,  # Specify the ticker
                   short_window: int = 12,
                   long_window: int = 26,
                   signal_window: int = 9) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD) and its Signal line.

    Args:
        data (pd.DataFrame): Stock data with MultiIndex columns.
        ticker (str): Ticker symbol to calculate MACD for.
        short_window (int): The period for the short-term EMA.
        long_window (int): The period for the long-term EMA.
        signal_window (int): The period for the Signal line (EMA of MACD).

    Returns:
        pd.DataFrame: A DataFrame containing MACD and Signal line.
    """
    # Access the 'Close' prices for the specific ticker
    close_prices = data['Close']

    # Calculate EMAs
    short_ema = close_prices.ewm(span=short_window, adjust=False).mean()
    long_ema = close_prices.ewm(span=long_window, adjust=False).mean()

    # Calculate MACD and Signal Line
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()

    # Return as DataFrame
    return pd.DataFrame({'MACD': macd, 'Signal Line': signal_line}, index=data.index)


# Calculate Bollinger Bands
def calculate_bollinger_bands(data: pd.DataFrame,
                              ticker: str,
                              window: int = 20,
                              num_std_dev: int = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        data (pd.DataFrame): Stock data with MultiIndex columns.
        ticker (str): Ticker symbol to calculate Bollinger Bands for.
        window (int): Moving average window size.
        num_std_dev (int): Number of standard deviations for bands.

    Returns:
        pd.DataFrame: Upper, Middle (SMA), and Lower bands.
    """
    # Extract 'Close' prices for the specific ticker
    close_prices = data['Close']

    # Calculate SMA and rolling standard deviation
    sma = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()

    # Calculate Bollinger Bands
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)

    # Return DataFrame with index explicitly set
    return pd.DataFrame({'Upper Band': upper_band,
                         'Middle Band (SMA)': sma,
                         'Lower Band': lower_band},
                         index=close_prices.index)

# Calculate On-Balance Volume (OBV)
def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV).

    Args:
        data (pd.DataFrame): Stock data with 'Close' and 'Volume' columns.

    Returns:
        pd.Series: A series of OBV values.
    """
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i - 1]:
            obv.append(obv[-1] + data['Volume'][i])
        elif data['Close'][i] < data['Close'][i - 1]:
            obv.append(obv[-1] - data['Volume'][i])
        else:
            obv.append(obv[-1])

    return pd.Series(obv, index=data.index)


# Calculate Stochastic Oscillator (%K and %D)
def calculate_stochastic_oscillator(data: pd.DataFrame, window: int = 14, signal_window: int = 3) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator (%K and %D).

    Args:
        data (pd.DataFrame): Stock data with 'High', 'Low', and 'Close' columns.
        window (int): The period over which to calculate the %K.
        signal_window (int): The period for the %D line, which is the moving average of %K.

    Returns:
        pd.DataFrame: A DataFrame with %K and %D values.
    """
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()

    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)

    # %D = SMA of %K
    stoch_d = stoch_k.rolling(window=signal_window).mean()

    return pd.DataFrame({'%K': stoch_k, '%D': stoch_d})

# Calculate Volume-Price Trend (VPT)
def calculate_vpt(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Price Trend (VPT).

    Args:
        data (pd.DataFrame): Stock data with 'Close' and 'Volume' columns.

    Returns:
        pd.Series: VPT values.
    """
    price_change = data['Close'].pct_change()  # Calculate percentage change
    vpt = (price_change * data['Volume']).cumsum()  # Cumulative sum of VPT
    vpt.iloc[0] = 0  # Set initial value to 0
    return vpt

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