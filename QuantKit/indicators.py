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
def calculate_macd(data: pd.DataFrame, short_window: int = 12, long_window: int = 26,
                   signal_window: int = 9) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD) and its Signal line.

    Args:
        data (pd.DataFrame): Stock data with a 'Close' column.
        short_window (int): The period for the short-term EMA.
        long_window (int): The period for the long-term EMA.
        signal_window (int): The period for the Signal line (EMA of MACD).

    Returns:
        pd.DataFrame: A DataFrame containing MACD and Signal line.
    """
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)

    macd = short_ema - long_ema
    signal_line = calculate_ema(pd.DataFrame(macd), signal_window)

    return pd.DataFrame({'MACD': macd, 'Signal Line': signal_line})


# Calculate Bollinger Bands
def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std_dev: int = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        data (pd.DataFrame): Stock data with a 'Close' column.
        window (int): The period over which to calculate the moving average.
        num_std_dev (int): The number of standard deviations for the upper and lower bands.

    Returns:
        pd.DataFrame: A DataFrame containing the Upper, Middle (SMA), and Lower bands.
    """
    sma = calculate_sma(data, window)
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)

    return pd.DataFrame({'Upper Band': upper_band, 'Middle Band (SMA)': sma, 'Lower Band': lower_band})


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
def calculate_vpt(data: pd.DataFrame):
    vpt = [0]  # Initialize VPT with the first data point
    for i in range(1, len(data)):
        price_change = (data['Close'][i] - data['Close'][i-1]) / data['Close'][i-1]
        volume_change = data['Volume'][i]  # Use the volume for the day
        vpt.append(vpt[-1] + price_change * volume_change)
    return vpt

if __name__ == "__main__":
    # Example Usage with Sample Data
    import yfinance as yf

    # Fetch historical data for a stock (AAPL)
    stock = yf.Ticker("AAPL")
    data = stock.history(period="1y")

    # Calculate Technical Indicators
    print("\n----- Simple Moving Average (SMA) -----")
    sma = calculate_sma(data, window=20)
    print(sma.tail())

    print("\n----- Exponential Moving Average (EMA) -----")
    ema = calculate_ema(data, window=20)
    print(ema.tail())

    print("\n----- Relative Strength Index (RSI) -----")
    rsi = calculate_rsi(data, window=14)
    print(rsi.tail())

    print("\n----- Moving Average Convergence Divergence (MACD) -----")
    macd = calculate_macd(data)
    print(macd.tail())

    print("\n----- Bollinger Bands -----")
    bollinger_bands = calculate_bollinger_bands(data)
    print(bollinger_bands.tail())

    print("\n----- On-Balance Volume (OBV) -----")
    obv = calculate_obv(data)
    print(obv.tail())

    print("\n----- Stochastic Oscillator -----")
    stochastic_oscillator = calculate_stochastic_oscillator(data)
    print(stochastic_oscillator.tail())