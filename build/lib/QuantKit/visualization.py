import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
from indicators import calculate_sma, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_rsi, calculate_vpt
from data_fetching import fetch_data

# Auto-detect backend or default to Agg (headless)
try:
    import IPython
    matplotlib.use('module://ipykernel.pylab.backend_inline')  # For Jupyter Notebooks
except ImportError:
    matplotlib.use('Agg')  # Headless or script environments


# Function to plot historical stock price with SMA and EMA overlays
def plot_stock_price(symbol: str,
                     start_date: str,
                     end_date: str,
                     sma_window: int = 50,
                     ema_window: int = 200,
                     title_font_size: int = 16,
                     title_font_color: str = "black",
                     label_font_size: int = 12,
                     label_font_color: str = "gray",
                     grid: bool = True,
                     price_color: str = "black",
                     sma_color: str = "blue",
                     ema_color: str = "red"):

    """
        Plots the stock price along with Simple Moving Average (SMA) and Exponential Moving Average (EMA).

        Parameters:
            symbol (str): Stock ticker symbol.
            start_date (str): Start date for fetching stock data (format: 'YYYY-MM-DD').
            end_date (str): End date for fetching stock data (format: 'YYYY-MM-DD').
            sma_window (int): Window size for calculating SMA. Default is 50.
            ema_window (int): Window size for calculating EMA. Default is 200.
            title_font_size (int): Font size for the plot title. Default is 16.
            title_font_color (str): Color of the plot title text. Default is 'black'.
            label_font_size (int): Font size for axis labels. Default is 12.
            label_font_color (str): Color of axis labels. Default is 'gray'.
            grid (bool): Whether to display grid lines. Default is True.
            price_color (str): Line color for stock price. Default is 'black'.
            sma_color (str): Line color for SMA. Default is 'blue'.
            ema_color (str): Line color for EMA. Default is 'red'.

        Returns:
            None
        """

    # Fetch stock data
    data = fetch_data(symbol, start_date, end_date)

    # Calculate indicators
    sma = calculate_sma(data, sma_window)
    ema = calculate_ema(data, ema_window)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot the stock price as a line chart
    plt.plot(data['Close'], label='Stock Price', color=price_color, linewidth=1.2)

    # Plot SMA and EMA overlays
    plt.plot(sma, label=f'SMA {sma_window}', color=sma_color, linewidth=1.5, linestyle='--')
    plt.plot(ema, label=f'EMA {ema_window}', color=ema_color, linewidth=1.5, linestyle='-.')

    # Title and labels
    plt.title(f'{symbol} Stock Price with SMA and EMA', fontsize=title_font_size, color=title_font_color)
    plt.xlabel('Date', fontsize=label_font_size, color=label_font_color)
    plt.ylabel('Price', fontsize=label_font_size, color=label_font_color)

    # Customize tick labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Add legend with a custom location
    plt.legend(loc='upper left', fontsize=12, shadow=True)

    # Include grid if 'grid' is set to True
    if grid:
        plt.grid()

    plt.tight_layout()  # Ensure the layout does not get cut off
    plt.show()


# Function to plot MACD, Signal Line, and Histogram
def plot_macd(symbol: str,
              start_date: str,
              end_date: str,
              title_font_size: int = 16,
              title_font_color: str = "black",
              label_font_size: int = 12,
              label_font_color: str = "gray",
              grid: bool = True,
              macd_color: str = "green",
              signal_color: str = "red",
              histogram_color: str = "blue"):
    """
    Plots the MACD (Moving Average Convergence Divergence) along with Signal Line and Histogram.

    Parameters:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date for fetching stock data (format: 'YYYY-MM-DD').
        end_date (str): End date for fetching stock data (format: 'YYYY-MM-DD').
        title_font_size (int): Font size for the plot title. Default is 16.
        title_font_color (str): Color of the plot title text. Default is 'black'.
        label_font_size (int): Font size for axis labels. Default is 12.
        label_font_color (str): Color of axis labels. Default is 'gray'.
        grid (bool): Whether to display grid lines. Default is True.
        macd_color (str): Line color for MACD. Default is 'green'.
        signal_color (str): Line color for Signal Line. Default is 'red'.
        histogram_color (str): Color for MACD Histogram. Default is 'blue'.

    Returns:
        None
     """

    # Fetch stock data using fetch_data
    data = fetch_data(symbol, start_date, end_date)

    # Calculate MACD
    macd_data = calculate_macd(data, ticker=symbol)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot MACD and Signal Line
    plt.plot(macd_data.index, macd_data['MACD'], label='MACD', color=macd_color, linewidth=1.5)
    plt.plot(macd_data.index, macd_data['Signal Line'], label='Signal Line', color=signal_color, linewidth=1.5)

    # Plot MACD Histogram
    plt.bar(macd_data.index, macd_data['MACD'] - macd_data['Signal Line'], label='MACD Histogram',
            color=histogram_color, alpha=0.5)

    # Title and labels
    plt.title(f'{symbol} MACD Analysis', fontsize=title_font_size, color=title_font_color)
    plt.xlabel('Date', fontsize=label_font_size, color=label_font_color)
    plt.ylabel('Value', fontsize=label_font_size, color=label_font_color)

    # Customize tick labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid if 'grid' is set to True
    if grid:
        plt.grid(True)

    # Add legend with a custom location
    plt.legend(loc='upper left', fontsize=12, shadow=True)

    plt.tight_layout()  # Ensure the layout does not get cut off
    plt.show()


# Function to plot Bollinger Bands with stock price
def plot_bollinger_bands(symbol: str,
                         start_date: str,
                         end_date: str,
                         window: int = 20,
                         num_std_dev: int = 2,
                         title_font_size: int = 16,
                         title_font_color: str = "black",
                         label_font_size: int = 12,
                         label_font_color: str = "gray",
                         grid: bool = True,
                         price_color: str = "black",
                         upper_band_color: str = "green",
                         lower_band_color: str = "red"):
    """
    Plots the Bollinger Bands for the given stock symbol within a specified date range.

    Args:
        symbol (str): Ticker symbol of the stock.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
        window (int, optional): Window size for the moving average. Default is 20.
        num_std_dev (int, optional): Number of standard deviations for the bands. Default is 2.
        title_font_size (int, optional): Font size for the plot title. Default is 16.
        title_font_color (str, optional): Font color for the plot title. Default is "black".
        label_font_size (int, optional): Font size for axis labels. Default is 12.
        label_font_color (str, optional): Font color for axis labels. Default is "gray".
        grid (bool, optional): Whether to display grid lines. Default is True.
        price_color (str, optional): Color of the stock price line. Default is "black".
        upper_band_color (str, optional): Color of the upper band line. Default is "green".
        lower_band_color (str, optional): Color of the lower band line. Default is "red".

    Returns:
        None
    """

    # Fetch stock data
    data = fetch_data(symbol, start_date, end_date)

    # Calculate Bollinger Bands
    bands = calculate_bollinger_bands(data, ticker=symbol, window=window, num_std_dev=num_std_dev)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot stock price
    plt.plot(data['Close'], label='Stock Price', color=price_color, linewidth=1.5)

    # Plot Bollinger Bands
    plt.plot(bands['Upper Band'], label='Upper Band', color=upper_band_color, linestyle='--', linewidth=1.5)
    plt.plot(bands['Middle Band (SMA)'], label='Middle Band (SMA)', color='blue', linewidth=1.5)
    plt.plot(bands['Lower Band'], label='Lower Band', color=lower_band_color, linestyle='--', linewidth=1.5)

    # Title and labels
    plt.title(f'{symbol} Stock Price with Bollinger Bands', fontsize=title_font_size, color=title_font_color)
    plt.xlabel('Date', fontsize=label_font_size, color=label_font_color)
    plt.ylabel('Price', fontsize=label_font_size, color=label_font_color)

    # Customize tick labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid if 'grid' is set to True
    if grid:
        plt.grid(True)

    # Add legend with a custom location
    plt.legend(loc='upper left', fontsize=12, shadow=True)

    plt.tight_layout()  # Ensure the layout does not get cut off
    plt.show()


# Function to plot RSI with overbought and oversold zones
def plot_rsi(symbol: str,
             start_date: str,
             end_date: str,
             window: int = 14,
             title_font_size: int = 16,
             title_font_color: str = "black",
             label_font_size: int = 12,
             label_font_color: str = "gray",
             grid: bool = True,
             rsi_color: str = "purple",
             overbought_color: str = "red",
             oversold_color: str = "green"):
    """
    Plots the Relative Strength Index (RSI) for the given stock symbol within a specified date range.

    Args:
        symbol (str): Ticker symbol of the stock.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
        window (int, optional): Window size for RSI calculation. Default is 14.
        title_font_size (int, optional): Font size for the plot title. Default is 16.
        title_font_color (str, optional): Font color for the plot title. Default is "black".
        label_font_size (int, optional): Font size for axis labels. Default is 12.
        label_font_color (str, optional): Font color for axis labels. Default is "gray".
        grid (bool, optional): Whether to display grid lines. Default is True.
        rsi_color (str, optional): Color of the RSI line. Default is "purple".
        overbought_color (str, optional): Color for overbought level indicator. Default is "red".
        oversold_color (str, optional): Color for oversold level indicator. Default is "green".

    Returns:
        None
    """

    # Fetch stock data
    data = fetch_data(symbol, start_date, end_date)

    # Calculate RSI
    rsi = calculate_rsi(data, window)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot RSI
    plt.plot(data.index, rsi, label='RSI', color=rsi_color, linewidth=1.5)

    # Plot overbought and oversold levels
    plt.axhline(y=70, color=overbought_color, linestyle='--', label='Overbought', linewidth=1.2)
    plt.axhline(y=30, color=oversold_color, linestyle='--', label='Oversold', linewidth=1.2)

    # Title and labels
    plt.title(f'{symbol} Relative Strength Index (RSI)', fontsize=title_font_size, color=title_font_color)
    plt.xlabel('Date', fontsize=label_font_size, color=label_font_color)
    plt.ylabel('RSI', fontsize=label_font_size, color=label_font_color)

    # Customize tick labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid if 'grid' is set to True
    if grid:
        plt.grid(True)

    # Add legend with a custom location
    plt.legend(loc='upper left', fontsize=12, shadow=True)

    plt.tight_layout()  # Ensure the layout does not get cut off
    plt.show()


# Plot Volume-Price Trend (VPT)
def plot_vpt(symbol: str,
             start_date: str,
             end_date: str,
             title_font_size: int = 16,
             title_font_color: str = "black",
             label_font_size: int = 12,
             label_font_color: str = "gray",
             grid: bool = True,
             vpt_color: str = "blue"):
    """
    Plots the Volume-Price Trend (VPT) for the given stock symbol within a specified date range.

    Args:
        symbol (str): Ticker symbol of the stock.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
        title_font_size (int, optional): Font size for the plot title. Default is 16.
        title_font_color (str, optional): Font color for the plot title. Default is "black".
        label_font_size (int, optional): Font size for axis labels. Default is 12.
        label_font_color (str, optional): Font color for axis labels. Default is "gray".
        grid (bool, optional): Whether to display grid lines. Default is True.
        vpt_color (str, optional): Color of the VPT line. Default is "blue".

    Returns:
        None
    """

    # Fetch stock data
    data = fetch_data(symbol, start_date, end_date)

    # Calculate VPT
    vpt = calculate_vpt(data)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot VPT
    plt.plot(data.index, vpt, label='VPT', color=vpt_color, linewidth=1.5)

    # Title and labels
    plt.title(f'{symbol} Volume-Price Trend (VPT)', fontsize=title_font_size, color=title_font_color)
    plt.xlabel('Date', fontsize=label_font_size, color=label_font_color)
    plt.ylabel('VPT', fontsize=label_font_size, color=label_font_color)

    # Customize tick labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid if 'grid' is set to True
    if grid:
        plt.grid(True)

    # Add legend with a custom location
    plt.legend(loc='upper left', fontsize=12, shadow=True)

    plt.tight_layout()
    plt.show()