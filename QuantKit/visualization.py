import plotly.graph_objs as go
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import yfinance as yf
from indicators import calculate_sma, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_rsi, calculate_vpt


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

    # Fetch stock data
    data = yf.download(symbol, start=start_date, end=end_date)

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

    # Fetch stock data
    data = yf.download(symbol, start=start_date, end=end_date)

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
    # Fetch stock data
    data = yf.download(symbol, start=start_date, end=end_date)

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
    # Fetch stock data
    data = yf.download(symbol, start=start_date, end=end_date)

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
    # Fetch stock data
    data = yf.download(symbol, start=start_date, end=end_date)

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

    plt.tight_layout()  # Ensure the layout does not get cut off
    plt.show()