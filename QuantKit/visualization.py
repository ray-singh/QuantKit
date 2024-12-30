import plotly.graph_objs as go
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import yfinance as yf
from indicators import calculate_sma, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_rsi


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
def plot_macd(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate MACD, Signal, and Histogram
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)

    fig = go.Figure()

    # Plot MACD
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MACD'],
        mode='lines',
        name=f'{ticker} MACD',
        line=dict(color='blue')
    ))

    # Plot Signal Line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Signal'],
        mode='lines',
        name=f'{ticker} Signal Line',
        line=dict(color='red')
    ))

    # Plot Histogram
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Histogram'],
        name=f'{ticker} Histogram',
        marker=dict(color='gray'),
        opacity=0.5
    ))

    fig.update_layout(
        title=f'{ticker} MACD Analysis',
        xaxis_title='Date',
        yaxis_title='MACD Value',
        template='plotly_dark'
    )
    fig.show()


# Function to plot Bollinger Bands with stock price
def plot_bollinger_bands(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate Bollinger Bands
    data['UpperBand'], data['LowerBand'] = calculate_bollinger_bands(data)

    fig = go.Figure()

    # Plot stock price
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f'{ticker} Candlesticks'
    ))

    # Plot Upper Bollinger Band
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['UpperBand'],
        mode='lines',
        name=f'{ticker} Upper Bollinger Band',
        line=dict(color='green')
    ))

    # Plot Lower Bollinger Band
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['LowerBand'],
        mode='lines',
        name=f'{ticker} Lower Bollinger Band',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'{ticker} Bollinger Bands',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    fig.show()


# Function to plot RSI with overbought and oversold zones
def plot_rsi(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate RSI
    data['RSI'] = calculate_rsi(data)

    fig = go.Figure()

    # Plot RSI
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        mode='lines',
        name=f'{ticker} RSI',
        line=dict(color='blue')
    ))

    # Plot Overbought zone
    fig.add_trace(go.Scatter(
        x=data.index,
        y=[70] * len(data),
        mode='lines',
        name='Overbought (70)',
        line=dict(color='red', dash='dash')
    ))

    # Plot Oversold zone
    fig.add_trace(go.Scatter(
        x=data.index,
        y=[30] * len(data),
        mode='lines',
        name='Oversold (30)',
        line=dict(color='green', dash='dash')
    ))

    fig.update_layout(
        title=f'{ticker} RSI',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        template='plotly_dark'
    )
    fig.show()


# Plot Volume-Price Trend (VPT)
def plot_vpt(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate VPT
    data['VPT'] = calculate_vpt(data)

    fig = go.Figure()

    # Plot stock price
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f'{ticker} Candlesticks'
    ))

    # Plot VPT
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['VPT'],
        mode='lines',
        name=f'{ticker} VPT',
        line=dict(color='purple')
    ))

    fig.update_layout(
        title=f'{ticker} Volume-Price Trend (VPT)',
        xaxis_title='Date',
        yaxis_title='Price / VPT',
        template='plotly_dark'
    )
    fig.show()
