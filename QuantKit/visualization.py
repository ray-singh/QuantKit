import plotly.graph_objs as go
import matplotlib.pyplot as plt
import yfinance as yf
from indicators import calculate_sma, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_rsi


# Function to plot historical stock price with SMA and EMA overlays
def plot_stock_price(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate SMA and EMA
    data['SMA'] = calculate_sma(data)
    data['EMA'] = calculate_ema(data)

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

    # Plot Simple Moving Average (SMA)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA'],
        mode='lines',
        name=f'{ticker} SMA',
        line=dict(color='blue')
    ))

    # Plot Exponential Moving Average (EMA)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA'],
        mode='lines',
        name=f'{ticker} EMA',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title=f'{ticker} Price with SMA and EMA',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    fig.show()


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
