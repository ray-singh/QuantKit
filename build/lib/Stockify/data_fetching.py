import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests

def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given stock ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOG').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing Date, Open, High, Low, Close, Volume data.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}.")
        print(f"Fetched historical data for {ticker}")
        return data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def fetch_company_info(ticker: str) -> dict:
    """
    Fetch company information, such as sector, industry, and company description.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Dictionary containing key company information.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_info = {
            "Name": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
            "Description": info.get("longBusinessSummary", "N/A")
        }
        print(f"Fetched company info for {ticker}")
        return company_info
    except Exception as e:
        print(f"Error fetching company info: {e}")
        return {}


def fetch_live_price(ticker: str) -> float:
    """
    Fetch the real-time price of a given stock ticker.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float: Current stock price.
    """
    try:
        stock = yf.Ticker(ticker)
        live_data = stock.history(period="1d")
        if live_data.empty:
            raise ValueError(f"No real-time price data available for {ticker}.")
        price = live_data['Close'][-1]
        print(f"Fetched live price for {ticker}: ${price:.2f}")
        return price
    except Exception as e:
        print(f"Error fetching live price: {e}")
        return 0.0


def fetch_multiple_tickers_data(tickers: list, start_date: str, end_date: str) -> dict:
    """
    Fetch historical data for multiple stock tickers.

    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        dict: Dictionary with ticker symbols as keys and DataFrames as values.
    """
    results = {}
    for ticker in tickers:
        data = fetch_data(ticker, start_date, end_date)
        results[ticker] = data
    print(f"Fetched historical data for {len(tickers)} tickers.")
    return results

def fetch_financials(ticker: str) -> dict:
    """
    Fetch financial data like income statement, balance sheet, and cash flow using yfinance.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Dictionary containing financial data (income statement, balance sheet, etc.).
    """
    try:
        stock = yf.Ticker(ticker)
        financials = {
            'income_statement': stock.financials,
            'balance_sheet': stock.balance_sheet,
            'cash_flow': stock.cashflow,
        }
        print(f"Fetched financial data for {ticker}")
        return financials
    except Exception as e:
        print(f"Error fetching financials: {e}")
        return {}

def get_stock_values(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch the time series of stock closing prices.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        source (str): Data source ('yfinance' or 'alpha_vantage', default is 'yfinance').

    Returns:
        pd.Series: Time series of stock closing prices with dates as the index.
    """
    # Fetch data using your existing fetch_data function
    stock_data = fetch_data(ticker, start_date, end_date)

    # Check if data is empty or invalid
    if stock_data is None or stock_data.empty:
        raise ValueError("No data fetched for the given ticker and date range.")

    # Return only the closing prices as a Series
    if "Close" in stock_data.columns:
        return stock_data["Close"]
    else:
        raise KeyError("'Close' column not found in the stock data.")


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Calculate stock returns based on closing prices.

    Args:
        prices (pd.Series): Time series of stock closing prices.
        method (str): Method to calculate returns ('simple' or 'log'). Defaults to 'simple'.

    Returns:
        pd.Series: Series of calculated returns with dates as the index.

    Raises:
        ValueError: If the method is invalid or prices are empty.
    """
    if prices.empty:
        raise ValueError("Input price series is empty.")

    if method == "simple":
        returns = prices.pct_change()  # Simple returns
    elif method == "log":
        returns = np.log(prices / prices.shift(1))  # Log returns
    else:
        raise ValueError("Invalid method. Choose 'simple' or 'log'.")

    return returns.dropna()
