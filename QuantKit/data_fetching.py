import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime
import requests

# Load environment variables from .env file
load_dotenv()

# Fetch API key from environment
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("API key not found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")

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
    Fetch financial data like income statement, balance sheet, and cash flow from Alpha Vantage API.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Dictionary containing financial data (income statement, balance sheet, etc.).
    """
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'INCOME_STATEMENT',
        'symbol': ticker,
        'apikey': ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'error' in data:
            raise ValueError(f"Error fetching financial data for {ticker}: {data['error']}")
        print(f"Fetched financial data for {ticker}")
        return data
    except Exception as e:
        print(f"Error fetching financials: {e}")
        return {}

def get_stock_values(ticker: str, start_date: str, end_date: str, source: str = "yfinance") -> pd.Series:
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

    # Return only the closing prices as a Series
    if "Close" in stock_data.columns:
        return stock_data["Close"]
    else:
        raise KeyError("'Close' column not found in the stock data.")


def get_daily_returns(stock_values: pd.Series) -> pd.Series:
    """
    Calculate the daily returns from a series of stock values.

    Args:
        stock_values (pd.Series): Time series of stock prices with dates as the index.

    Returns:
        pd.Series: Time series of daily returns.
    """
    if not isinstance(stock_values, pd.Series):
        raise ValueError("Input 'stock_values' must be a pandas Series.")

    # Calculate daily returns using percentage change
    daily_returns = stock_values.pct_change().dropna()

    return daily_returns

if __name__ == "__main__":
    # Example usage
    print("----- Fetch Historical Data -----")
    historical_data = fetch_data("AAPL", "2024-01-01", "2024-06-01")
    print(historical_data.head())

    print("\n----- Fetch Company Info -----")
    company_info = fetch_company_info("AAPL")
    print(company_info)

    print("\n----- Fetch Live Price -----")
    live_price = fetch_live_price("AAPL")
    print(f"Live Price: ${live_price:.2f}")

    print("\n----- Fetch Multiple Tickers Data -----")
    multiple_data = fetch_multiple_tickers_data(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")
    for ticker, data in multiple_data.items():
        print(f"{ticker} Data:")
        print(data.head())