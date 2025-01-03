# portfolio.py
import pandas as pd
import numpy as np
from Stockify.data_fetching import fetch_data, fetch_company_info, fetch_live_price, calculate_returns
from typing import List, Dict
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Auto-detect backend or default to Agg (headless)
try:
    import IPython
    matplotlib.use('module://ipykernel.pylab.backend_inline')  # For Jupyter Notebooks
except ImportError:
    matplotlib.use('Agg')  # Headless or script environments

class Portfolio:
    def __init__(self, stock_symbols=None, start_date='2020-01-01', end_date='2024-01-01', weights=None):
        """
        Initialize the Portfolio with stock symbols and fetch data from Yahoo Finance.

        Parameters:
        stock_symbols (list): A list of stock symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date for the data (default is '2020-01-01').
        end_date (str): End date for the data (default is '2024-01-01').
        weights (dict or list, optional): A dictionary or list of portfolio weights for each stock.
        """
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.stocks = self.fetch_stocks_data()
        self.weights = weights if weights else {stock: 1 / len(stock_symbols) for stock in stock_symbols}
        self.returns = self.returns()

    def fetch_stocks_data(self):
        """
        Fetch stock price data using the fetch_data function from data_fetching.py.

        Returns:
        dict: A dictionary where keys are stock symbols and values are pandas DataFrames with stock price data.
        """
        stocks_data = {}
        for symbol in self.stock_symbols:
            data = fetch_data(symbol, self.start_date, self.end_date)
            if not data.empty:
                stocks_data[symbol] = data
        return stocks_data

    def returns(self, method):
        """
        Calculate daily returns for each stock in the portfolio.

        Returns:
        pd.DataFrame: DataFrame of daily returns for each stock in the portfolio.
        """
        returns = pd.DataFrame()
        for stock, data in self.stocks.items():
            returns[stock] = calculate_returns(data['Close'], method)
        return returns

    def calculate_portfolio_return(self):
        """
        Calculate the portfolio's return based on weights and stock returns.

        Returns:
        float: The annualized portfolio return.
        """
        weighted_returns = self.returns.dot(list(self.weights.values()))
        portfolio_return = weighted_returns.mean() * 252  # Annualize return
        return portfolio_return

    def calculate_volatility(self):
        """
        Calculate the portfolio's volatility (standard deviation).

        Returns:
        float: The annualized portfolio volatility.
        """
        covariance_matrix = self.returns.cov() * 252  # Annualized covariance
        portfolio_variance = np.dot(list(self.weights.values()), np.dot(covariance_matrix, list(self.weights.values())))
        portfolio_volatility = np.sqrt(portfolio_variance)
        return portfolio_volatility


    def add_stock(self, stock_symbol, weight=None):
        """
        Add a new stock to the portfolio and fetch its data.

        Parameters:
        stock_symbol (str): The stock symbol to add (e.g., 'GOOG').
        weight (float, optional): The weight of the new stock in the portfolio. If None, an equal weight is assigned.
        """
        if stock_symbol not in self.stocks:
            data = fetch_data(stock_symbol, self.start_date, self.end_date)
            if not data.empty:
                self.stocks[stock_symbol] = data
                self.weights[stock_symbol] = weight if weight else 1 / len(self.stocks)
                self.returns = self.returns()

    def remove_stock(self, stock_symbol):
        """
        Remove a stock from the portfolio.

        Parameters:
        stock_symbol (str): The stock symbol to be removed.
        """
        if stock_symbol in self.stocks:
            del self.stocks[stock_symbol]
            del self.weights[stock_symbol]
            self.returns = self.returns()
        else:
            print(f"Stock {stock_symbol} not found in portfolio.")

    def get_company_info(self, stock_symbol):
        """
        Get basic company information such as sector, industry, and description.

        Parameters:
        stock_symbol (str): The stock symbol to get information for.

        Returns:
        dict: Company information dictionary.
        """
        return fetch_company_info(stock_symbol)

    def get_live_price(self, stock_symbol):
        """
        Get the live price of a stock.

        Parameters:
        stock_symbol (str): The stock symbol to get the live price for.

        Returns:
        float: The current stock price.
        """
        return fetch_live_price(stock_symbol)

    def plot_portfolio_performance(self):
        """
        Plot the cumulative returns of the portfolio.
        """
        weighted_returns = self.returns.dot(list(self.weights.values()))
        cumulative_returns = (1 + weighted_returns).cumprod()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, label='Portfolio Cumulative Returns', color='blue')
        plt.title('Portfolio Cumulative Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_stock_allocation(self):
        """
        Plot a pie chart showing stock allocation in the portfolio.
        """
        plt.figure(figsize=(8, 8))
        plt.pie(self.weights.values(), labels=self.weights.keys(), autopct='%1.1f%%', startangle=140,
                colors=sns.color_palette('pastel'))
        plt.title("Portfolio Stock Allocation")
        plt.show()

    def plot_individual_stock_performance(self):
        """
        Plot the cumulative returns of individual stocks in the portfolio.
        """
        plt.figure(figsize=(10, 6))
        for stock, data in self.stocks.items():
            stock_returns = (1 + data['Close'].pct_change()).cumprod()
            plt.plot(stock_returns.index, stock_returns, label=stock)
        plt.title("Individual Stock Performance")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_risk_vs_return(self):
        """
        Plot risk (volatility) vs return for all stocks in the portfolio.
        """
        returns = self.returns.mean() * 252  # Annualized returns
        volatilities = self.returns.std() * np.sqrt(252)  # Annualized volatilities
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=volatilities, y=returns, s=100, color='red')
        for stock in self.stock_symbols:
            plt.text(volatilities[stock], returns[stock], stock, fontsize=10)
        plt.title("Risk vs Return")
        plt.xlabel("Volatility (Risk)")
        plt.ylabel("Annualized Return")
        plt.grid()
        plt.show()

    def compare_with_benchmark(self, benchmark_symbol):
        """
        Compare the portfolio's cumulative returns with a benchmark.

        Parameters:
        benchmark_symbol (str): Stock symbol for the benchmark (e.g., 'SPY').
        """
        benchmark_data = fetch_data(benchmark_symbol, self.start_date, self.end_date)['Close'].pct_change()
        portfolio_weighted_returns = self.returns.dot(list(self.weights.values()))
        portfolio_cumulative = (1 + portfolio_weighted_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_data).cumprod()

        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_cumulative.index, portfolio_cumulative, label='Portfolio', color='blue')
        plt.plot(benchmark_cumulative.index, benchmark_cumulative, label=benchmark_symbol, color='green')
        plt.title("Portfolio vs Benchmark Performance")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid()
        plt.show()

#--------------------------------- Methods for portfolio analyses -----------------------------------------#

def recommend_stocks_to_sell(self, pe_ratio_threshold: float, div_yield_threshold: float) -> List[Dict]:
    """
    Identify underperforming stocks in the portfolio that should be sold off.

    Args:
        pe_ratio_threshold (float): Maximum acceptable P/E ratio.
        div_yield_threshold (float): Minimum acceptable dividend yield.

    Returns:
        List[Dict]: List of dictionaries containing underperforming stock information.
    """
    underperforming_stocks = []

    for ticker in self.stock_symbols:
        info = fetch_company_info(ticker)
        if not info:
            continue  # Skip if unable to fetch info

        pe_ratio = info.get("trailingPE", None)
        div_yield = info.get("dividendYield", None)

        # Normalize dividend yield percentage if available
        div_yield = div_yield * 100 if div_yield is not None else None

        # Check if the stock is underperforming
        if (pe_ratio and pe_ratio > pe_ratio_threshold) or (div_yield and div_yield < div_yield_threshold):
            underperforming_stocks.append({
                "Ticker": ticker,
                "Name": info.get("Name", "N/A"),
                "P/E Ratio": pe_ratio,
                "Dividend Yield (%)": div_yield,
                "Sector": info.get("Sector", "N/A"),
                "Industry": info.get("Industry", "N/A")
            })
    return underperforming_stocks

def optimize_portfolio(data, method='sharpe'):
    """
    Optimize the portfolio using mean-variance optimization.

    Parameters:
    method (str, optional): The optimization method. 'sharpe' for maximizing the Sharpe ratio (default), or 'volatility' for minimizing volatility.

    Returns:
    dict: A dictionary of optimized portfolio weights.
    """
    from scipy.optimize import minimize

    num_assets = len(data.stocks)
    initial_weights = np.ones(num_assets) / num_assets  # Initial equal weights
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling

    def objective(weights):
        weighted_returns = data.returns.dot(weights)
        if method == 'sharpe':
            excess_returns = weighted_returns.mean() - 0.01  # Default risk-free rate
            return -excess_returns / weighted_returns.std()  # Minimize negative Sharpe ratio
        elif method == 'volatility':
            return np.sqrt(np.dot(weights.T, np.dot(data.returns.cov() * 252, weights)))  # Minimize volatility

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x
    return dict(zip(data.stocks.keys(), optimized_weights))


def calculate_annualized_return(portfolio: Portfolio) -> pd.Series:
    """
    Calculate the annualized return of the portfolio.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.

    Returns:
        pd.Series: Annualized return of the portfolio.
    """
    daily_returns = portfolio.returns()
    annualized_return = daily_returns.mean() * 252  # Assume 252 trading days in a year
    return annualized_return


def sharpe_ratio(portfolio: Portfolio, risk_free_rate=0.01) -> float:
    """
    Calculate the Sharpe Ratio for the portfolio, which measures risk-adjusted return.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.
        risk_free_rate (float): Risk-free rate for comparison (default is 1%).

    Returns:
        float: Sharpe Ratio for the portfolio.
    """
    daily_returns = portfolio.returns()
    annualized_return = calculate_annualized_return(portfolio)
    volatility = portfolio.calculate_volatility()

    # Sharpe Ratio formula: (mean return - risk-free rate) / volatility
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    return sharpe_ratio


def sortino_ratio(portfolio: Portfolio, risk_free_rate=0.01) -> float:
    """
    Calculate the Sortino Ratio for the portfolio, which focuses on downside risk.

    Args:
        portfolio (Portfolio): An instance of the Portfolio class.
        risk_free_rate (float): Risk-free rate for comparison (default is 1%).

    Returns:
        float: Sortino Ratio for the portfolio.
    """
    daily_returns = portfolio.returns()
    downside_returns = daily_returns[daily_returns < 0]
    annualized_return = calculate_annualized_return(portfolio)

    # Calculate downside deviation (standard deviation of negative returns)
    downside_deviation = downside_returns.std() * np.sqrt(252)

    # Sortino Ratio formula: (mean return - risk-free rate) / downside deviation
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
    return sortino_ratio

#--------------------------------- Methods for comparative portfolio analyses -----------------------------------------#
def compare_returns(portfolios: List[Portfolio], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Compare cumulative returns of multiple portfolios over a specific time period.

    Args:
        portfolios (List[Portfolio]): List of Portfolio objects.
        start_date (str): Start date for analysis (YYYY-MM-DD).
        end_date (str): End date for analysis (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame with cumulative returns for each portfolio.
    """
    results = {}
    for portfolio in portfolios:
        cumulative_return = 0
        for ticker in portfolio.stock_symbols:
            data = fetch_data(ticker, start_date, end_date)
            if not data.empty:
                daily_returns = data['Close'].pct_change().dropna()
                cumulative_return += daily_returns.add(1).prod() - 1
        results[portfolio] = cumulative_return
    return pd.DataFrame(results.items(), columns=["Portfolio", "Cumulative Return"])


def compare_volatility(portfolios: List[Portfolio], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Compare the volatility (standard deviation of returns) of multiple portfolios.

    Args:
        portfolios (List[Portfolio]): List of Portfolio objects.
        start_date (str): Start date for analysis (YYYY-MM-DD).
        end_date (str): End date for analysis (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame with volatilities for each portfolio.
    """
    results = {}
    for portfolio in portfolios:
        total_volatility = 0
        for ticker in portfolio.stock_symbols:
            data = fetch_data(ticker, start_date, end_date)
            if not data.empty:
                daily_returns = data['Close'].pct_change().dropna()
                total_volatility += daily_returns.std()
        results[portfolio] = total_volatility
    return pd.DataFrame(results.items(), columns=["Portfolio", "Volatility"])


def compare_sharpe_ratios(portfolios: List[Portfolio], start_date: str, end_date: str, risk_free_rate=0.01) -> pd.DataFrame:
    """
    Compare Sharpe ratios of multiple portfolios.

    Args:
        portfolios (List[Portfolio]): List of Portfolio objects.
        start_date (str): Start date for analysis (YYYY-MM-DD).
        end_date (str): End date for analysis (YYYY-MM-DD).
        risk_free_rate (float): Risk-free rate (default 0.01).

    Returns:
        pd.DataFrame: DataFrame with Sharpe ratios for each portfolio.
    """
    results = {}
    for portfolio in portfolios:
        portfolio_return = 0
        portfolio_volatility = 0
        for ticker in portfolio.stock_symbols:
            data = fetch_data(ticker, start_date, end_date)
            if not data.empty:
                daily_returns = data['Close'].pct_change().dropna()
                portfolio_return += daily_returns.mean()
                portfolio_volatility += daily_returns.std()
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility else 0
        results[portfolio] = sharpe_ratio
    return pd.DataFrame(results.items(), columns=["Portfolio", "Sharpe Ratio"])


def compare_compositions(portfolios: List[Portfolio]) -> pd.DataFrame:
    """
    Compare the composition (sector/industry distribution) of multiple portfolios.

    Args:
        portfolios (List[Portfolio]): List of Portfolio objects.

    Returns:
        pd.DataFrame: DataFrame showing sector/industry allocation for each portfolio.
    """
    composition_results = {}
    for portfolio in portfolios:
        sector_counts = {}
        for ticker in portfolio.stock_symbols:
            info = fetch_company_info(ticker)
            if info:
                sector = info.get("Sector", "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        composition_results[portfolio] = sector_counts
    return pd.DataFrame(composition_results).fillna(0)
