from QuantKit.Portfolio import *

#--------------------Example Use Cases-----------------------------#

# Assuming we have a portfolio instance with a list of stock symbols
portfolio = Portfolio(stock_symbols=['AAPL', 'TSLA', 'GOOGL', 'AMZN'])

# Specify the P/E ratio and dividend yield thresholds
pe_ratio_threshold = 20.0  # Maximum acceptable P/E ratio
div_yield_threshold = 3.0  # Minimum acceptable dividend yield (%)

# Call the function to get the underperforming stocks
underperforming_stocks = recommend_stocks_to_sell(portfolio, pe_ratio_threshold, div_yield_threshold)

# Output the results
print("Stocks to consider selling:")
for stock in underperforming_stocks:
    print(stock)

# Call the function to optimize portfolio based on the Sharpe ratio
optimized_weights = optimize_portfolio(portfolio, method='sharpe')

# Output the optimized weights for each stock
print("Optimized Portfolio Weights (Sharpe Ratio):")
for stock, weight in optimized_weights.items():
    print(f"{stock}: {weight:.2f}")

# New portfolio instance
portfolio = Portfolio(stock_symbols=['AAPL', 'TSLA', 'GOOGL', 'AMZN'])

# Call the function to calculate the annualized return
annualized_return = calculate_annualized_return(portfolio)

# Output the result
print(f"Annualized Return of the Portfolio: {annualized_return}%")

# Call the function to calculate the Sharpe Ratio
sharpe_ratio_value = sharpe_ratio(portfolio, risk_free_rate=0.01)

# Output the result
print(f"Sharpe Ratio of the Portfolio: {sharpe_ratio_value}")

# Call the function to calculate the Sortino Ratio
sortino_ratio_value = sortino_ratio(portfolio, risk_free_rate=0.01)

# Output the result
print(f"Sortino Ratio of the Portfolio: {sortino_ratio_value}")