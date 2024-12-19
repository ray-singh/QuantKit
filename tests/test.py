import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Switch to the TkAgg backend
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import requests
from datetime import datetime

# Example: Fetch stock data using yfinance
data = yf.download('AAPL', start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'))

# Plot stock closing price using matplotlib
plt.plot(data['Close'])
plt.title('AAPL Stock Closing Price')
plt.show()

# Use Plotly for a better interactive plot
fig = px.line(data, x=data.index, y='Close', title='AAPL Stock Closing Price')
fig.show()
