import yfinance as yf
import pandas as pd
from datetime import datetime
import requests

ALPHA_VANTAGE_API_KEY = 'RBO630UAW5YGHXRE'

def sma(data: ,  window=20) -> pd.DataFrame:

