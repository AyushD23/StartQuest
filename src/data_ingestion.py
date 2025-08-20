
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import os

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches historical stock data for a list of tickers using yfinance.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date for data in 'YYYY-MM-DD' format.
        end_date (str): End date for data in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame containing historical stock data for all tickers,
                          or None if an error occurs.
    """
    print(f"Fetching stock data for {tickers} from {start_date} to {end_date}...")
    try:
        # Download data for all tickers
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            print(f"Warning: No data fetched for tickers {tickers}. Check ticker symbols or date range.")
            return None
        print("Stock data fetched successfully.")
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def load_example_company_list(filepath):
    """
    Loads a list of example company tickers from a CSV file.
    In a real scenario, this might involve more complex scraping or API calls
    to get actual startup data.

    Args:
        filepath (str): Path to the CSV file containing 'Ticker' column.

    Returns:
        list: A list of ticker symbols.
    """
    print(f"Loading example company list from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}. Please create `data/raw/example_companies.csv`.")
        return []
    try:
        df = pd.read_csv(filepath)
        if 'Ticker' not in df.columns:
            print(f"Error: '{filepath}' must contain a 'Ticker' column.")
            return []
        tickers = df['Ticker'].tolist()
        print(f"Loaded {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"Error loading company list: {e}")
        return []

# Example of a dummy scraping function (not used in current train.py but for context)
def scrape_dummy_startup_info(url="http://example.com"):
    """
    A placeholder function to simulate scraping startup data.
    In a real application, this would parse detailed startup profiles.
    For this example, it just returns dummy data.
    """
    print(f"Attempting to scrape dummy data from {url}...")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        # Simulate finding some data
        dummy_data = {
            "startup_name": "Example Tech Inc.",
            "industry": "Software",
            "funding_round": "Series A",
            "employee_count": "50-100"
        }
        print("Dummy scraping successful.")
        return dummy_data
    except requests.exceptions.RequestException as e:
        print(f"Error during dummy web scraping: {e}")
        return None
