import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculates the Sharpe Ratio.

    Args:
        returns (pandas.Series): Daily returns of an asset.
        risk_free_rate (float): Annual risk-free rate (default 0.0 for simplicity).

    Returns:
        float: Sharpe Ratio.
    """
    if returns.empty or returns.std() == 0:
        return 0.0 # Avoid division by zero
    excess_returns = returns - (risk_free_rate / 252) # Daily risk-free rate for 252 trading days
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) # Annualize

def calculate_sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
    """
    Calculates the Sortino Ratio.

    Args:
        returns (pandas.Series): Daily returns of an asset.
        risk_free_rate (float): Annual risk-free rate (default 0.0 for simplicity).
        target_return (float): The minimum acceptable return (MAR).

    Returns:
        float: Sortino Ratio.
    """
    if returns.empty:
        return 0.0

    downside_returns = returns[returns < target_return]
    if downside_returns.empty:
        # If no downside returns, downside deviation is 0, so Sortino is effectively infinite.
        # Return a large number to indicate good performance with no downside risk.
        return 1e9

    downside_deviation = np.std(downside_returns)
    if downside_deviation == 0:
        return 1e9

    excess_returns = returns - (risk_free_rate / 252)
    return np.mean(excess_returns) / downside_deviation * np.sqrt(252)

def calculate_rsi(series, window=14):
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        series (pandas.Series): Price series (e.g., 'Close' prices).
        window (int): The lookback period for RSI calculation.

    Returns:
        pandas.Series: RSI values.
    """
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)

    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use EWM (Exponentially Weighted Moving Average) for typical RSI calculation
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volatility(series, window=20):
    """
    Calculates a simple rolling volatility (standard deviation of returns).

    Args:
        series (pandas.Series): Price series.
        window (int): The lookback period for volatility calculation.

    Returns:
        pandas.Series: Annualized volatility values.
    """
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    returns = series.pct_change()
    return returns.rolling(window=window).std() * np.sqrt(252) # Annualized volatility

def engineer_features(data_df):
    """
    Engineers financial features from raw historical stock data.

    Args:
        data_df (pandas.DataFrame): MultiIndex DataFrame from yfinance download,
                                   with 'Adj Close' and 'Volume' levels.

    Returns:
        pandas.DataFrame: A DataFrame with engineered features and target variable.
                          Returns None if input data is not sufficient.
    """
    if data_df is None or data_df.empty:
        print("Error: Input data for feature engineering is empty or None.")
        return None

    engineered_features_list = []

    # Identify tickers in the DataFrame
    if isinstance(data_df.columns, pd.MultiIndex):
        tickers = data_df.columns.get_level_values(1).unique()
        # Filter out non-numeric columns if present after yfinance download
        data_df = data_df.select_dtypes(include=np.number)
    else: # Assume single ticker data if not MultiIndex
        tickers = ['SINGLE_TICKER'] # Dummy ticker name for single stock data
        # Ensure column names match expected patterns for single ticker (e.g., 'Adj Close')
        if 'Adj Close' not in data_df.columns or 'Volume' not in data_df.columns:
            print("Error: Single ticker data must contain 'Adj Close' and 'Volume' columns.")
            return None
        # Convert to MultiIndex temporarily to unify processing
        data_df = pd.concat({tickers[0]: data_df}, axis=1, names=['Ticker_Level'])


    for ticker in tickers:
        try:
            # Extract 'Adj Close' and 'Volume' for the current ticker
            close_prices = data_df['Adj Close'][ticker]
            volume = data_df['Volume'][ticker]

            # Calculate daily returns
            returns = close_prices.pct_change().dropna()

            # Calculate features. Use min_periods to ensure enough data for initial calculations.
            # For Sharpe/Sortino, using a shorter rolling window (e.g., 60 days) might be more practical
            # than annual (252) if daily predictions are desired. Adjust as needed.
            sharpe = returns.rolling(window=60, min_periods=20).apply(calculate_sharpe_ratio, raw=False).rename('sharpe_ratio')
            sortino = returns.rolling(window=60, min_periods=20).apply(calculate_sortino_ratio, raw=False).rename('sortino_ratio')
            rsi = calculate_rsi(close_prices, window=14).rename('rsi')
            volatility = calculate_volatility(close_prices, window=20).rename('volatility')

            # Lagged returns as an example of time series features
            lag_returns_1d = returns.shift(1).rename('lag_returns_1d')
            lag_returns_5d = returns.shift(5).rename('lag_returns_5d')

            # Combine all features for the current ticker
            ticker_features = pd.concat([
                sharpe, sortino, rsi, volatility,
                lag_returns_1d, lag_returns_5d, volume.rename('volume')
            ], axis=1)

            ticker_features['ticker'] = ticker
            engineered_features_list.append(ticker_features)

        except KeyError as e:
            print(f"KeyError processing {ticker}: {e}. Skipping this ticker. Make sure 'Adj Close' and 'Volume' columns exist.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for ticker {ticker}: {e}. Skipping.")
            continue

    if not engineered_features_list:
        print("No features could be engineered for any ticker. Check input data and date range.")
        return None

    all_features_df = pd.concat(engineered_features_list).dropna()

    # --- Define a simple binary target variable ---
    # Target: 1 if the stock's adjusted close price increases by more than 2% in the next 30 trading days, 0 otherwise.
    # This is a simplified example; a real investment target would be more nuanced.

    # Ensure to use the original `data_df` for calculating future prices to maintain alignment
    # and avoid issues if some features were dropped.
    if isinstance(data_df.columns, pd.MultiIndex):
        # Flatten the Adj Close for calculation, then re-align
        adj_close_flat = data_df['Adj Close'].stack()
        future_prices_flat = adj_close_flat.groupby(level=1).shift(-30) # Shift back 30 days
        current_prices_flat = adj_close_flat
    else:
        # If original data was single ticker, use direct columns
        current_prices_flat = data_df['Adj Close']
        future_prices_flat = data_df['Adj Close'].shift(-30)

    # Calculate future returns
    # Create a temporary DataFrame to handle potential misalignment after dropping NAs in feature engineering
    temp_df_for_target = pd.DataFrame({
        'current_price': current_prices_flat,
        'future_price': future_prices_flat
    }).dropna()

    future_returns = (temp_df_for_target['future_price'] / temp_df_for_target['current_price']) - 1

    # Map the future returns back to the index of our `all_features_df`
    # Use reindex to align and fill NaN where no corresponding future return exists (e.g., end of data)
    all_features_df['target'] = future_returns.reindex(all_features_df.index)

    # Simple threshold for target: 1 if future return > 0.02 (2%), 0 otherwise
    # Set to NaN if original target was NaN, then drop NaNs
    all_features_df['target'] = all_features_df['target'].apply(lambda x: 1 if x > 0.02 else (0 if not pd.isna(x) else np.nan))
    all_features_df = all_features_df.dropna(subset=['target'])
    all_features_df['target'] = all_features_df['target'].astype(int) # Ensure target is integer type

    print(f"Engineered features for {len(tickers)} tickers. Resulting DataFrame shape: {all_features_df.shape}")
    print("\nSample of engineered features and target:")
    print(all_features_df.head())
    print("\nTarget distribution:")
    print(all_features_df['target'].value_counts(normalize=True))

    return all_features_df
