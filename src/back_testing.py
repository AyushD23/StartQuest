import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def perform_simple_backtest(model, features_df):
    """
    Performs a simplified walk-forward back-test on the model.
    This function simulates evaluating the model on unseen data over time.

    Args:
        model: The trained scikit-learn model.
        features_df (pandas.DataFrame): DataFrame containing features and 'target' for backtesting.
                                        Assumes data is chronologically ordered and indexed by date.

    Returns:
        pandas.DataFrame: A DataFrame with daily predictions, actuals, and cumulative strategy returns.
    """
    if model is None or features_df is None or features_df.empty:
        print("Error: Model or features DataFrame is missing for backtesting.")
        return None

    print("\n--- Performing Simple Backtest ---")

    # Ensure data is sorted by date (index) for proper time-series simulation
    # If the index is not datetime, you might need to convert it.
    if not isinstance(features_df.index, pd.DatetimeIndex):
        try:
            features_df.index = pd.to_datetime(features_df.index)
        except Exception:
            print("Warning: DataFrame index is not datetime and cannot be converted. Backtest results might be unreliable.")
    features_df = features_df.sort_index()

    # Separate features and target
    X_test = features_df.drop(columns=['target', 'ticker'], errors='ignore')
    y_test = features_df['target']
    tickers = features_df['ticker'] # Keep tickers for detailed view

    # Predict probabilities for the positive class (investment opportunity)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba (e.g., if model was just `predict`)
        y_prob = model.predict(X_test)
        print("Warning: Model does not have 'predict_proba'. Using direct predictions (0 or 1).")

    # Apply a threshold to convert probabilities to binary predictions
    # A threshold of 0.5 is common, but can be tuned based on desired precision/recall
    y_pred = (y_prob >= 0.5).astype(int)

    # Create a DataFrame to store results
    backtest_results = pd.DataFrame({
        'predicted_opportunity': y_pred,
        'actual_outcome': y_test,
        'probability_of_opportunity': y_prob,
        'ticker': tickers
    }, index=features_df.index)

    # Evaluate overall performance of the backtest period
    print("\nBacktest Period Classification Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (Positive Class):", precision_score(y_test, y_pred, zero_division=0))
    print("Recall (Positive Class):", recall_score(y_test, y_pred, zero_division=0))
    print("F1-Score (Positive Class):", f1_score(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    if len(y_test.unique()) == 2:
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    else:
        print("ROC-AUC: Not applicable (only one class present in backtest target).")


    # --- Simulate a simple investment strategy (for illustrative purposes) ---
    # This is a very basic simulation. A real backtest would require granular
    # daily returns, managing a portfolio, transaction costs, and slippage.

    # Assume: if 'predicted_opportunity' is 1, we "invest".
    # If the 'actual_outcome' for that predicted opportunity was also 1 (meaning it was indeed a good investment),
    # we assign a fixed positive return. Otherwise, 0 return.

    # Example: If we predicted good (1) and it was actually good (1), we get 2% return.
    # Otherwise, if we predicted good (1) but it was not (0), we get 0% return (or a loss in a real scenario).
    # If we predicted not good (0), we get 0% return.

    backtest_results['daily_strategy_return'] = backtest_results.apply(
        lambda row: 0.02 if row['predicted_opportunity'] == 1 and row['actual_outcome'] == 1 else 0.0,
        axis=1
    )

    # Group by date and sum daily returns to get overall portfolio return for each day.
    # This assumes equal weighting for all predicted opportunities on a given day.
    daily_returns_summed = backtest_results.groupby(level=0)['daily_strategy_return'].sum().fillna(0)

    # Calculate cumulative returns. Start with 1 (initial capital) for cumulative product.
    backtest_results['cumulative_strategy_return'] = (1 + daily_returns_summed).cumprod() - 1

    print("\nSample Backtest Results (First 5 rows):")
    print(backtest_results.head())
    print(f"\nTotal Cumulative Strategy Return: {backtest_results['cumulative_strategy_return'].iloc[-1]:.2%}")

    return backtest_results
