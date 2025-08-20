
import os
import pandas as pd
from src.data_ingestion import fetch_stock_data, load_example_company_list
from src.feature_engineering import engineer_features
from src.model_training import train_model, load_model
from src.backtesting import perform_simple_backtest

def main():
    """
    Main function to run the StartQuest Machine Learning pipeline.
    """
    print("Starting StartQuest Machine Learning pipeline...")

    # --- Configuration ---
    RAW_DATA_PATH = 'data/raw/example_companies.csv'
    PROCESSED_DATA_DIR = 'data/processed/'
    PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'features_targets.csv')
    MODEL_PATH = 'startquest_model.pkl'
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01' # Ensure enough historical data for feature engineering (e.g., 30-day future target)

    # Ensure data directories exist
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


    # --- Step 1: Data Ingestion ---
    # Load example company tickers from the CSV file
    tickers = load_example_company_list(RAW_DATA_PATH)
    if not tickers:
        print("No tickers loaded. Please ensure 'data/raw/example_companies.csv' exists and is correctly formatted.")
        return

    # Fetch historical stock data for the loaded tickers
    raw_data_df = fetch_stock_data(tickers, START_DATE, END_DATE)
    if raw_data_df is None or raw_data_df.empty:
        print("Failed to fetch raw stock data. Exiting.")
        return

    # --- Step 2: Feature Engineering ---
    # Engineer financial features and create the target variable based on future price movements
    features_targets_df = engineer_features(raw_data_df)
    if features_targets_df is None or features_targets_df.empty:
        print("Failed to engineer features. Exiting.")
        return

    # Save processed data for potential future use or debugging
    try:
        features_targets_df.to_csv(PROCESSED_DATA_PATH, index=True)
        print(f"Processed features and targets saved to {PROCESSED_DATA_PATH}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

    # --- Step 3: Model Training ---
    # Train the machine learning model using the engineered features and target
    trained_model = train_model(features_targets_df, MODEL_PATH)
    if trained_model is None:
        print("Model training failed. Exiting.")
        return

    # --- Step 4: Simple Backtesting ---
    # Evaluate the trained model's performance on the historical data using a simplified backtest.
    # IMPORTANT: In a production system, backtesting should always be done on data
    # that the model has *never* seen during training to avoid look-ahead bias.
    # For this demonstration, we are using the same dataset for simplicity.
    backtest_results = perform_simple_backtest(trained_model, features_targets_df)
    if backtest_results is None:
        print("Backtesting failed.")

    print("\nStartQuest Machine Learning pipeline completed.")

if __name__ == "__main__":
    main()
