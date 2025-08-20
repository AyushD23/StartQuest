# StartQuest: Machine Learning for Investment Prediction 🚀

## Overview
**StartQuest** is a machine learning project designed to identify and predict promising investment opportunities by analyzing historical financial time series data and deriving key financial metrics. This repository showcases a complete end-to-end pipeline, from data ingestion and sophisticated feature engineering to robust model training using a Random Forest Classifier and a simplified back-testing framework. While this version uses public stock data for demonstration, its architecture is built to be adaptable for private startup investment analysis.

---

## Features ✨
* **Data Ingestion (`src/data_ingestion.py`):**
    * **Automated Data Fetching:** Seamlessly pulls historical stock market data for a list of specified tickers using the powerful `yfinance` library.
    * **Scalable Design:** Structured to integrate with more diverse data sources, including potential web scraping (via `BeautifulSoup` and `requests`) for non-public startup information or proprietary financial databases.

* **Feature Engineering (`src/feature_engineering.py`):**
    * **Financial Metric Generation:** Computes industry-standard and custom financial indicators from raw price and volume data. This includes:
        * **Sharpe Ratio:** Measures risk-adjusted return.
        * **Sortino Ratio:** Focuses on downside risk-adjusted return.
        * **Relative Strength Index (RSI):** A momentum oscillator used in technical analysis.
        * **Volatility:** Annualized standard deviation of returns.
    * **Time-Series Insights:** Incorporates lagged returns to capture short-term price movements and temporal dependencies.

* **Modeling (`src/model_training.py`):**
    * **Robust Classification:** Trains a **Random Forest Classifier** to predict investment potential (e.g., whether an asset will outperform a certain threshold).
    * **Performance Evaluation:** Provides comprehensive evaluation metrics including **Accuracy, Precision, Recall, F1-Score, ROC-AUC**, and a **Confusion Matrix** to assess model effectiveness.
    * **Model Persistence:** Saves the trained model using `joblib` for easy re-use and deployment without retraining.

* **Back-testing (`src/backtesting.py`):**
    * **Simulated Performance:** Evaluates the model's hypothetical performance on historical data, providing insights into its potential profitability.
    * **Strategy Return Calculation:** Calculates cumulative strategy returns based on the model's investment signals, giving a tangible measure of its financial impact.
    * **(Note):** This is a simplified framework; a production-grade system would require more granular portfolio management, transaction costs, and slippage considerations.

---

## Setup and Installation 🛠️
To get this project up and running on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/StartQuest-ML.git](https://github.com/your-username/StartQuest-ML.git)
    cd StartQuest-ML
    ```
    *(Remember to replace `[https://github.com/your-username/StartQuest-ML.git](https://github.com/your-username/StartQuest-ML.git)` with your actual repository URL once hosted.)*

2.  **Create a Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate  # On Windows
    ```
    This isolates project dependencies from your global Python environment.

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This command installs all necessary Python libraries listed in `requirements.txt`.

---

## Usage Guide 🚀
Follow these steps to run the investment prediction pipeline:

1.  **Prepare your Company List:**
    Create or modify the `data/raw/example_companies.csv` file. This CSV should contain a single column named `Ticker` with the stock symbols of the companies you wish to analyze.
    **Example `data/raw/example_companies.csv` content:**
    ```csv
    Ticker
    AAPL
    MSFT
    GOOGL
    AMZN
    TSLA
    ```

2.  **Run the Main Pipeline Script:**
    Execute the central `train.py` script from the project's root directory:
    ```bash
    python train.py
    ```
    This script orchestrates the entire workflow:
    * It fetches historical data for the tickers specified.
    * Engineers all the financial features and computes the target variable.
    * Trains the Random Forest model on the prepared dataset.
    * Performs a basic back-test, printing performance metrics and a summary of the simulated strategy's returns.
    * Saves the trained machine learning model as `startquest_model.pkl` in the project's root directory for future predictions.

    You will see console output detailing each step, including data shapes, model evaluation reports, and back-testing results.

---

## Project Structure 📁
The repository is organized into logical directories to ensure modularity and ease of maintenance:

├── data/                       # Stores all input and output data
│   ├── processed/              # Contains processed data (e.g., features_targets.csv)
│   │   └── features_targets.csv
│   └── raw/                    # Stores raw input data (e.g., list of companies)
│       └── example_companies.csv
├── src/                        # Contains the core Python modules for the ML pipeline
│   ├── init.py             # Makes 'src' a Python package
│   ├── data_ingestion.py       # Handles fetching and loading raw financial data
│   ├── feature_engineering.py  # Implements calculations for financial metrics and target creation
│   ├── model_training.py       # Manages model training, evaluation, and saving
│   └── backtesting.py          # Contains logic for simulating trading strategies
├── .gitignore                  # Specifies files/directories to be ignored by Git
├── README.md                   # This comprehensive project description
├── requirements.txt            # Lists all Python package dependencies
└── train.py                    # The main script that runs the entire ML pipeline



---

## Dependencies 📋
All required Python libraries are listed in `requirements.txt`. Key dependencies include:
* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical operations.
* `scikit-learn`: For machine learning models and utilities.
* `yfinance`: For fetching historical stock data.
* `beautifulsoup4` & `requests`: Included for potential future web scraping capabilities (not directly used in current `yfinance`-based data ingestion).
* `joblib`: For saving and loading Python objects, specifically the trained ML model.

---

## Future Enhancements 📈
This project serves as a strong foundation, and there are many avenues for further development:

* **Advanced Data Integration:**
    * Integrate with actual startup data APIs (e.g., Crunchbase, PitchBook, or proprietary databases) for more realistic startup investment scenarios.
    * Implement robust error handling and retry mechanisms for data fetching.
* **Sophisticated Feature Engineering:**
    * Explore more advanced technical indicators (e.g., MACD, Bollinger Bands).
    * Incorporate fundamental analysis metrics (e.g., P/E ratio, debt-to-equity) if available for startups.
    * Apply natural language processing (NLP) to analyze news sentiment or company descriptions as features.
* **Model Improvement:**
    * Experiment with other machine learning models (e.g., Gradient Boosting Machines like XGBoost/LightGBM, neural networks for time series).
    * Implement hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV.
    * Address class imbalance more rigorously if the "good investment" class is rare.
* **Robust Back-testing Framework:**
    * Develop a more detailed back-testing engine that accounts for transaction costs, slippage, and portfolio diversification.
    * Visualize equity curves and drawdowns to assess strategy risk and return profiles.
* **Deployment and Monitoring:**
    * Build a simple web dashboard (e.g., using Flask/Django or Streamlit) to visualize predictions and model performance.
    * Set up automated retraining and monitoring pipelines to ensure model relevance and accuracy over time.
