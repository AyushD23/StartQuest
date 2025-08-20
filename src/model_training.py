import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import joblib
import os

def train_model(features_df, model_path="startquest_model.pkl"):
    """
    Trains a RandomForestClassifier model and saves it.

    Args:
        features_df (pandas.DataFrame): DataFrame containing features and 'target' column.
        model_path (str): Path to save the trained model.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained model.
    """
    if features_df is None or features_df.empty:
        print("Error: No data available for model training.")
        return None

    # Separate features (X) and target (y)
    # Ensure 'ticker' column is dropped from features as it's an identifier, not a feature
    X = features_df.drop(columns=['target', 'ticker'], errors='ignore')
    y = features_df['target']

    # Handle cases where all targets are the same (no variability)
    if len(y.unique()) < 2:
        print("Warning: Target variable has only one class. Cannot train a classifier effectively.")
        print("This might happen if your data window is too small or the target definition is too narrow.")
        return None

    # Split data into training and testing sets
    # Using stratify to ensure target distribution is similar in train/test sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Error splitting data with stratification: {e}. This usually means one class has too few samples for stratification.")
        print("Attempting split without stratification (use with caution, especially for imbalanced data).")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Target distribution in training:\n{y_train.value_counts(normalize=True)}")
    print(f"Target distribution in test:\n{y_test.value_counts(normalize=True)}")

    # Initialize and train the RandomForestClassifier
    # n_estimators: number of trees in the forest.
    # class_weight='balanced': handles class imbalance by adjusting weights inversely proportional to class frequencies.
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    print("Training Random Forest Classifier...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of positive class

    print("\n--- Model Evaluation ---")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ROC AUC Score is useful for imbalanced datasets and evaluating probability predictions
    if len(y_test.unique()) == 2: # Check if both classes are present in y_test
        print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    else:
        print("ROC-AUC Score: Not applicable (only one class in test set).")


    # Save the trained model
    try:
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    return model

def load_model(model_path="startquest_model.pkl"):
    """
    Loads a pre-trained model.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The loaded model, or None if not found.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
