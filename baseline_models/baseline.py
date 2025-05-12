import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CONFIG = {
    "data_path": './forta_hacked_address_features.csv',
    "test_size": 0.3,
    "random_state": 42,
    "target_column": 'FLAG',
    "address_column": 'Address',
    "received_ether_col": 'Total Ether Received',
    "received_tx_col": 'Received Transactions',
    "avg_val_received_col": 'avg val received',
    "thresholds_to_test": np.arange(0.1, 1.1, 0.1),
    # In a real scenario, the optimal_threshold would be determined from evaluation
    # and potentially stored in a config file or database.
    "optimal_threshold": 0.5 # Example threshold - this should be determined during training/evaluation
}

# --- Data Loading ---
def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        logging.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

# --- Feature Engineering ---
def create_avg_val_received(data, received_ether_col, received_tx_col, avg_val_received_col):
    """Creates the 'avg val received' column."""
    if data is None:
        return None

    logging.info(f"Creating '{avg_val_received_col}' column.")
    # Ensure columns exist
    if received_ether_col not in data.columns or received_tx_col not in data.columns:
        logging.error(f"Required columns ('{received_ether_col}', '{received_tx_col}') not found for feature engineering.")
        return None

    data[avg_val_received_col] = data.apply(
        lambda row: row[received_ether_col] / row[received_tx_col] if row[received_tx_col] > 0 else 0,
        axis=1
    )
    logging.info(f"'{avg_val_received_col}' column created.")
    return data

# --- Data Balancing (SMOTE) ---
def apply_smote(X, y, random_state):
    """Applies SMOTE to balance the dataset."""
    if X is None or y is None:
        return None, None

    logging.info("Applying SMOTE to balance the dataset.")
    try:
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        logging.info("SMOTE applied successfully. Dataset balanced.")
        logging.info(f"Original class distribution: {y.value_counts().to_dict()}")
        logging.info(f"Balanced class distribution: {y_balanced.value_counts().to_dict()}")
        return X_balanced, y_balanced
    except Exception as e:
        logging.error(f"Error applying SMOTE: {e}")
        return None, None

# --- Model Evaluation (Threshold-based) ---
def evaluate_threshold_model(X_test, y_test, thresholds, feature_col):
    """Evaluates the threshold-based model for a range of thresholds."""
    if X_test is None or y_test is None or feature_col not in X_test.columns:
        logging.error("Missing data or feature column for evaluation.")
        return

    logging.info(f"Evaluating model with threshold on '{feature_col}'.")

    evaluation_results = {}

    for threshold in thresholds:
        logging.info(f"Evaluating with threshold: {threshold:.1f}")

        # Predicting based on threshold
        y_pred = (X_test[feature_col] > threshold).astype(int)

        # Confusion Matrix and Classification Report
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, zero_division=1)

        logging.info('Confusion Matrix:')
        logging.info(conf_matrix)
        logging.info('\nClassification Report:')
        logging.info(class_report)
        logging.info('-' * 50)

        evaluation_results[threshold] = {
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
            # You might want to extract specific metrics here, e.g., F1-score for the positive class
        }

    # In a production scenario, you would analyze evaluation_results to select the best threshold
    # based on your specific business metrics (e.g., maximizing recall with a minimum precision).
    # For this example, we just demonstrate the evaluation loop.
    logging.info("Evaluation complete.")
    return evaluation_results

# --- Inference ---
def predict_with_threshold(data, threshold, feature_col, address_col):
    """
    Makes predictions on new, unseen data using a pre-determined threshold
    on the specified feature column.
    """
    if data is None or feature_col not in data.columns or address_col not in data.columns:
        logging.error("Missing data or required columns for prediction.")
        return None

    logging.info(f"Making predictions using threshold {threshold:.2f} on '{feature_col}'.")

    try:
        # Apply the threshold
        predictions = (data[feature_col] > threshold).astype(int)

        # Create a result DataFrame with Address and Predictions
        results_df = pd.DataFrame({
            address_col: data[address_col],
            CONFIG["target_column"]: predictions
        })

        logging.info("Prediction complete.")
        return results_df
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- Training and Evaluation Phase ---
    logging.info("--- Starting Training and Evaluation Phase ---")

    # Load data
    data = load_data(CONFIG["data_path"])

    if data is not None:
        # Feature Engineering
        data = create_avg_val_received(data, CONFIG["received_ether_col"], CONFIG["received_tx_col"], CONFIG["avg_val_received_col"])

        if data is not None:
            # Separate features and labels
            # For SMOTE, we typically exclude identifier columns
            X = data.drop(columns=[CONFIG["address_column"], CONFIG["target_column"]])
            y = data[CONFIG["target_column"]]

            # Apply SMOTE
            X_balanced, y_balanced = apply_smote(X, y, CONFIG["random_state"])

            if X_balanced is not None and y_balanced is not None:
                 # Recreate balanced DataFrame to include the engineered feature for splitting and evaluation
                X_balanced_df = pd.DataFrame(X_balanced, columns=X.columns)

                # Splitting the balanced data for training and testing the threshold rule
                # Note: In a real scenario, you might split BEFORE SMOTE to avoid data leakage
                # and apply SMOTE only on the training fold. For this baseline revision,
                # we follow the original structure but acknowledge this limitation.
                # A more robust approach would train a proper classifier after SMOTE on the training split.
                _, X_test_balanced, _, y_test_balanced = train_test_split(
                    X_balanced_df, y_balanced, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y_balanced
                )

                # Evaluate the threshold model on the balanced test set
                # Note: Evaluating on balanced test set might give an optimistic view.
                # It's often better to evaluate on an unbalanced test set if that's
                # representative of production data.
                evaluation_results = evaluate_threshold_model(
                    X_test_balanced, y_test_balanced, CONFIG["thresholds_to_test"], CONFIG["avg_val_received_col"]
                )

                # --- Decision Point: Select Optimal Threshold ---
                # Analyze evaluation_results here to pick the best threshold
                # based on production requirements (e.g., desired precision/recall trade-off).
                # For this example, we'll just use the threshold defined in CONFIG.
                logging.info(f"\n--- Selected Optimal Threshold: {CONFIG['optimal_threshold']:.2f} (Example) ---")


    # --- Inference Phase (Example) ---
    # This part simulates making predictions on new, unseen data in production.
    logging.info("\n--- Starting Inference Phase (Example) ---")

    # Simulate loading new data (e.g., a new batch of addresses)
    # In a real system, this data would come from a live feed, database, etc.
    # For this example, we'll sample some data from the original file BEFORE balancing.
    logging.info(f"Simulating loading new data from {CONFIG['data_path']}")
    new_data_raw = load_data(CONFIG["data_path"]) # Load original data again

    if new_data_raw is not None:
        # Perform feature engineering on the new data
        new_data_processed = create_avg_val_received(
            new_data_raw.copy(), # Use a copy to avoid modifying the original loaded data
            CONFIG["received_ether_col"],
            CONFIG["received_tx_col"],
            CONFIG["avg_val_received_col"]
        )

        if new_data_processed is not None:
            # Make predictions using the pre-determined optimal threshold
            predictions_df = predict_with_threshold(
                new_data_processed,
                CONFIG["optimal_threshold"],
                CONFIG["avg_val_received_col"],
                CONFIG["address_column"]
            )

            if predictions_df is not None:
                logging.info("\nExample Predictions on New Data:")
                logging.info(predictions_df.head()) # Display first few predictions