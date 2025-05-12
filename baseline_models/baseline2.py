import pandas as pd
import joblib
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CONFIG = {
    "data_path": './forta_hacked_address_features.csv',
    "model_path": './random_forest_model.joblib',
    "scaler_path": './scaler.joblib',
    "test_size": 0.2,
    "random_state": 42,
    "target_column": 'FLAG',
    "ignore_columns": ['Address', 'FLAG']
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

# --- Preprocessing ---
def preprocess_data(data, is_train=True, scaler=None):
    """
    Preprocesses the data by separating features and labels, and scaling features.
    If is_train is True, fits and returns a new scaler.
    If is_train is False, uses the provided scaler to transform data.
    """
    if data is None:
        return None, None, None

    logging.info("Starting data preprocessing.")

    X = data.drop(columns=CONFIG["ignore_columns"])
    y = data[CONFIG["target_column"]]

    if is_train:
        logging.info("Fitting and applying StandardScaler.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info("Preprocessing complete.")
        return X_scaled, y, scaler
    else:
        if scaler is None:
            logging.error("Scaler not provided for testing/inference.")
            return None, y, None
        logging.info("Applying provided StandardScaler.")
        X_scaled = scaler.transform(X)
        logging.info("Preprocessing complete.")
        return X_scaled, y, scaler

# --- Model Training ---
def train_model(X_train, y_train):
    """Trains the RandomForestClassifier model."""
    if X_train is None or y_train is None:
        return None

    logging.info("Starting model training.")

    # Compute class weights to handle imbalance
    classes = y_train.unique()
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    logging.info(f"Computed Class Weights: {class_weight_dict}")

    # Initialize Random Forest Classifier with class weights
    rf_classifier = RandomForestClassifier(random_state=CONFIG["random_state"], class_weight=class_weight_dict)

    # Fit the model
    rf_classifier.fit(X_train, y_train)
    logging.info("Model training complete.")
    return rf_classifier

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model."""
    if model is None or X_test is None or y_test is None:
        return

    logging.info("Starting model evaluation.")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    logging.info(f'Accuracy: {accuracy:.2f}')
    logging.info(f'Precision: {precision:.2f}')
    logging.info(f'Recall: {recall:.2f}')
    logging.info(f'F1 Score: {f1:.2f}')
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred, zero_division=1))
    logging.info("Model evaluation complete.")

# --- Model Persistence ---
def save_model(model, scaler, model_path, scaler_path):
    """Saves the trained model and scaler to disk."""
    if model is None or scaler is None:
        logging.warning("Model or scaler is None, skipping save.")
        return

    try:
        logging.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

    try:
        logging.info(f"Saving scaler to {scaler_path}")
        joblib.dump(scaler, scaler_path)
        logging.info("Scaler saved successfully.")
    except Exception as e:
        logging.error(f"Error saving scaler: {e}")

def load_model(model_path, scaler_path):
    """Loads the trained model and scaler from disk."""
    model = None
    scaler = None
    try:
        logging.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Error: Model file not found at {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

    try:
        logging.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        logging.info("Scaler loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Error: Scaler file not found at {scaler_path}")
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")

    return model, scaler

# --- Prediction/Inference ---
def predict(data, model, scaler):
    """Makes predictions on new data using the loaded model and scaler."""
    if data is None or model is None or scaler is None:
        logging.error("Data, model, or scaler not provided for prediction.")
        return None

    logging.info("Starting prediction.")
    try:
        # Preprocess data using the loaded scaler
        X_scaled, _, _ = preprocess_data(data.copy(), is_train=False, scaler=scaler)

        if X_scaled is None:
            logging.error("Preprocessing failed for prediction data.")
            return None

        predictions = model.predict(X_scaled)
        logging.info("Prediction complete.")
        return predictions
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

# --- Main Execution Flow (Training and Saving) ---
if __name__ == "__main__":
    # Load data
    data = load_data(CONFIG["data_path"])

    if data is not None:
        # Preprocess data and split
        X, y, scaler = preprocess_data(data, is_train=True)

        if X is not None and y is not None and scaler is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y
            )

            # Train model
            model = train_model(X_train, y_train)

            if model is not None:
                # Evaluate model
                evaluate_model(model, X_test, y_test)

                # Save model and scaler
                save_model(model, scaler, CONFIG["model_path"], CONFIG["scaler_path"])

    # --- Example of Loading and Predicting (can be in a separate script/service) ---
    logging.info("-" * 50)
    logging.info("Demonstrating model loading and prediction:")

    loaded_model, loaded_scaler = load_model(CONFIG["model_path"], CONFIG["scaler_path"])

    if loaded_model and loaded_scaler:
        # Create some dummy new data for prediction
        # In a real scenario, this would be new data loaded from a source
        dummy_new_data = data.sample(5).drop(columns=[CONFIG["target_column"]]) # Sample 5 rows from original data excluding target
        # Need to add the target column back with dummy values if preprocess_data expects it, or modify preprocess_data
        # Assuming preprocess_data can handle data without the target for inference
        logging.info("\nDummy New Data for Prediction:")
        logging.info(dummy_new_data)

        predictions = predict(dummy_new_data, loaded_model, loaded_scaler)

        if predictions is not None:
            logging.info("\nPredictions on Dummy Data:")
            logging.info(predictions)