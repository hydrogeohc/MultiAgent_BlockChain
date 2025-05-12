import os
import json
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Application Configuration ---
APP_CONFIG = {
    "data_file_path": os.getenv("DATA_FILE_PATH", "./forta_hacked_address_features.csv"),
    # Specific API keys for different providers
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"), # Get from environment
    "llm_model_name": os.getenv("LLM_MODEL_NAME", "gpt-4o"), # Default to gpt-4o (or "anthropic/claude-3-5-sonnet-20240620")
    "llm_temperature": float(os.getenv("LLM_TEMPERATURE", 0.1)),  # Default temperature
    "crew_process": os.getenv("CREW_PROCESS", "hierarchical"),  # Default to hierarchical
    "llm_request_timeout": float(os.getenv("LLM_REQUEST_TIMEOUT", 60.0)), # Configurable timeout
    # Add other configurations here (e.g., thresholds, database connections, API endpoints)
    "ethics_checker_config": {
        "flag_column": os.getenv("ETHICS_FLAG_COLUMN", "FLAG"),
        "fraud_label": int(os.getenv("ETHICS_FRAUD_LABEL", 1)),
        "normal_label": int(os.getenv("ETHICS_NORMAL_LABEL", 0)),
        "high_correlation_threshold": float(os.getenv("ETHICS_HIGH_CORRELATION_THRESHOLD", 0.8)),
        # Transparency quantiles could be loaded from a JSON string env var or a file
        "transparency_quantile_percentiles": json.loads(os.getenv("ETHICS_TRANSPARENCY_QUANTILES", '{}')) or {
            'max value received': 0.95,
            'total transactions (including tnx to create contract': 0.95,
            'Time Diff between first and last (Mins)': 0.05
        },
         # Add names of columns relevant for correlation/bias check if different from all numeric
         "bias_feature_columns": None # Or list relevant columns
    },
     "contract_miner_config": {
         "flag_column": os.getenv("MINER_FLAG_COLUMN", "FLAG"),
         "random_state": int(os.getenv("MINER_RANDOM_STATE", 42)),
         "fraud_label": int(os.getenv("MINER_FRAUD_LABEL", 1)), # Ensure consistency
         "normal_label": int(os.getenv("MINER_NORMAL_LABEL", 0)), # Ensure consistency
         # Range and step could be dynamic based on tasks or config
     }
    # Add config for Fraud Detection algorithms (e.g., model paths, parameters)
}

# Add any configuration validation logic here
def validate_config(config: Dict[str, Any]):
    """Performs basic validation of the application configuration."""
    if not config["data_file_path"]:
        raise ValueError("DATA_FILE_PATH is not set in environment variables or .env file.")

    llm_model = config["llm_model_name"].lower()
    if "anthropic" in llm_model and not config.get("anthropic_api_key"):
         raise ValueError("ANTHROPIC_API_KEY is not set for the selected LLM model.")
    elif "openai" in llm_model and not config.get("openai_api_key"):
         raise ValueError("OPENAI_API_KEY is not set for the selected LLM model.")
    # Add other critical validation checks as needed