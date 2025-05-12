import logging
import os

def setup_logging(level=logging.INFO, log_file: Optional[str] = None):
    """Configures the application's logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()] # Log to console

    if log_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file)) # Example: Log to a file

    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    logging.getLogger(__name__).info("Logging configured.") # Log from this module