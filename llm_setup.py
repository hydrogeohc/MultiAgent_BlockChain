import logging
from typing import Optional, Dict, Any
from crewai import LLM # Using CrewAI's native LLM wrapper

# Get a logger for this module
logger = logging.getLogger(__name__)

def initialize_llm(config: Dict[str, Any]) -> Optional[LLM]:
    """
    Initializes the CrewAI LLM based on configuration.

    Args:
        config (Dict[str, Any]): Application configuration dictionary.

    Returns:
        Optional[LLM]: The initialized CrewAI LLM instance, or None if initialization fails.
    """
    model_name = config.get("llm_model_name")
    temperature = config.get("llm_temperature", 0.1)
    request_timeout = config.get("llm_request_timeout", 60.0)

    logger.info(f"Initializing LLM: {model_name} with temperature {temperature} and timeout {request_timeout}s")

    if not model_name:
        logger.error("LLM model name is not specified in configuration.")
        return None

    try:
        # CrewAI's LLM wrapper handles picking up API keys from environment variables
        # based on the model name specified.
        llm = LLM(
            model=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            # Add other relevant parameters from config if supported by LLM wrapper
        )
        # Basic check to see if the LLM is responsive (optional but good)
        # try:
        #      llm.invoke("Hello world!", max_tokens=5) # Use a small test call
        #      logger.info("LLM responsiveness check passed.")
        # except Exception as e:
        #      logger.warning(f"LLM responsiveness check failed: {e}")


        logger.info(f"LLM '{model_name}' initialized successfully.")
        return llm

    except Exception as e:
        logger.error(f"Error initializing LLM '{model_name}': {e}", exc_info=True)
        return None