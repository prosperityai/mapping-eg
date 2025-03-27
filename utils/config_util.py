# File: utils/config_util.py

import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger('config_util')

def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables or .env file.

    Returns:
        Dictionary with configuration values
    """
    # Try to load from .env file if it exists
    load_dotenv(verbose=True)

    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OpenAI API key not found in environment variables or .env file")

    # Get other configuration values
    config = {
        "openai_api_key": openai_api_key,
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o"),
        "chunk_size": int(os.environ.get("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", "200")),
        "debug_mode": os.environ.get("DEBUG_MODE", "false").lower() == "true"
    }

    return config

def validate_api_keys(config: Dict[str, Any]) -> Optional[str]:
    """
    Validate required API keys are present.

    Args:
        config: Configuration dictionary

    Returns:
        Error message if validation fails, None otherwise
    """
    if not config.get("openai_api_key"):
        return "OpenAI API key is missing. Please add it to environment variables or .env file."

    return None
