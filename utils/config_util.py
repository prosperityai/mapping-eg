# File: utils/config_util.py

import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import traceback

# For Azure AD token authentication
try:
    from azure.identity import CertificateCredential
except ImportError:
    logging.warning("azure-identity package not installed. Azure AD token authentication will not be available.")

# Configure logging
logger = logging.getLogger('config_util')

def setup_proxies():
    """Set up proxy configuration from environment variables"""
    # Get proxy settings from environment
    http_proxy = os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("HTTPS_PROXY")
    no_proxy = os.getenv("NO_PROXY")

    # Validate proxy URLs before setting them
    if http_proxy:
        if "://" not in http_proxy or http_proxy.endswith("port"):
            logger.warning(f"Invalid HTTP_PROXY format: {http_proxy}. Skipping.")
        else:
            os.environ["http_proxy"] = http_proxy
            logger.info(f"HTTP proxy set to: {http_proxy}")

    if https_proxy:
        if "://" not in https_proxy or https_proxy.endswith("port"):
            logger.warning(f"Invalid HTTPS_PROXY format: {https_proxy}. Skipping.")
        else:
            os.environ["https_proxy"] = https_proxy
            logger.info(f"HTTPS proxy set to: {https_proxy}")

    if no_proxy:
        if 'no_proxy' in os.environ:
            os.environ["no_proxy"] = os.environ["no_proxy"] + " " + no_proxy
        else:
            os.environ["no_proxy"] = no_proxy
        logger.info(f"NO_PROXY set to: {os.environ['no_proxy']}")

    logger.info("Proxy setup completed")

def generate_azure_ad_token() -> Optional[str]:
    """
    Generate an access token using Azure AD Certificate Authentication.

    Returns:
        Access token string or None if generation fails
    """
    try:
        tenant_id = os.getenv("AD_TENANT_ID")
        client_id = os.getenv("AD_CLIENT_ID")
        certificate_path = os.getenv("OPEN_AI_CERTIFICATE_PATH")

        if not all([tenant_id, client_id, certificate_path]):
            logger.warning("Missing Azure AD credentials (tenant_id, client_id, or certificate_path)")
            return None

        scope = "https://cognitiveservices.azure.com/.default"

        credential = CertificateCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            certificate_path=certificate_path,
        )

        token = credential.get_token(scope).token
        logger.info("Azure AD access token generated successfully")

        return token
    except Exception as e:
        logger.error(f"Failed to generate Azure AD token: {str(e)}\n{traceback.format_exc()}")
        return None

def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables or .env file.
    Supports both standard OpenAI and Azure OpenAI configurations.

    Returns:
        Dictionary with configuration values
    """
    # Try to load from .env file if it exists
    load_dotenv(verbose=True)

    try:
        # Setup proxy configuration if needed
        setup_proxies()
    except Exception as e:
        logger.error(f"Error setting up proxies: {str(e)}")
        logger.info("Continuing without proxy configuration")

    # Determine if we're using Azure OpenAI
    use_azure = os.environ.get("USE_AZURE_OPENAI", "false").lower() == "true"

    # Basic configuration common to both OpenAI and Azure
    config = {
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o"),
        "chunk_size": int(os.environ.get("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", "200")),
        "debug_mode": os.environ.get("DEBUG_MODE", "false").lower() == "true",
        "use_azure": use_azure
    }

    # Clear potentially problematic environment variables
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        if var in os.environ and (os.environ[var].endswith("port") or "://" not in os.environ[var]):
            logger.warning(f"Removing invalid proxy setting: {var}={os.environ[var]}")
            del os.environ[var]

    if use_azure:
        # Azure OpenAI configuration
        logger.info("Configuring for Azure OpenAI")

        # Decide if we're using direct API key or Azure AD token
        use_azure_ad = os.environ.get("USE_AZURE_AD_AUTH", "false").lower() == "true"

        if use_azure_ad:
            # Azure AD token authentication
            logger.info("Using Azure AD token authentication")
            access_token = generate_azure_ad_token()

            if access_token:
                # Set up Azure OpenAI with token auth
                config["azure_api_type"] = "azure_ad"
                config["azure_api_key"] = access_token
            else:
                logger.error("Failed to generate Azure AD token. Falling back to API key if available.")
                use_azure_ad = False

        if not use_azure_ad:
            # Standard API key authentication
            azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not azure_api_key:
                logger.warning("Azure OpenAI API key not found")
            config["azure_api_key"] = azure_api_key
            config["azure_api_type"] = "azure"

        # Common Azure configuration
        config["azure_api_base"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
        config["azure_api_version"] = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")

        # Model deployments
        config["azure_embedding_deployment"] = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        config["azure_llm_deployment"] = os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT")

        # Set OpenAI environment variables for the LangChain integrations to automatically use Azure
        os.environ["OPENAI_API_TYPE"] = config["azure_api_type"]
        os.environ["OPENAI_API_BASE"] = config["azure_api_base"]
        os.environ["OPENAI_API_KEY"] = config["azure_api_key"]
        os.environ["OPENAI_API_VERSION"] = config["azure_api_version"]

        if config["azure_embedding_deployment"]:
            os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"] = config["azure_embedding_deployment"]
        if config["azure_llm_deployment"]:
            os.environ["OPENAI_DEPLOYMENT"] = config["azure_llm_deployment"]

    else:
        # Standard OpenAI configuration
        logger.info("Configuring for standard OpenAI")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OpenAI API key not found in environment variables or .env file")
        config["openai_api_key"] = openai_api_key

    return config

def validate_api_keys(config: Dict[str, Any]) -> Optional[str]:
    """
    Validate required API keys are present.

    Args:
        config: Configuration dictionary

    Returns:
        Error message if validation fails, None otherwise
    """
    if config.get("use_azure"):
        # Validate Azure OpenAI configuration
        if not config.get("azure_api_key"):
            return "Azure OpenAI API key or access token is missing"
        if not config.get("azure_api_base"):
            return "Azure OpenAI endpoint is missing"
        if not config.get("azure_api_version"):
            return "Azure OpenAI API version is missing"
        if not config.get("azure_embedding_deployment"):
            return "Azure OpenAI embedding model deployment name is missing"
    else:
        # Validate standard OpenAI configuration
        if not config.get("openai_api_key"):
            return "OpenAI API key is missing. Please add it to environment variables or .env file."

    return None
