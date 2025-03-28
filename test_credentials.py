#!/usr/bin/env python
# File: test_credentials.py

import os
import logging
import argparse
import json
import requests
from dotenv import load_dotenv
import traceback

# For Azure AD authentication
try:
    from azure.identity import CertificateCredential
except ImportError:
    print("Warning: azure-identity package not installed. Install with: pip install azure-identity")

# For standard OpenAI
try:
    import openai
    from openai import OpenAI, AzureOpenAI
except ImportError:
    print("Warning: openai package not installed. Install with: pip install openai")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_credentials')

def clean_environment_variables():
    """Clean problematic environment variables that might cause connection issues"""
    # Check for and remove problematic proxy settings
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        if var in os.environ and (
                "port" in os.environ[var] or  # Matches the literal string "port"
                "://" not in os.environ[var] or  # Missing protocol
                os.environ[var].endswith(":")  # Missing port number
        ):
            logger.warning(f"Removing invalid proxy setting: {var}={os.environ[var]}")
            del os.environ[var]

    logger.debug("Environment variables cleaned")

def generate_azure_ad_token() -> str:
    """Generate Azure AD token using certificate authentication"""
    logger.info("Generating Azure AD token...")

    tenant_id = os.getenv("AD_TENANT_ID")
    client_id = os.getenv("AD_CLIENT_ID")
    certificate_path = os.getenv("OPEN_AI_CERTIFICATE_PATH")

    if not all([tenant_id, client_id, certificate_path]):
        raise ValueError("Missing required Azure AD credentials: AD_TENANT_ID, AD_CLIENT_ID, OPEN_AI_CERTIFICATE_PATH")

    scope = "https://cognitiveservices.azure.com/.default"

    credential = CertificateCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        certificate_path=certificate_path,
    )

    token = credential.get_token(scope).token
    logger.info("Azure AD token generated successfully")

    return token

def test_standard_openai():
    """Test connection to standard OpenAI API"""
    logger.info("Testing Standard OpenAI connection...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return False

    try:
        # Initialize client
        client = OpenAI(api_key=api_key)

        # List models
        models = client.models.list()

        logger.info(f"Successfully connected to OpenAI API")
        logger.info(f"Available models:")

        for model in models.data:
            logger.info(f"- {model.id}")

        return True
    except Exception as e:
        logger.error(f"Error connecting to OpenAI: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def test_azure_openai_with_key():
    """Test connection to Azure OpenAI using API key"""
    logger.info("Testing Azure OpenAI connection with API key...")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

    if not all([api_key, endpoint]):
        logger.error("Missing required Azure OpenAI credentials: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")
        return False

    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

        # List deployments
        deployments = list_azure_deployments_direct(endpoint, api_version, api_key)

        if deployments:
            logger.info(f"Successfully connected to Azure OpenAI API")
            logger.info(f"Available deployments:")

            for deployment in deployments:
                model = deployment.get("model", "unknown")
                deployment_id = deployment.get("id", "unknown")
                logger.info(f"- {deployment_id} (model: {model})")

            return True
        else:
            logger.error("Failed to list Azure OpenAI deployments")
            return False

    except Exception as e:
        logger.error(f"Error connecting to Azure OpenAI with API key: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def test_azure_openai_with_token():
    """Test connection to Azure OpenAI using Azure AD token"""
    logger.info("Testing Azure OpenAI connection with Azure AD token...")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

    if not endpoint:
        logger.error("AZURE_OPENAI_ENDPOINT not found in environment variables")
        return False

    try:
        # Generate token
        token = generate_azure_ad_token()

        # List deployments using direct API call
        deployments = list_azure_deployments_direct(endpoint, api_version, token, token_auth=True)

        if deployments:
            logger.info(f"Successfully connected to Azure OpenAI API with token")
            logger.info(f"Available deployments:")

            for deployment in deployments:
                model = deployment.get("model", "unknown")
                deployment_id = deployment.get("id", "unknown")
                logger.info(f"- {deployment_id} (model: {model})")

            return True
        else:
            logger.error("Failed to list Azure OpenAI deployments")
            return False

    except Exception as e:
        logger.error(f"Error connecting to Azure OpenAI with token: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def list_azure_deployments_direct(endpoint, api_version, auth_key, token_auth=False):
    """
    List Azure OpenAI deployments using direct REST API call

    Args:
        endpoint: Azure OpenAI endpoint URL
        api_version: API version string
        auth_key: API key or Azure AD token
        token_auth: True if auth_key is an Azure AD token, False if it's an API key

    Returns:
        List of deployment objects or None if failed
    """
    # Ensure URL doesn't have trailing slash
    if endpoint.endswith('/'):
        endpoint = endpoint[:-1]

    # Try both possible endpoint formats
    urls = [
        f"{endpoint}/openai/deployments?api-version={api_version}",
        f"{endpoint}/deployments?api-version={api_version}"
    ]

    if token_auth:
        headers = {
            "Authorization": f"Bearer {auth_key}",
            "Content-Type": "application/json"
        }
    else:
        headers = {
            "api-key": auth_key,
            "Content-Type": "application/json"
        }

    for url in urls:
        try:
            logger.debug(f"Trying endpoint: {url}")
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                return data.get("value", [])
            else:
                logger.debug(f"API call failed with status {response.status_code}: {response.text}")
        except Exception as e:
            logger.debug(f"Request failed: {str(e)}")

    logger.error("All endpoint attempts failed")
    return None

def detect_available_connection_types():
    """Detect which connection types are available based on environment variables"""
    connection_types = []

    # Check for standard OpenAI credentials
    if os.getenv("OPENAI_API_KEY"):
        connection_types.append("standard")
        logger.info("Detected standard OpenAI API key")

    # Check for Azure OpenAI API key credentials
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        connection_types.append("azure-key")
        logger.info("Detected Azure OpenAI API key configuration")

    # Check for Azure AD token authentication prerequisites
    if (os.getenv("AD_TENANT_ID") and
            os.getenv("AD_CLIENT_ID") and
            os.getenv("OPEN_AI_CERTIFICATE_PATH") and
            os.getenv("AZURE_OPENAI_ENDPOINT")):
        connection_types.append("azure-token")
        logger.info("Detected Azure AD token authentication configuration")

    return connection_types

def main():
    parser = argparse.ArgumentParser(description='Test OpenAI and Azure OpenAI credentials')
    parser.add_argument('--type', choices=['standard', 'azure-key', 'azure-token', 'all', 'auto'],
                        default='auto', help='Type of credentials to test')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--env-file', default='.env', help='Path to .env file')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Load environment variables
    load_dotenv(args.env_file)
    logger.info(f"Loaded environment from {args.env_file}")

    # Clean environment variables to avoid connection issues
    clean_environment_variables()

    # Determine which connection types to test
    if args.type == 'auto':
        # Auto-detect connection types
        connection_types = detect_available_connection_types()
        if not connection_types:
            logger.error("No valid connection configurations detected in environment variables")
            return
        logger.info(f"Detected connection types: {', '.join(connection_types)}")
    elif args.type == 'all':
        # Test all types
        connection_types = ['standard', 'azure-key', 'azure-token']
        logger.info(f"Testing all connection types")
    else:
        # Test only the specified type
        connection_types = [args.type]
        logger.info(f"Testing only {args.type} connection")

    results = {}

    # Only test standard OpenAI if specified
    if 'standard' in connection_types:
        results['standard_openai'] = test_standard_openai()

    # Only test Azure OpenAI with API key if specified
    if 'azure-key' in connection_types:
        results['azure_openai_key'] = test_azure_openai_with_key()

    # Only test Azure OpenAI with token if specified
    if 'azure-token' in connection_types:
        results['azure_openai_token'] = test_azure_openai_with_token()

    # Print summary
    logger.info("\n=== TEST RESULTS SUMMARY ===")
    for test_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

def list_deployments():
    """
    List all available Azure OpenAI deployments and their details.

    Returns:
        A list of deployment objects
    """
    print("Listing all deployments...")

    try:
        # Create Azure OpenAI client with the token generated earlier
        client = AzureOpenAI(
            api_key=openai.api_key,
            api_version="2023-05-15",
            azure_endpoint=openai.api_base
        )

        # Get list of deployments
        deployments = client.models.list()

        # Print details about each deployment
        print(f"Found {len(deployments.data) if hasattr(deployments, 'data') else 0} deployments:")

        if hasattr(deployments, 'data'):
            for i, deployment in enumerate(deployments.data):
                print(f"\nDeployment #{i+1}:")
                print(f"  ID: {deployment.id}")

                # Print additional details if available
                for attr in ['model', 'created_at', 'owned_by', 'object']:
                    if hasattr(deployment, attr):
                        print(f"  {attr.replace('_', ' ').title()}: {getattr(deployment, attr)}")

        # Alternative approach using direct API call if the client method doesn't work
        if not hasattr(deployments, 'data') or len(deployments.data) == 0:
            print("\nTrying alternative API method to list deployments...")
            direct_deployments = list_deployments_direct()
            if direct_deployments and len(direct_deployments) > 0:
                print(f"Found {len(direct_deployments)} deployments via direct API call:")
                for i, deployment in enumerate(direct_deployments):
                    print(f"\nDeployment #{i+1}:")
                    for key, value in deployment.items():
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                print("No deployments found using direct API call")

        return deployments

    except Exception as e:
        print(f"Error listing deployments: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


def list_deployments_direct():
    """
    List deployments using direct REST API call as a fallback method.

    Returns:
        List of deployment objects or None if failed
    """
    import requests

    # Ensure URL doesn't have trailing slash
    base_url = openai.api_base
    if base_url.endswith('/'):
        base_url = base_url[:-1]

    # Try both possible endpoint formats
    urls = [
        f"{base_url}/openai/deployments?api-version=2023-05-15",
        f"{base_url}/deployments?api-version=2023-05-15"
    ]

    # Setup authentication header (using token from generate_access_token)
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    for url in urls:
        try:
            print(f"Trying endpoint: {url}")
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                return data.get("value", [])
            else:
                print(f"API call failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Request failed: {str(e)}")

    print("All endpoint attempts failed")
    return None

if __name__ == "__main__":
    main()
