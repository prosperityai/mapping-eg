# File: setup_check.py
# Run this script to check if your environment is properly set up

import os
import sys
import importlib
import dotenv

def check_directories():
    """Check if required directories exist and create them if needed"""
    directories = [
        "data",
        "data/temp",
        "data/documents",
        "data/knowledge_base",
        "data/vectors",
        "agents",
        "utils"
    ]

    print("\n=== Checking directories ===")
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ '{directory}' exists")
        else:
            print(f"❌ '{directory}' does not exist - creating it now")
            os.makedirs(directory, exist_ok=True)

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = ".env"

    print("\n=== Checking .env file ===")
    if os.path.exists(env_path):
        print(f"✅ '{env_path}' exists")
        # Load environment variables
        dotenv.load_dotenv(env_path)

        # Check required variables
        required_vars = ["OPENAI_API_KEY"]
        for var in required_vars:
            if os.environ.get(var):
                print(f"✅ {var} is set")
            else:
                print(f"❌ {var} is missing in .env file")
    else:
        print(f"❌ '{env_path}' does not exist - please create it")
        print("Sample .env file content:")
        print("""
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-4o

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Debug Mode
DEBUG_MODE=false
        """)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "langchain",
        "langchain_openai",
        "langchain_community",
        "faiss",
        "openai",
        "python-dotenv",
        "pypdf",
        "openpyxl",
        "unstructured",
        "pydantic"
    ]

    print("\n=== Checking dependencies ===")
    missing_packages = []

    for package in required_packages:
        try:
            # Handle packages with underscores in import
            import_name = package.replace("-", "_")
            importlib.import_module(import_name)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)

    if missing_packages:
        print("\nMissing packages. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")

def check_agent_files():
    """Check if agent files exist"""
    agent_files = [
        "agents/__init__.py",
        "agents/ingestion_agent.py",
        "agents/embedding_agent.py",
        "agents/classification_agent.py"
    ]

    print("\n=== Checking agent files ===")
    for file_path in agent_files:
        if os.path.exists(file_path):
            print(f"✅ '{file_path}' exists")
        else:
            print(f"❌ '{file_path}' is missing")

def main():
    """Run all checks"""
    print("=== Regulatory Mapping Tool Setup Checker ===")

    check_directories()
    check_env_file()
    check_dependencies()
    check_agent_files()

    print("\n=== Check complete ===")
    print("If you see any issues above, please fix them before running the application.")
    print("If everything looks good, you can run the application with:")
    print("streamlit run reg_map.py")

if __name__ == "__main__":
    main()
