# File: reg_map.py (Main Application)

import streamlit as st
import os
import traceback
import logging
import time

# Import agents
from agents import IngestionAgent, EmbeddingAgent, ClassificationAgent

# Import UI components
from ui import (
    display_upload_page,
    display_review_page,
    display_vectorize_page,
    display_classification_page,
    display_export_page
)

# Import utility functions
from utils import load_config, validate_api_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reg_map.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('reg_map')

# Setup constants and paths
DATA_DIR = "data"
TEMP_DIR = os.path.join(DATA_DIR, "temp")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")
VECTOR_DIR = os.path.join(DATA_DIR, "vectors")

def setup_directories():
    """Ensure all required directories exist"""
    directories = [DATA_DIR, TEMP_DIR, DOCUMENTS_DIR, KB_DIR, VECTOR_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory {directory} exists or was created")

def initialize_agents(config):
    """Initialize the agents with the provided configuration"""
    try:
        logger.info("Initializing ingestion agent")
        ingestion_agent = IngestionAgent(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )

        logger.info("Initializing embedding agent")
        # Initialize embedding agent (if not already in session)
        if "embedding_agent" not in st.session_state:
            st.session_state.embedding_agent = EmbeddingAgent(
                openai_api_key=config["openai_api_key"],
                model_name=config["embedding_model"]
            )
        embedding_agent = st.session_state.embedding_agent

        logger.info("Initializing classification agent")
        # Initialize classification agent (if not already in session)
        if "classification_agent" not in st.session_state:
            st.session_state.classification_agent = ClassificationAgent(
                openai_api_key=config["openai_api_key"],
                model_name=config["llm_model"]
            )
        classification_agent = st.session_state.classification_agent

        return ingestion_agent, embedding_agent, classification_agent

    except Exception as e:
        logger.error(f"Error initializing agents: {str(e)}", exc_info=True)
        st.error(f"Error initializing agents: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None

def initialize_session_state():
    """Initialize session state variables if not already set"""
    # Step tracking
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    # Error message handling
    if "error_message" not in st.session_state:
        st.session_state.error_message = None

    # Document storage
    if "regulatory_documents" not in st.session_state:
        st.session_state.regulatory_documents = []
    if "kyc_documents" not in st.session_state:
        st.session_state.kyc_documents = []
    if "kb_documents" not in st.session_state:
        st.session_state.kb_documents = []

    # Data frame storage for display
    if "regulatory_df" not in st.session_state:
        st.session_state.regulatory_df = None
    if "kyc_df" not in st.session_state:
        st.session_state.kyc_df = None
    if "kb_df" not in st.session_state:
        st.session_state.kb_df = None

    # Vector store tracking
    if "vector_stores_built" not in st.session_state:
        st.session_state.vector_stores_built = False

    # Classification tracking
    if "classified_requirements" not in st.session_state:
        st.session_state.classified_requirements = []
    if "current_requirement_idx" not in st.session_state:
        st.session_state.current_requirement_idx = 0
    if "classifications_completed" not in st.session_state:
        st.session_state.classifications_completed = False
    if "classification_results" not in st.session_state:
        st.session_state.classification_results = []

def main():
    """Main application entry point"""
    try:
        # Set up page configuration
        st.set_page_config(
            page_title="Regulatory Mapping Tool",
            page_icon="📋",
            layout="wide"
        )

        # Add a sidebar for debug options
        with st.sidebar:
            st.title("Regulatory Mapping Tool")

            # Debug mode toggle
            if "debug_mode" not in st.session_state:
                st.session_state.debug_mode = False

            st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)

            if st.session_state.debug_mode:
                st.write("### Session State Variables")

                # Display session state variables useful for debugging
                debug_vars = [
                    "current_step",
                    "vector_stores_built",
                    "reg_vector_store",
                    "kyc_vector_store",
                    "kb_vector_store"
                ]

                for var in debug_vars:
                    if var in st.session_state:
                        st.write(f"{var}: {st.session_state[var]}")
                    else:
                        st.write(f"{var}: Not set")

                # Reset button for troubleshooting
                if st.button("Reset Application State"):
                    # Keep only embedding and classification agents
                    keep_vars = ["embedding_agent", "classification_agent", "debug_mode"]
                    for key in list(st.session_state.keys()):
                        if key not in keep_vars:
                            del st.session_state[key]
                    st.success("Application state reset!")
                    st.rerun()

            # Navigation links
            st.write("### Navigation")
            steps = [
                "Upload Documents",
                "Review Documents",
                "Build Vector Indices",
                "Classify Requirements",  # Changed from "Classification & Mapping"
                "Export Results"          # Will be replaced with "Map Requirements" in the future
            ]

            for i, step in enumerate(steps):
                if st.button(f"{i+1}. {step}"):
                    st.session_state.current_step = i
                    st.rerun()

        # Display application title
        st.title("Regulatory Mapping Tool (LLM-driven)")

        # Ensure directories exist
        setup_directories()

        # Load configuration
        config = load_config()

        # Validate API keys
        error_msg = validate_api_keys(config)
        if error_msg:
            st.error(error_msg)
            st.stop()

        # Initialize session state
        initialize_session_state()

        # Log the current step
        logger.info(f"Current step: {st.session_state.current_step}")

        # Initialize agents
        ingestion_agent, embedding_agent, classification_agent = initialize_agents(config)
        if None in (ingestion_agent, embedding_agent, classification_agent):
            st.error("Failed to initialize one or more agents. Please check the logs.")
            st.stop()

        # Display the appropriate page based on current step
        if st.session_state.current_step == 0:
            logger.info("Displaying upload page")
            display_upload_page(ingestion_agent, KB_DIR)
        elif st.session_state.current_step == 1:
            logger.info("Displaying review page")
            display_review_page()
        elif st.session_state.current_step == 2:
            logger.info("Displaying vectorize page")
            display_vectorize_page(embedding_agent, VECTOR_DIR)
        elif st.session_state.current_step == 3:
            logger.info("Displaying classification page")
            display_classification_page(embedding_agent, classification_agent)
        elif st.session_state.current_step == 4:
            logger.info("Displaying export page")
            display_export_page()
        else:
            logger.error(f"Unknown step: {st.session_state.current_step}")
            st.error(f"Unknown step: {st.session_state.current_step}")
            st.session_state.current_step = 0
            st.rerun()

    except Exception as e:
        logger.error(f"Unexpected error in main application: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
