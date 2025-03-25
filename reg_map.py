import streamlit as st
import pandas as pd
import os
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import time
import logging
from langchain.schema import Document

# Import the ingestion agent
from agents.ingestion_agent import IngestionAgent

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

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

def display_documents_preview(documents: List[Document], num_docs: int = 5) -> None:
    """Display a preview of the ingested documents"""
    if not documents:
        st.warning("No documents to display")
        return

    st.write(f"### Preview of {len(documents)} Ingested Documents")

    # Show the first few documents
    for i, doc in enumerate(documents[:num_docs]):
        with st.expander(f"Document {i+1}: {doc.metadata.get('title', 'No Title')}"):
            st.write("**Content:**")
            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

            st.write("**Metadata:**")
            meta_df = pd.DataFrame([{"Key": k, "Value": str(v)} for k, v in doc.metadata.items()])
            st.dataframe(meta_df, use_container_width=True)

def handle_uploaded_file(uploaded_file, agent: IngestionAgent, file_type: str) -> Tuple[List[Document], pd.DataFrame, Optional[str]]:
    """Process an uploaded file using the ingestion agent"""
    if uploaded_file is None:
        return [], None, "No file uploaded"

    try:
        # Create temp file path
        temp_path, error = agent.save_uploaded_file(uploaded_file)
        if error:
            return [], None, error

        # Process using the ingestion agent
        if file_type == "regulations":
            documents, error = agent.ingest_regulations(temp_path)
        else:  # kyc_policy
            documents, error = agent.ingest_kyc_policy(temp_path)

        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Error removing temp file {temp_path}: {str(e)}")

        if error:
            return [], None, error

        # Convert to DataFrame for display
        if file_type == "regulations":
            # For regulations, create a DataFrame with all original data
            rows = []
            for doc in documents:
                row = {}
                # Add all metadata from original_data if it exists
                if "original_data" in doc.metadata:
                    row.update(doc.metadata["original_data"])
                else:
                    # Otherwise use direct metadata
                    for k, v in doc.metadata.items():
                        if k != "source" and k != "file_path" and k != "file_name":
                            row[k] = v
                rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
            else:
                df = None
        else:
            # For KYC policy, create a DataFrame with key fields
            rows = []
            for doc in documents:
                row = {
                    "Master Sheet": doc.metadata.get("master_sheet", ""),
                    "Minor Sheet": doc.metadata.get("minor_sheet", ""),
                    "Section": doc.metadata.get("section", ""),
                    "Title": doc.metadata.get("title", ""),
                    "Content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                }
                # Add other metadata
                for k, v in doc.metadata.items():
                    if k not in ["master_sheet", "minor_sheet", "section", "title", "source", "file_path", "file_name", "row_index"]:
                        row[k] = v
                rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
            else:
                df = None

        return documents, df, None

    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
        return [], None, f"Error processing file: {str(e)}"

def main():
    st.title("Regulatory Mapping MVP (LLM-driven)")

    # Initialize the ingestion agent
    ingestion_agent = IngestionAgent(chunk_size=1000, chunk_overlap=200)

    # Session state for persistence
    if "regulatory_documents" not in st.session_state:
        st.session_state.regulatory_documents = []
    if "kyc_documents" not in st.session_state:
        st.session_state.kyc_documents = []
    if "regulatory_df" not in st.session_state:
        st.session_state.regulatory_df = None
    if "kyc_df" not in st.session_state:
        st.session_state.kyc_df = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "error_message" not in st.session_state:
        st.session_state.error_message = None

    # Step 0: File upload interface
    if st.session_state.current_step == 0:
        st.header("Step 1: Upload Regulatory Documents")

        # Show any error messages from previous operations
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            st.session_state.error_message = None

        # File upload section for regulatory requirements
        uploaded_reg_file = st.file_uploader(
            "Upload Excel file with regulatory requirements",
            type=["xlsx", "xls"],
            help="This should be the Excel file containing the regulatory obligations you need to map"
        )

        if uploaded_reg_file is not None:
            with st.spinner("Processing regulatory requirements..."):
                documents, df, error = handle_uploaded_file(uploaded_reg_file, ingestion_agent, "regulations")

                if error:
                    st.error(f"Error processing regulatory file: {error}")
                else:
                    st.session_state.regulatory_documents = documents
                    st.session_state.regulatory_df = df
                    st.success(f"Successfully processed {len(documents)} requirements from '{uploaded_reg_file.name}'")

                    # Display preview
                    if df is not None:
                        st.write("Preview of regulatory requirements:")
                        st.dataframe(df.head(5), use_container_width=True)

        # Show existing documents if available
        elif st.session_state.regulatory_documents:
            st.success(f"Already loaded {len(st.session_state.regulatory_documents)} regulatory requirements")

            if st.session_state.regulatory_df is not None:
                st.write("Preview of regulatory requirements:")
                st.dataframe(st.session_state.regulatory_df.head(5), use_container_width=True)

        # File upload section for KYC policy
        st.markdown("---")
        st.subheader("Upload KYC Policy (B1 Mastersheet)")

        uploaded_kyc_file = st.file_uploader(
            "Upload Excel file with KYC policy (B1 Mastersheet)",
            type=["xlsx", "xls"],
            key="kyc_uploader",
            help="This should be the B1 Mastersheet containing your internal KYC policies"
        )

        if uploaded_kyc_file is not None:
            with st.spinner("Processing KYC policy..."):
                documents, df, error = handle_uploaded_file(uploaded_kyc_file, ingestion_agent, "kyc_policy")

                if error:
                    st.error(f"Error processing KYC policy: {error}")
                else:
                    st.session_state.kyc_documents = documents
                    st.session_state.kyc_df = df
                    st.success(f"Successfully processed {len(documents)} policy items from '{uploaded_kyc_file.name}'")

                    # Display preview
                    if df is not None:
                        st.write("Preview of KYC policy:")
                        st.dataframe(df.head(5), use_container_width=True)

        # Show existing documents if available
        elif st.session_state.kyc_documents:
            st.success(f"Already loaded {len(st.session_state.kyc_documents)} KYC policy items")

            if st.session_state.kyc_df is not None:
                st.write("Preview of KYC policy:")
                st.dataframe(st.session_state.kyc_df.head(5), use_container_width=True)

        # Navigation buttons
        if st.session_state.regulatory_documents and st.session_state.kyc_documents:
            if st.button("Next: Create Vector Index"):
                st.session_state.current_step = 1
                st.experimental_rerun()

    # Step 1: Document details and validation
    elif st.session_state.current_step == 1:
        st.header("Step 2: Review Ingested Documents")

        # Display summary statistics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Regulatory Requirements", len(st.session_state.regulatory_documents))
            st.write("**Regulatory Document Details:**")
            if st.session_state.regulatory_documents:
                # Show sample data
                st.write(f"- Source: {st.session_state.regulatory_documents[0].metadata.get('file_name', 'Unknown')}")

                # Count by columns if available
                if st.session_state.regulatory_df is not None:
                    df = st.session_state.regulatory_df

                    # Try to find interesting columns to summarize
                    for col in df.columns:
                        if col.lower() in ['type', 'category', 'status', 'priority', 'section']:
                            value_counts = df[col].value_counts().to_dict()
                            st.write(f"- {col} distribution:")
                            for val, count in value_counts.items():
                                st.write(f"  - {val}: {count}")
                            break

        with col2:
            st.metric("KYC Policy Items", len(st.session_state.kyc_documents))
            st.write("**KYC Policy Details:**")
            if st.session_state.kyc_documents:
                # Show sample data
                st.write(f"- Source: {st.session_state.kyc_documents[0].metadata.get('file_name', 'Unknown')}")

                # Count by Master Sheet if available
                if st.session_state.kyc_df is not None:
                    df = st.session_state.kyc_df

                    if "Master Sheet" in df.columns:
                        master_counts = df["Master Sheet"].value_counts().to_dict()
                        st.write("- Master Sheet distribution:")
                        for val, count in list(master_counts.items())[:5]:  # Show top 5
                            st.write(f"  - {val}: {count}")
                        if len(master_counts) > 5:
                            st.write(f"  - ... and {len(master_counts) - 5} more")

        # Display document previews
        st.markdown("---")

        tab1, tab2 = st.tabs(["Regulatory Requirements", "KYC Policy"])

        with tab1:
            display_documents_preview(st.session_state.regulatory_documents)

        with tab2:
            display_documents_preview(st.session_state.kyc_documents)

        # Navigation
        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅️ Back to Upload"):
                st.session_state.current_step = 0
                st.experimental_rerun()

        with col2:
            if st.button("Next: Build Vector Index ➡️"):
                # This would be the next step after document ingestion
                st.session_state.error_message = "Vector Index functionality not yet implemented"
                st.session_state.current_step = 0  # Go back to first step for now
                st.experimental_rerun()

# Run the application
if __name__ == "__main__":
    main()
