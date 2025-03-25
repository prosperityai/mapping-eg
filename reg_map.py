# File: reg_map.py (updated with knowledge base support)

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
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)

def display_documents_preview(documents: List[Document], num_docs: int = 5, is_kb: bool = False) -> None:
    """Display a preview of the ingested documents"""
    if not documents:
        st.warning("No documents to display")
        return

    st.write(f"### Preview of {len(documents)} Ingested Documents")

    # Show the first few documents
    for i, doc in enumerate(documents[:num_docs]):
        # Create an appropriate title based on document type
        if is_kb:
            title = f"Document {i+1}: {doc.metadata.get('file_name', 'Unknown')} - Page {doc.metadata.get('page', 'Unknown')}"
        else:
            title = f"Document {i+1}: {doc.metadata.get('title', 'No Title')}"

        with st.expander(title):
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
        elif file_type == "kyc_policy":
            documents, error = agent.ingest_kyc_policy(temp_path)
        elif file_type == "knowledge_base":
            documents, error = agent.ingest_pdf_document(temp_path)

            # For knowledge base, also save a permanent copy
            if not error:
                kb_file_path = os.path.join(KB_DIR, uploaded_file.name)
                with open(kb_file_path, "wb") as f:
                    # Reopen the temp file in binary mode
                    with open(temp_path, "rb") as temp_f:
                        f.write(temp_f.read())
                logger.info(f"Saved knowledge base file to: {kb_file_path}")
        else:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Error removing temp file {temp_path}: {str(e)}")
            return [], None, f"Unknown file type: {file_type}"

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
        elif file_type == "kyc_policy":
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
        elif file_type == "knowledge_base":
            # For knowledge base, create a simple DataFrame with file and page info
            rows = []
            for doc in documents:
                row = {
                    "File": doc.metadata.get("file_name", "Unknown"),
                    "Page": doc.metadata.get("page", "Unknown"),
                    "Content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                }
                rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
            else:
                df = None
        else:
            df = None

        return documents, df, None

    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
        return [], None, f"Error processing file: {str(e)}"

def handle_multiple_uploads(uploaded_files, agent: IngestionAgent, file_type: str) -> Tuple[List[Document], pd.DataFrame, Optional[str]]:
    """Process multiple uploaded files using the ingestion agent"""
    if not uploaded_files:
        return [], None, "No files uploaded"

    all_documents = []
    all_rows = []

    for uploaded_file in uploaded_files:
        try:
            documents, df, error = handle_uploaded_file(uploaded_file, agent, file_type)
            if error:
                return [], None, f"Error processing {uploaded_file.name}: {error}"

            all_documents.extend(documents)

            # Combine DataFrames if possible
            if df is not None and not df.empty:
                if file_type == "knowledge_base":
                    all_rows.extend(df.to_dict('records'))

        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}", exc_info=True)
            return [], None, f"Error processing {uploaded_file.name}: {str(e)}"

    # Create combined DataFrame
    if all_rows:
        combined_df = pd.DataFrame(all_rows)
    else:
        combined_df = None

    return all_documents, combined_df, None

def main():
    st.title("Regulatory Mapping MVP (LLM-driven)")

    # Initialize the ingestion agent
    ingestion_agent = IngestionAgent(chunk_size=1000, chunk_overlap=200)

    # Session state for persistence
    if "regulatory_documents" not in st.session_state:
        st.session_state.regulatory_documents = []
    if "kyc_documents" not in st.session_state:
        st.session_state.kyc_documents = []
    if "kb_documents" not in st.session_state:
        st.session_state.kb_documents = []
    if "regulatory_df" not in st.session_state:
        st.session_state.regulatory_df = None
    if "kyc_df" not in st.session_state:
        st.session_state.kyc_df = None
    if "kb_df" not in st.session_state:
        st.session_state.kb_df = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "error_message" not in st.session_state:
        st.session_state.error_message = None

    # Step 0: File upload interface
    if st.session_state.current_step == 0:
        st.header("Step 1: Upload Documents")

        # Show any error messages from previous operations
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            st.session_state.error_message = None

        # File upload section for regulatory requirements
        st.subheader("1. Regulatory Requirements")
        st.write("Upload the Excel file containing the regulatory obligations you need to map.")

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
        st.subheader("2. KYC Policy (B1 Mastersheet)")
        st.write("Upload the Excel file containing your internal KYC policies (B1 Mastersheet).")

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

        # File upload section for knowledge base
        st.markdown("---")
        st.subheader("3. Knowledge Base Documents")
        st.write("Upload PDF documents that contain additional information for classifying regulatory obligations.")

        uploaded_kb_files = st.file_uploader(
            "Upload PDF knowledge base documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="kb_uploader",
            help="Upload one or more PDF documents containing policy guidance, definitions, and other relevant information"
        )

        if uploaded_kb_files:
            with st.spinner("Processing knowledge base documents..."):
                documents, df, error = handle_multiple_uploads(uploaded_kb_files, ingestion_agent, "knowledge_base")

                if error:
                    st.error(f"Error processing knowledge base documents: {error}")
                else:
                    # Add to existing KB documents if any
                    st.session_state.kb_documents.extend(documents)

                    # Update DataFrame
                    if df is not None:
                        if st.session_state.kb_df is not None:
                            # Combine with existing DataFrame
                            st.session_state.kb_df = pd.concat([st.session_state.kb_df, df])
                        else:
                            st.session_state.kb_df = df

                    # Display success message
                    file_names = [f.name for f in uploaded_kb_files]
                    st.success(f"Successfully processed {len(documents)} pages from {len(file_names)} knowledge base documents")

                    # Display preview
                    if df is not None:
                        st.write("Preview of knowledge base content:")
                        st.dataframe(df.head(5), use_container_width=True)

        # Show existing KB documents if available
        elif st.session_state.kb_documents:
            st.success(f"Already loaded {len(st.session_state.kb_documents)} knowledge base document chunks")

            if st.session_state.kb_df is not None:
                st.write("Preview of knowledge base content:")
                st.dataframe(st.session_state.kb_df.head(5), use_container_width=True)

        # Also provide an option to use existing knowledge base documents from storage
        kb_files_in_storage = [f for f in os.listdir(KB_DIR) if f.lower().endswith('.pdf')]
        if kb_files_in_storage and not st.session_state.kb_documents:
            st.write("Previously uploaded knowledge base documents:")
            if st.button("Load Existing Knowledge Base Documents"):
                with st.spinner("Loading existing knowledge base documents..."):
                    try:
                        documents, error = ingestion_agent.ingest_directory_of_documents(KB_DIR)
                        if error:
                            st.error(f"Error loading existing knowledge base: {error}")
                        else:
                            st.session_state.kb_documents = documents

                            # Create DataFrame for display
                            rows = []
                            for doc in documents:
                                row = {
                                    "File": doc.metadata.get("file_name", "Unknown"),
                                    "Page": doc.metadata.get("page", "Unknown"),
                                    "Content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                                }
                                rows.append(row)

                            if rows:
                                st.session_state.kb_df = pd.DataFrame(rows)

                            st.success(f"Successfully loaded {len(documents)} document chunks from {len(kb_files_in_storage)} existing PDF files")
                    except Exception as e:
                        st.error(f"Error loading existing knowledge base: {str(e)}")

        # Navigation buttons
        st.markdown("---")
        ready_to_proceed = (
                st.session_state.regulatory_documents and
                st.session_state.kyc_documents and
                st.session_state.kb_documents
        )

        if not ready_to_proceed:
            missing = []
            if not st.session_state.regulatory_documents:
                missing.append("Regulatory Requirements")
            if not st.session_state.kyc_documents:
                missing.append("KYC Policy")
            if not st.session_state.kb_documents:
                missing.append("Knowledge Base Documents")

            st.warning(f"Please upload all required documents before proceeding. Missing: {', '.join(missing)}")

        if ready_to_proceed:
            if st.button("Next: Review Documents"):
                st.session_state.current_step = 1
                st.experimental_rerun()

    # Step 1: Document details and validation
    elif st.session_state.current_step == 1:
        st.header("Step 2: Review Ingested Documents")

        # Display summary statistics
        col1, col2, col3 = st.columns(3)

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
                            for val, count in list(value_counts.items())[:3]:  # Show top 3
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
                        for val, count in list(master_counts.items())[:3]:  # Show top 3
                            st.write(f"  - {val}: {count}")
                        if len(master_counts) > 3:
                            st.write(f"  - ... and {len(master_counts) - 3} more")

        with col3:
            st.metric("Knowledge Base Chunks", len(st.session_state.kb_documents))
            st.write("**Knowledge Base Details:**")
            if st.session_state.kb_documents:
                # Count unique PDF files
                file_names = [doc.metadata.get("file_name", "Unknown") for doc in st.session_state.kb_documents]
                unique_files = len(set(file_names))
                st.write(f"- {unique_files} unique PDF documents")
                st.write(f"- {len(st.session_state.kb_documents)} total chunks")

                # List the file names
                st.write("- Files:")
                for file in list(set(file_names))[:3]:  # Show top 3
                    st.write(f"  - {file}")
                if unique_files > 3:
                    st.write(f"  - ... and {unique_files - 3} more")

        # Display document previews
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Regulatory Requirements", "KYC Policy", "Knowledge Base"])

        with tab1:
            display_documents_preview(st.session_state.regulatory_documents)

        with tab2:
            display_documents_preview(st.session_state.kyc_documents)

        with tab3:
            display_documents_preview(st.session_state.kb_documents, is_kb=True)

        # Navigation
        st.markdown("---")
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
