# File: ui/upload_page.py

import streamlit as st
import pandas as pd
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
from langchain.schema import Document

# Configure logging
logger = logging.getLogger('upload_page')

def handle_uploaded_file(uploaded_file, agent, file_type: str, kb_dir: str) -> Tuple[List[Document], pd.DataFrame, Optional[str]]:
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
                kb_file_path = os.path.join(kb_dir, uploaded_file.name)
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

def handle_multiple_uploads(uploaded_files, agent, file_type: str, kb_dir: str) -> Tuple[List[Document], pd.DataFrame, Optional[str]]:
    """Process multiple uploaded files using the ingestion agent"""
    if not uploaded_files:
        return [], None, "No files uploaded"

    all_documents = []
    all_rows = []

    for uploaded_file in uploaded_files:
        try:
            documents, df, error = handle_uploaded_file(uploaded_file, agent, file_type, kb_dir)
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

def display_upload_page(ingestion_agent, kb_dir):
    """Display the upload interface"""
    st.header("Step 1: Upload Documents")

    # Show any error messages from previous operations
    if "error_message" in st.session_state and st.session_state.error_message:
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
            documents, df, error = handle_uploaded_file(uploaded_reg_file, ingestion_agent, "regulations", kb_dir)

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
    elif "regulatory_documents" in st.session_state and st.session_state.regulatory_documents:
        st.success(f"Already loaded {len(st.session_state.regulatory_documents)} regulatory requirements")

        if "regulatory_df" in st.session_state and st.session_state.regulatory_df is not None:
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
            documents, df, error = handle_uploaded_file(uploaded_kyc_file, ingestion_agent, "kyc_policy", kb_dir)

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
    elif "kyc_documents" in st.session_state and st.session_state.kyc_documents:
        st.success(f"Already loaded {len(st.session_state.kyc_documents)} KYC policy items")

        if "kyc_df" in st.session_state and st.session_state.kyc_df is not None:
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
            documents, df, error = handle_multiple_uploads(uploaded_kb_files, ingestion_agent, "knowledge_base", kb_dir)

            if error:
                st.error(f"Error processing knowledge base documents: {error}")
            else:
                # Add to existing KB documents if any
                if "kb_documents" not in st.session_state:
                    st.session_state.kb_documents = []

                st.session_state.kb_documents.extend(documents)

                # Update DataFrame
                if df is not None:
                    if "kb_df" in st.session_state and st.session_state.kb_df is not None:
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
    elif "kb_documents" in st.session_state and st.session_state.kb_documents:
        st.success(f"Already loaded {len(st.session_state.kb_documents)} knowledge base document chunks")

        if "kb_df" in st.session_state and st.session_state.kb_df is not None:
            st.write("Preview of knowledge base content:")
            st.dataframe(st.session_state.kb_df.head(5), use_container_width=True)

    # Also provide an option to use existing knowledge base documents from storage
    kb_files_in_storage = [f for f in os.listdir(kb_dir) if f.lower().endswith('.pdf')]
    if kb_files_in_storage and ("kb_documents" not in st.session_state or not st.session_state.kb_documents):
        st.write("Previously uploaded knowledge base documents:")
        if st.button("Load Existing Knowledge Base Documents"):
            with st.spinner("Loading existing knowledge base documents..."):
                try:
                    documents, error = ingestion_agent.ingest_directory_of_documents(kb_dir)
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
            "regulatory_documents" in st.session_state and st.session_state.regulatory_documents
    )

    if not ready_to_proceed:
        st.warning("Please upload at least the Regulatory Requirements file before proceeding.")

    # Show hints about optional files
    if ready_to_proceed and (
            "kyc_documents" not in st.session_state or not st.session_state.kyc_documents or
            "kb_documents" not in st.session_state or not st.session_state.kb_documents
    ):
        st.info("""
        Note: You have uploaded the regulatory requirements, which is the minimum needed to proceed.
        
        For better results, consider also uploading:
        - KYC Policy (B1 Mastersheet) for more accurate mapping
        - Knowledge Base Documents for better classification and context
        
        You can proceed now if these files were uploaded in a previous session.
        """)

    if ready_to_proceed:
        if st.button("Next: Review Documents"):
            st.session_state.current_step = 1
            st.rerun()
