# File: ui/review_page.py

import streamlit as st
import pandas as pd
from typing import List
import logging
from langchain.schema import Document

# Configure logging
logger = logging.getLogger('review_page')

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

def display_review_page():
    """Display the review interface"""
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
            st.rerun()

    with col2:
        if st.button("Next: Build Vector Index ➡️"):
            st.session_state.current_step = 2
            st.rerun()
