# File: ui/vectorize_page.py

import streamlit as st
import os
from typing import List, Tuple, Optional
import logging
import time
from langchain.schema import Document

# Configure logging
logger = logging.getLogger('vectorize_page')

def build_vector_store(embedding_agent,
                       documents: List[Document],
                       store_name: str,
                       persist_directory: str,
                       progress_container=None) -> Tuple[bool, Optional[str]]:
    """
    Build a vector store from documents using the embedding agent.

    Args:
        embedding_agent: The embedding agent to use
        documents: List of documents to embed
        store_name: Name for the vector store
        persist_directory: Directory to persist the vector store
        progress_container: Streamlit container for progress updates

    Returns:
        Tuple of (success, error_message)
    """
    if not documents:
        return False, "No documents provided"

    try:
        # Log start
        logger.info(f"Building vector store '{store_name}' with {len(documents)} documents")
        if progress_container:
            progress_container.text(f"Creating vector store with {len(documents)} documents...")

        # Create vector store
        embedding_agent.create_vector_store(
            documents=documents,
            store_name=store_name,
            persist_directory=persist_directory
        )

        # Log success
        logger.info(f"Successfully built vector store '{store_name}'")
        if progress_container:
            progress_container.text(f"Successfully created vector store '{store_name}'!")

        return True, None

    except Exception as e:
        error_msg = f"Error building vector store: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if progress_container:
            progress_container.error(error_msg)

        return False, error_msg

def display_vectorize_page(embedding_agent, vector_dir: str):
    """Display the vector index building interface"""
    st.header("Step 3: Build Vector Indices")

    st.write("""
    In this step, we'll create vector embeddings for all the documents. 
    This allows the system to find similar documents efficiently during the mapping process.
    """)

    # Define vector store names - use constants that match what's used in classification page
    REG_VECTOR_STORE = "regulatory_requirements"
    KYC_VECTOR_STORE = "kyc_policy"
    KB_VECTOR_STORE = "knowledge_base"
    COMBINED_VECTOR_STORE = "combined_store"

    # Store these in session state to ensure consistency across pages
    st.session_state.reg_vector_store = REG_VECTOR_STORE
    st.session_state.kyc_vector_store = KYC_VECTOR_STORE
    st.session_state.kb_vector_store = KB_VECTOR_STORE
    st.session_state.combined_vector_store = COMBINED_VECTOR_STORE

    # Check if vector stores already exist
    try:
        vector_stores_exist = all([
            os.path.exists(os.path.join(vector_dir, store))
            for store in [REG_VECTOR_STORE, KYC_VECTOR_STORE, KB_VECTOR_STORE]
        ])
    except:
        vector_stores_exist = False

    # Display vector store options
    if vector_stores_exist:
        st.info("Vector stores already exist. You can use the existing ones or rebuild them.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Use Existing Vector Stores"):
                with st.spinner("Loading existing vector stores..."):
                    try:
                        # Load existing vector stores
                        reg_result = embedding_agent.load_vector_store(REG_VECTOR_STORE, vector_dir)
                        kyc_result = embedding_agent.load_vector_store(KYC_VECTOR_STORE, vector_dir)
                        kb_result = embedding_agent.load_vector_store(KB_VECTOR_STORE, vector_dir)

                        # Verify they were loaded correctly
                        if not all([reg_result, kyc_result, kb_result]):
                            missing = []
                            if not reg_result: missing.append(REG_VECTOR_STORE)
                            if not kyc_result: missing.append(KYC_VECTOR_STORE)
                            if not kb_result: missing.append(KB_VECTOR_STORE)
                            st.error(f"Failed to load some vector stores: {', '.join(missing)}")
                            st.stop()

                        # Check if combined store exists
                        if os.path.exists(os.path.join(vector_dir, COMBINED_VECTOR_STORE)):
                            embedding_agent.load_vector_store(COMBINED_VECTOR_STORE, vector_dir)

                        st.session_state.vector_stores_built = True

                        # Display available store names for debugging
                        available_stores = embedding_agent.list_vector_stores()
                        st.success(f"Successfully loaded vector stores: {', '.join(available_stores)}")

                        # Add delay to ensure user sees the success message
                        time.sleep(2)

                        # Move to next step
                        st.session_state.current_step = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading vector stores: {str(e)}")

        with col2:
            rebuild_button = st.button("Rebuild Vector Stores")
    else:
        # If no existing stores, just show build button
        rebuild_button = st.button("Build Vector Stores")

    # Build or rebuild vector stores
    if 'rebuild_button' in locals() and rebuild_button:
        with st.spinner("Building vector stores..."):
            # Create containers for progress updates
            reg_container = st.empty()
            kyc_container = st.empty()
            kb_container = st.empty()
            combined_container = st.empty()

            # Build regulatory vector store
            reg_container.info("Creating vector store for regulatory requirements...")
            success, error = build_vector_store(
                embedding_agent,
                st.session_state.regulatory_documents,
                REG_VECTOR_STORE,
                vector_dir,
                reg_container
            )

            if not success:
                st.error(f"Failed to build regulatory vector store: {error}")
                st.stop()

            # Build KYC policy vector store
            kyc_container.info("Creating vector store for KYC policy...")
            success, error = build_vector_store(
                embedding_agent,
                st.session_state.kyc_documents,
                KYC_VECTOR_STORE,
                vector_dir,
                kyc_container
            )

            if not success:
                st.error(f"Failed to build KYC policy vector store: {error}")
                st.stop()

            # Build knowledge base vector store
            kb_container.info("Creating vector store for knowledge base...")
            success, error = build_vector_store(
                embedding_agent,
                st.session_state.kb_documents,
                KB_VECTOR_STORE,
                vector_dir,
                kb_container
            )

            if not success:
                st.error(f"Failed to build knowledge base vector store: {error}")
                st.stop()

            # Create combined vector store
            combined_container.info("Creating combined vector store...")
            try:
                embedding_agent.combine_vector_stores(
                    target_name=COMBINED_VECTOR_STORE,
                    source_names=[KYC_VECTOR_STORE, KB_VECTOR_STORE],
                    persist_directory=vector_dir
                )
                combined_container.success("Successfully created combined vector store!")
            except Exception as e:
                combined_container.error(f"Error creating combined vector store: {str(e)}")
                # Not critical, so continue

            # Display success message
            st.session_state.vector_stores_built = True
            st.success("All vector stores built successfully!")

            # Add delay to ensure user sees the success message
            time.sleep(2)

            # Move to next step
            st.session_state.current_step = 3
            st.rerun()

    # Navigation
    st.markdown("---")
    if st.button("⬅️ Back to Document Review"):
        st.session_state.current_step = 1
        st.rerun()
