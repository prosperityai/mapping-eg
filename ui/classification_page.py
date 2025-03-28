# File: ui/classification_page.py (Simplified for Classification Only)

import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any
import logging
from langchain.schema import Document

# Configure logging
logger = logging.getLogger('classification_page')

def get_relevant_kb_docs(embedding_agent,
                         requirement_text: str,
                         num_docs: int = 3) -> List[Document]:
    """
    Retrieve relevant knowledge base documents for a requirement.

    Args:
        embedding_agent: The embedding agent to use for retrieval
        requirement_text: The text of the requirement
        num_docs: Number of documents to retrieve

    Returns:
        List of relevant documents
    """
    try:
        # Get the KB vector store name from session state for consistency
        kb_store_name = st.session_state.get("kb_vector_store", "knowledge_base")

        # Get the KB vector store
        kb_store = embedding_agent.get_vector_store(kb_store_name)
        if not kb_store:
            logger.warning(f"Knowledge base vector store '{kb_store_name}' not found")
            st.warning(f"Knowledge base vector store '{kb_store_name}' not found. Please go back and build the vector indices.")
            return []

        # Retrieve similar documents with scores
        similar_docs = kb_store.similarity_search_with_score(
            requirement_text,
            k=num_docs
        )

        # Add similarity score to metadata
        documents = []
        for doc, score in similar_docs:
            doc.metadata["similarity_score"] = score
            documents.append(doc)

        return documents
    except Exception as e:
        logger.error(f"Error retrieving KB documents: {str(e)}", exc_info=True)
        return []

def classify_requirement(classification_agent,
                         requirement: Document,
                         kb_docs: List[Document]) -> Dict[str, Any]:
    """
    Classify a regulatory requirement using the classification agent.

    Args:
        classification_agent: The classification agent to use
        requirement: The requirement document
        kb_docs: Knowledge base documents for context

    Returns:
        Classification result
    """
    try:
        # Call the classification agent
        result = classification_agent.classify_requirement(requirement, kb_docs)
        return result
    except Exception as e:
        logger.error(f"Error classifying requirement: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "type": "Unknown",
            "confidence": 0.0,
            "explanation": f"Error during classification: {str(e)}"
        }

def batch_classify_requirements(classification_agent,
                                embedding_agent,
                                requirements: List[Document],
                                progress_placeholder=None) -> List[Dict[str, Any]]:
    """
    Classify multiple requirements in batch.

    Args:
        classification_agent: The classification agent to use
        embedding_agent: The embedding agent for KB retrieval
        requirements: List of requirement documents
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        List of classification results
    """
    results = []
    total = len(requirements)

    # Define the function to get KB docs for batch classification
    def get_kb_docs(requirement):
        if isinstance(requirement, Document):
            return get_relevant_kb_docs(embedding_agent, requirement.page_content)
        else:
            return get_relevant_kb_docs(embedding_agent, requirement)

    # Update progress callback
    def update_progress(progress):
        if progress_placeholder:
            progress_bar = progress_placeholder.progress(0.0)
            progress_bar.progress(progress)

    # Use batch classification
    results = classification_agent.batch_classify(
        requirements=requirements,
        get_kb_docs_func=get_kb_docs,
        batch_size=5,
        progress_callback=update_progress
    )

    return results

def display_classification_page(embedding_agent, classification_agent):
    """Display the classification interface"""
    # Debug log to ensure this function is being called
    logger.info("Classification page is being displayed")

    st.header("Step 4: Requirement Classification")

    # Check if vector stores have been built
    if not st.session_state.vector_stores_built:
        st.error("Vector stores have not been built yet. Please go back and build them first.")
        if st.button("Back to Vector Store Creation"):
            st.session_state.current_step = 2
            st.rerun()
        st.stop()

    # Load and verify vector stores
    available_stores = embedding_agent.list_vector_stores()

    reg_store_name = st.session_state.get("reg_vector_store", "regulatory_requirements")
    kyc_store_name = st.session_state.get("kyc_vector_store", "kyc_policy")
    kb_store_name = st.session_state.get("kb_vector_store", "knowledge_base")

    # Verify required stores exist
    missing_stores = []
    for store_name in [reg_store_name, kyc_store_name, kb_store_name]:
        if store_name not in available_stores:
            missing_stores.append(store_name)

    if missing_stores:
        st.error(f"Missing required vector stores: {', '.join(missing_stores)}")
        st.write("Please go back and rebuild the vector stores.")
        if st.button("Rebuild Vector Stores"):
            st.session_state.current_step = 2
            st.rerun()
        st.stop()

    st.write("""
    In this step, we'll classify each regulatory requirement as either:
    - **Customer Due Diligence (CDD)**: Related to customer identification, verification, and monitoring
    - **Program-Level**: Related to overall compliance program framework, governance, and policies
    
    This classification will help determine how each requirement should be mapped to policies in the next step.
    """)

    # Initialize the retrieval and classification state if needed
    if "classified_requirements" not in st.session_state:
        st.session_state.classified_requirements = []

    if "current_requirement_idx" not in st.session_state:
        st.session_state.current_requirement_idx = 0

    if "classifications_completed" not in st.session_state:
        st.session_state.classifications_completed = False

    if "classification_results" not in st.session_state:
        st.session_state.classification_results = []

    # Total number of requirements
    total_requirements = len(st.session_state.regulatory_documents)

    # Classification process - first batch classify if not already done
    if not st.session_state.classifications_completed and not st.session_state.classification_results:
        st.subheader("Classification Method")

        # Provide option to batch classify or do one by one
        classify_option = st.radio(
            "How would you like to classify requirements?",
            ["Classify All At Once", "Classify One By One"]
        )

        if classify_option == "Classify All At Once":
            if st.button("Begin Batch Classification"):
                # Add a progress section
                st.write("### Classification Progress")
                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0.0)
                status_placeholder = st.empty()
                status_placeholder.info("Classifying requirements...")

                # Run batch classification
                with st.spinner("Classifying all requirements..."):
                    try:
                        classification_results = batch_classify_requirements(
                            classification_agent=classification_agent,
                            embedding_agent=embedding_agent,
                            requirements=st.session_state.regulatory_documents,
                            progress_placeholder=progress_placeholder
                        )

                        # Store classification results in session state
                        st.session_state.classification_results = classification_results
                        st.session_state.classifications_completed = True

                        # Update classified requirements with classification results
                        for i, result in enumerate(classification_results):
                            req_doc = st.session_state.regulatory_documents[i]

                            # Get relevant KB docs
                            kb_docs = get_relevant_kb_docs(
                                embedding_agent=embedding_agent,
                                requirement_text=req_doc.page_content,
                                num_docs=3
                            )

                            # Create classified requirement entry (simplified)
                            classified_req = {
                                "requirement": {
                                    "text": req_doc.page_content,
                                    "metadata": req_doc.metadata
                                },
                                "similar_kb_docs": [
                                    {
                                        "text": doc.page_content,
                                        "metadata": doc.metadata
                                    } for doc in kb_docs
                                ],
                                "classification": {
                                    "type": result.get("type", "Unknown"),
                                    "confidence": result.get("confidence", 0.0),
                                    "explanation": result.get("explanation", "")
                                },
                                "manual_classification": False,
                                "processed_at": time.time()
                            }

                            # Add to classified requirements
                            st.session_state.classified_requirements.append(classified_req)

                        status_placeholder.success(f"Successfully classified all {total_requirements} requirements!")
                        time.sleep(1)  # Give user a moment to see success message
                        st.rerun()  # Refresh to show results

                    except Exception as e:
                        status_placeholder.error(f"Error during batch classification: {str(e)}")
                        logger.error(f"Batch classification error: {str(e)}", exc_info=True)
        else:
            # Handle one-by-one case
            st.session_state.classifications_completed = False
            st.info("You've chosen to classify requirements one by one. Let's begin!")
            if not st.session_state.classified_requirements:
                # Reset to start the process
                st.session_state.current_requirement_idx = 0
            time.sleep(1)
            st.rerun()

    # If classifications are completed, show summary and move to mapping
    elif st.session_state.classifications_completed and st.session_state.classification_results:
        st.subheader("Classification Results")

        # Create a DataFrame for classification results
        results_data = []
        for i, result in enumerate(st.session_state.classification_results):
            req_doc = st.session_state.regulatory_documents[i]
            req_text = req_doc.page_content[:100] + "..." if len(req_doc.page_content) > 100 else req_doc.page_content

            results_data.append({
                "Index": i + 1,
                "Requirement": req_text,
                "Type": result.get("type", "Unknown"),
                "Confidence": f"{result.get('confidence', 0.0):.2f}",
                "Needs Review": "Yes" if result.get("confidence", 0.0) < 0.7 else "No"
            })

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

        # Show summary statistics
        st.write("### Classification Summary")
        classification_types = [r.get("type", "Unknown") for r in st.session_state.classification_results]
        type_counts = {}
        for t in classification_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Classification by Type:**")
            for t, count in type_counts.items():
                st.write(f"- {t}: {count} ({count/total_requirements*100:.1f}%)")

        with col2:
            # Calculate average confidence
            confidences = [r.get("confidence", 0.0) for r in st.session_state.classification_results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            st.write("**Confidence Metrics:**")
            st.write(f"- Average confidence: {avg_confidence:.2f}")
            low_confidence = sum(1 for c in confidences if c < 0.7)
            st.write(f"- Requirements needing review: {low_confidence} ({low_confidence/total_requirements*100:.1f}%)")

        # Options to proceed or review
        st.markdown("---")
        st.write("### Next Steps")
        st.write("You can now proceed to mapping or review and edit the classifications.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Proceed to Mapping"):
                # This will eventually be updated to go to mapping page
                st.session_state.current_step = 4
                st.rerun()

        with col2:
            if st.button("Review Classifications"):
                # Reset for individual review
                st.session_state.classifications_completed = False
                st.session_state.current_requirement_idx = 0
                st.rerun()

    # Individual requirement classification handling
    else:
        current_idx = st.session_state.current_requirement_idx

        # If all requirements have been processed, show summary
        if current_idx >= total_requirements:
            st.success(f"All {total_requirements} requirements have been classified!")

            # Show summary statistics
            classifications = [req.get("classification", {}) for req in st.session_state.classified_requirements]

            # Count by type (CDD or Program-Level)
            types = [c.get("type", "Unknown") for c in classifications]
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1

            # Display stats
            st.write("**Classification by Type:**")
            for t, count in type_counts.items():
                st.write(f"- {t}: {count} ({count/total_requirements*100:.1f}%)")

            # Show proceed button
            if st.button("Proceed to Mapping"):
                st.session_state.current_step = 4
                st.rerun()

            st.stop()

        # Get the current requirement to process
        current_req_doc = st.session_state.regulatory_documents[current_idx]
        current_req_text = current_req_doc.page_content

        # Show the current requirement
        st.subheader(f"Classifying Requirement {current_idx+1}/{total_requirements}")
        st.write("**Requirement Text:**")
        st.info(current_req_text)

        # Show metadata if available
        if st.checkbox("Show Requirement Metadata"):
            meta_df = pd.DataFrame([{"Key": k, "Value": str(v)} for k, v in current_req_doc.metadata.items()])
            st.dataframe(meta_df, use_container_width=True)

        # Retrieve relevant knowledge base documents
        kb_docs = get_relevant_kb_docs(
            embedding_agent=embedding_agent,
            requirement_text=current_req_text,
            num_docs=3
        )

        # Show relevant KB documents
        st.write("**Relevant Knowledge Base Documents:**")
        for i, doc in enumerate(kb_docs):
            with st.expander(f"KB Document {i+1}: {doc.metadata.get('file_name', 'Unknown')} - Page {doc.metadata.get('page', 'Unknown')}"):
                st.write("**Content:**")
                st.write(doc.page_content)

                st.write("**Source:**")
                st.write(f"File: {doc.metadata.get('file_name', 'Unknown')}")
                st.write(f"Page: {doc.metadata.get('page', 'Unknown')}")
                if "similarity_score" in doc.metadata:
                    st.write(f"Similarity: {doc.metadata['similarity_score']:.2f}")

        # Classification form
        st.write("**Classification:**")

        # Check if we already have pre-classified results for this requirement
        initial_type = ""
        initial_explanation = ""
        initial_confidence = 0.0

        if st.session_state.classification_results and current_idx < len(st.session_state.classification_results):
            result = st.session_state.classification_results[current_idx]
            initial_type = result.get("type", "")
            initial_explanation = result.get("explanation", "")
            initial_confidence = result.get("confidence", 0.0)

            # Show the LLM's assessment
            st.write("### LLM Classification")
            status_color = "green" if initial_confidence >= 0.8 else "orange" if initial_confidence >= 0.6 else "red"
            st.markdown(f"""
            **Type:** {initial_type}  
            **Confidence:** <span style='color:{status_color};'>{initial_confidence:.2f}</span>  
            **Explanation:** {initial_explanation}
            """, unsafe_allow_html=True)

        # Initialize classification options if not already in session
        if f"classification_type_{current_idx}" not in st.session_state:
            st.session_state[f"classification_type_{current_idx}"] = initial_type or "CDD"
        if f"explanation_{current_idx}" not in st.session_state:
            st.session_state[f"explanation_{current_idx}"] = initial_explanation or ""

        # Form for user classification (simplified - only asking for type)
        with st.form(key=f"classification_form_{current_idx}"):
            # Type classification
            classification_type = st.radio(
                "Requirement Type:",
                options=["CDD", "Program-Level"],
                key=f"classification_type_{current_idx}"
            )

            # Explanation
            explanation = st.text_area(
                "Explanation/Notes:",
                key=f"explanation_{current_idx}"
            )

            # Submit button
            submit_classification = st.form_submit_button("Submit Classification")

        # If classification was submitted
        if submit_classification:
            # Create classification object (simplified)
            classification = {
                "type": classification_type,
                "explanation": explanation,
                "confidence": 1.0,  # Manual classification so confidence is 1.0
            }

            # Create classified requirement (simplified)
            classified_req = {
                "requirement": {
                    "text": current_req_text,
                    "metadata": current_req_doc.metadata
                },
                "similar_kb_docs": [
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in kb_docs
                ],
                "classification": classification,
                "manual_classification": True,
                "processed_at": time.time()
            }

            # Add to classified requirements
            if current_idx < len(st.session_state.classified_requirements):
                # Update existing requirement
                st.session_state.classified_requirements[current_idx] = classified_req
            else:
                # Add new requirement
                st.session_state.classified_requirements.append(classified_req)

            # Move to next requirement
            st.session_state.current_requirement_idx += 1
            st.rerun()

        # Option to auto-classify this requirement
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            # Use the classification agent to classify this requirement
            if st.button("Classify Using LLM"):
                with st.spinner("Classifying requirement..."):
                    try:
                        # Call classification agent
                        result = classify_requirement(
                            classification_agent=classification_agent,
                            requirement=current_req_doc,
                            kb_docs=kb_docs
                        )

                        # Store results in new session state variables that don't conflict with the form
                        llm_type = result.get("type", "Program-Level")
                        llm_explanation = result.get("explanation", "")
                        llm_confidence = result.get("confidence", 0.0)

                        st.success("Classification complete! The LLM suggests:")

                        # Display the results
                        st.write(f"**Type:** {llm_type}")
                        st.write(f"**Confidence:** {llm_confidence:.2f}")
                        st.write(f"**Explanation:** {llm_explanation}")

                        # Provide a button to use these suggestions
                        if st.button("Use This Classification"):
                            # Create a new classified requirement with the LLM suggestions
                            classification = {
                                "type": llm_type,
                                "explanation": llm_explanation,
                                "confidence": llm_confidence,
                            }

                            classified_req = {
                                "requirement": {
                                    "text": current_req_text,
                                    "metadata": current_req_doc.metadata
                                },
                                "similar_kb_docs": [
                                    {
                                        "text": doc.page_content,
                                        "metadata": doc.metadata
                                    } for doc in kb_docs
                                ],
                                "classification": classification,
                                "manual_classification": False,
                                "processed_at": time.time()
                            }

                            # Add to classified requirements
                            if current_idx < len(st.session_state.classified_requirements):
                                st.session_state.classified_requirements[current_idx] = classified_req
                            else:
                                st.session_state.classified_requirements.append(classified_req)

                            # Move to next requirement
                            st.session_state.current_requirement_idx += 1
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error classifying requirement: {str(e)}")
                        logger.error(f"Classification error: {str(e)}", exc_info=True)

        with col2:
            # Skip button - move to next without classifying
            if st.button("Skip to Next Requirement"):
                st.session_state.current_requirement_idx += 1
                st.rerun()

        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅️ Back to Vector Stores"):
                st.session_state.current_step = 2
                st.rerun()

        with col2:
            if st.button("Skip to Mapping"):
                st.session_state.current_step = 4
                st.rerun()
