# File: ui/mapping_page.py

import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any
import logging
from langchain.schema import Document

# Configure logging
logger = logging.getLogger('mapping_page')

def get_relevant_kyc_policies(embedding_agent,
                              requirement_text: str,
                              num_docs: int = 5) -> List[Document]:
    """
    Retrieve relevant KYC policy documents for a requirement.

    Args:
        embedding_agent: The embedding agent to use for retrieval
        requirement_text: The text of the requirement
        num_docs: Number of documents to retrieve

    Returns:
        List of relevant documents
    """
    try:
        # Get the KYC vector store name from session state for consistency
        kyc_store_name = st.session_state.get("kyc_vector_store", "kyc_policy")

        # Get the KYC vector store
        kyc_store = embedding_agent.get_vector_store(kyc_store_name)
        if not kyc_store:
            logger.warning(f"KYC policy vector store '{kyc_store_name}' not found")
            st.warning(f"KYC policy vector store '{kyc_store_name}' not found. Please go back and build the vector indices.")
            return []

        # Retrieve similar documents with scores
        similar_docs = kyc_store.similarity_search_with_score(
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
        logger.error(f"Error retrieving KYC policies: {str(e)}", exc_info=True)
        return []

def map_requirement(mapping_agent,
                   requirement: Dict[str, Any],
                   kyc_policies: List[Document]) -> Dict[str, Any]:
    """
    Map a requirement to KYC policies using the mapping agent.

    Args:
        mapping_agent: The mapping agent to use
        requirement: Classified requirement dict
        kyc_policies: KYC policy documents

    Returns:
        Mapping result
    """
    try:
        # Call the mapping agent
        result = mapping_agent.map_requirement(requirement, kyc_policies)
        return result
    except Exception as e:
        logger.error(f"Error mapping requirement: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "coverage": "Unknown",
            "mapped_policies": [],
            "coverage_explanation": f"Error during mapping: {str(e)}",
            "gap_analysis": "",
            "confidence": 0.0
        }

def batch_map_requirements(mapping_agent,
                          embedding_agent,
                          classified_requirements: List[Dict[str, Any]],
                          progress_placeholder=None) -> List[Dict[str, Any]]:
    """
    Map multiple classified requirements in batch.

    Args:
        mapping_agent: The mapping agent to use
        embedding_agent: The embedding agent for KYC policy retrieval
        classified_requirements: List of classified requirements
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        List of mapping results
    """
    # Define the function to get KYC policy docs
    def get_kyc_docs(requirement):
        req_text = requirement.get("requirement", {}).get("text", "")
        return get_relevant_kyc_policies(embedding_agent, req_text)

    # Update progress callback
    def update_progress(progress):
        if progress_placeholder:
            progress_bar = progress_placeholder.progress(0.0)
            progress_bar.progress(progress)

    # Use batch mapping
    results = mapping_agent.batch_map(
        classified_requirements=classified_requirements,
        get_kyc_docs_func=get_kyc_docs,
        batch_size=3,  # Smaller batch size for more complex mapping
        progress_callback=update_progress
    )

    return results

def display_mapped_policies(mapped_policies):
    """Display mapped policies in a structured way"""
    if not mapped_policies:
        st.write("No policies mapped.")
        return

    for i, policy in enumerate(mapped_policies):
        # Determine color based on relevance score
        if policy.get("relevance_score", 0) > 0.8:
            color = "green"
        elif policy.get("relevance_score", 0) > 0.5:
            color = "orange"
        else:
            color = "red"

        with st.expander(f"Policy {i+1}: {policy.get('master_sheet', '')}-{policy.get('minor_sheet', '')}: {policy.get('title', '')}"):
            st.markdown(f"""
            **Master Sheet:** {policy.get('master_sheet', 'N/A')}
            **Minor Sheet:** {policy.get('minor_sheet', 'N/A')}
            **Section:** {policy.get('section', 'N/A')}
            **Title:** {policy.get('title', 'N/A')}
            **Relevance Score:** <span style='color:{color};'>{policy.get('relevance_score', 0):.2f}</span>

            **Explanation:**
            {policy.get('explanation', 'No explanation provided.')}
            """, unsafe_allow_html=True)

def display_mapping_page(embedding_agent, mapping_agent):
    """Display the mapping interface"""
    # Debug log to ensure this function is being called
    logger.info("Mapping page is being displayed")

    st.header("Step 5: Map Requirements to Policies")

    # Verify that we have classified requirements to work with
    if "classified_requirements" not in st.session_state or not st.session_state.classified_requirements:
        st.error("No classified requirements found. Please go back and classify the requirements first.")
        if st.button("Back to Classification"):
            st.session_state.current_step = 3
            st.rerun()
        st.stop()

    # Initialize session state for mapping results if needed
    if "mapped_requirements" not in st.session_state:
        st.session_state.mapped_requirements = []

    if "mappings_completed" not in st.session_state:
        st.session_state.mappings_completed = False

    if "current_mapping_idx" not in st.session_state:
        st.session_state.current_mapping_idx = 0

    # Total number of requirements
    total_requirements = len(st.session_state.classified_requirements)

    st.write(f"""
    In this step, we'll map each classified regulatory requirement to relevant KYC policies.
    This will help determine:

    - Whether existing policies provide **Equivalent** coverage
    - Where **Partial Uplift** is needed
    - Where **Full Uplift** is required to address gaps

    The LLM will carefully analyze the semantic relationship between requirements and policies,
    identifying specific B1 Mastersheet references that are relevant.
    """)

    # Provide choice between batch mapping and individual mapping
    if not st.session_state.mappings_completed and not st.session_state.mapped_requirements:
        st.subheader("Mapping Method")

        mapping_option = st.radio(
            "How would you like to map requirements to policies?",
            ["Map All At Once", "Map One By One"]
        )

        if mapping_option == "Map All At Once":
            if st.button("Begin Batch Mapping"):
                # Add a progress section
                st.write("### Mapping Progress")
                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0.0)
                status_placeholder = st.empty()
                status_placeholder.info("Mapping requirements to policies...")

                # Run batch mapping
                with st.spinner("Mapping all requirements..."):
                    try:
                        mapping_results = batch_map_requirements(
                            mapping_agent=mapping_agent,
                            embedding_agent=embedding_agent,
                            classified_requirements=st.session_state.classified_requirements,
                            progress_placeholder=progress_placeholder
                        )

                        # Store mapping results
                        st.session_state.mapped_requirements = mapping_results
                        st.session_state.mappings_completed = True

                        status_placeholder.success(f"Successfully mapped all {total_requirements} requirements!")
                        time.sleep(1)  # Give user a moment to see success message
                        st.rerun()  # Refresh to show results

                    except Exception as e:
                        status_placeholder.error(f"Error during batch mapping: {str(e)}")
                        logger.error(f"Batch mapping error: {str(e)}", exc_info=True)
        else:
            # Handle one-by-one case
            st.session_state.mappings_completed = False
            st.info("You've chosen to map requirements one by one. Let's begin!")
            if not st.session_state.mapped_requirements:
                # Reset to start the process
                st.session_state.current_mapping_idx = 0
            time.sleep(1)
            st.rerun()

    # If mappings are completed, show summary
    elif st.session_state.mappings_completed and st.session_state.mapped_requirements:
        st.subheader("Mapping Results")

        # Create a DataFrame for mapping results
        results_data = []
        for i, result in enumerate(st.session_state.mapped_requirements):
            req = result.get("requirement", {})
            req_text = req.get("text", "")[:100] + "..." if len(req.get("text", "")) > 100 else req.get("text", "")

            # Get mapped policies
            mapped_policies = result.get("mapped_policies", [])
            num_policies = len(mapped_policies)

            # Extract Master Sheet and Minor Sheet information
            master_sheets = []
            minor_sheets = []

            for policy in mapped_policies:
                master_sheet = policy.get('master_sheet', '')
                minor_sheet = policy.get('minor_sheet', '')

                if master_sheet and master_sheet not in master_sheets:
                    master_sheets.append(master_sheet)
                if minor_sheet and minor_sheet not in minor_sheets:
                    minor_sheets.append(minor_sheet)

            results_data.append({
                "Index": i + 1,
                "Requirement": req_text,
                "Type": req.get("type", "Unknown"),
                "Coverage": result.get("coverage", "Unknown"),
                "Master Sheets": ", ".join(master_sheets),
                "Minor Sheets": ", ".join(minor_sheets),
                "Mapped Policies": num_policies,
                "Confidence": f"{result.get('confidence', 0.0):.2f}"
            })

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        # Show summary statistics
        st.write("### Mapping Summary")
        coverages = [r.get("coverage", "Unknown") for r in st.session_state.mapped_requirements]
        coverage_counts = {}
        for c in coverages:
            coverage_counts[c] = coverage_counts.get(c, 0) + 1

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Coverage Distribution:**")
            for c, count in coverage_counts.items():
                if c == "Equivalent":
                    color = "green"
                elif c == "Partial Uplift":
                    color = "orange"
                elif c == "Full Uplift":
                    color = "red"
                else:
                    color = "gray"
                st.markdown(f"- <span style='color:{color};'>{c}</span>: {count} ({count/total_requirements*100:.1f}%)", unsafe_allow_html=True)

        with col2:
            # Calculate average confidence
            confidences = [r.get("confidence", 0.0) for r in st.session_state.mapped_requirements]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            st.write("**Confidence Metrics:**")
            st.write(f"- Average confidence: {avg_confidence:.2f}")
            low_confidence = sum(1 for c in confidences if c < 0.7)
            st.write(f"- Mappings needing review: {low_confidence} ({low_confidence/total_requirements*100:.1f}%)")

        # Options to proceed or review
        st.markdown("---")
        st.write("### Next Steps")
        st.write("You can now proceed to export the final results or review the mappings.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Proceed to Export"):
                st.session_state.current_step = 5
                st.rerun()

        with col2:
            if st.button("Review Mappings"):
                # Reset for individual review
                st.session_state.mappings_completed = False
                st.session_state.current_mapping_idx = 0
                st.rerun()

    # Individual mapping handling
    else:
        current_idx = st.session_state.current_mapping_idx

        # If all requirements have been mapped, show summary
        if current_idx >= total_requirements:
            st.success(f"All {total_requirements} requirements have been mapped!")

            # Show summary statistics
            coverages = [req.get("coverage", "Unknown") for req in st.session_state.mapped_requirements]
            coverage_counts = {}
            for c in coverages:
                coverage_counts[c] = coverage_counts.get(c, 0) + 1

            # Display stats
            st.write("**Coverage Distribution:**")
            for c, count in coverage_counts.items():
                st.write(f"- {c}: {count} ({count/total_requirements*100:.1f}%)")

            # Show proceed button
            if st.button("Proceed to Export"):
                st.session_state.current_step = 5
                st.rerun()

            st.stop()

        # Get the current requirement to map
        current_req = st.session_state.classified_requirements[current_idx]
        current_req_text = current_req.get("requirement", {}).get("text", "")
        current_req_type = current_req.get("classification", {}).get("type", "Unknown")
        current_req_explanation = current_req.get("classification", {}).get("explanation", "")

        # Show the current requirement
        st.subheader(f"Mapping Requirement {current_idx+1}/{total_requirements}")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Requirement Text:**")
            st.info(current_req_text)

        with col2:
            st.write("**Classification:**")
            st.success(f"Type: {current_req_type}")
            with st.expander("Classification Explanation"):
                st.write(current_req_explanation)

        # Retrieve relevant KYC policies
        kyc_policies = get_relevant_kyc_policies(
            embedding_agent=embedding_agent,
            requirement_text=current_req_text,
            num_docs=5
        )

        # Show relevant KYC policies
        st.write("**Relevant KYC Policies:**")
        for i, doc in enumerate(kyc_policies):
            with st.expander(f"Policy {i+1}: {doc.metadata.get('master_sheet', '')}-{doc.metadata.get('minor_sheet', '')}: {doc.metadata.get('title', 'No Title')}"):
                st.write("**Content:**")
                st.write(doc.page_content)

                st.write("**Metadata:**")
                st.write(f"Master Sheet: {doc.metadata.get('master_sheet', 'N/A')}")
                st.write(f"Minor Sheet: {doc.metadata.get('minor_sheet', 'N/A')}")
                st.write(f"Section: {doc.metadata.get('section', 'N/A')}")
                st.write(f"Title: {doc.metadata.get('title', 'N/A')}")
                if "similarity_score" in doc.metadata:
                    st.write(f"Similarity: {doc.metadata['similarity_score']:.2f}")

        # Check if we already have a mapping result for this requirement
        existing_mapping = None
        if len(st.session_state.mapped_requirements) > current_idx:
            existing_mapping = st.session_state.mapped_requirements[current_idx]

        # If we already have a mapping result, show it
        if existing_mapping:
            st.subheader("Current Mapping")

            # Determine coverage color
            coverage = existing_mapping.get("coverage", "Unknown")
            if coverage == "Equivalent":
                coverage_color = "green"
            elif coverage == "Partial Uplift":
                coverage_color = "orange"
            elif coverage == "Full Uplift":
                coverage_color = "red"
            else:
                coverage_color = "gray"

            st.markdown(f"""
            **Coverage:** <span style='color:{coverage_color};'>{coverage}</span>
            **Confidence:** {existing_mapping.get('confidence', 0.0):.2f}

            **Coverage Explanation:**
            {existing_mapping.get('coverage_explanation', 'No explanation provided.')}

            **Gap Analysis:**
            {existing_mapping.get('gap_analysis', 'No gap analysis provided.')}
            """, unsafe_allow_html=True)

            st.write("**Mapped Policies:**")
            display_mapped_policies(existing_mapping.get("mapped_policies", []))

        # Button to re-map this requirement
        if st.button("Re-Map This Requirement"):
            with st.spinner("Mapping requirement..."):
                try:
                    # Call mapping agent
                    mapping_result = map_requirement(
                        mapping_agent=mapping_agent,
                        requirement=current_req,
                        kyc_policies=kyc_policies
                    )

                    # Update mapped requirements
                    if current_idx < len(st.session_state.mapped_requirements):
                        st.session_state.mapped_requirements[current_idx] = mapping_result
                    else:
                        st.session_state.mapped_requirements.append(mapping_result)

                    st.success("Mapping complete!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error mapping requirement: {str(e)}")
                    logger.error(f"Mapping error: {str(e)}", exc_info=True)

        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⬅️ Previous Requirement"):
                if current_idx > 0:
                    st.session_state.current_mapping_idx -= 1
                    st.rerun()
                else:
                    st.warning("Already at the first requirement.")

        with col2:
            if current_idx < total_requirements - 1:
                if st.button("Skip to Next ➡️"):
                    st.session_state.current_mapping_idx += 1
                    st.rerun()
            else:
                if st.button("Complete Mapping ✅"):
                    st.session_state.mappings_completed = True
                    st.rerun()

        with col3:
            if st.button("Back to Classification"):
                st.session_state.current_step = 3
                st.rerun()

        # If there's no existing mapping, automatically run the mapping agent
        if not existing_mapping:
            with st.spinner("Analyzing requirement and mapping to policies..."):
                st.write("### The LLM is mapping this requirement to policies")

                # Show thinking process to emphasize the LLM's reasoning
                thinking_container = st.empty()
                thinking_container.info("""
                **Thinking process:**
                
                1. Analyzing the specific regulatory requirement in detail
                2. Understanding the key obligations and compliance elements
                3. Reviewing each KYC policy to assess relevance
                4. Evaluating how completely the policies address the requirement
                5. Identifying specific gaps between requirement and policies
                6. Determining the overall coverage level
                7. Calculating confidence in the mapping assessment
                """)

                # Give the user a sense that the model is working
                time.sleep(2)

                try:
                    # Call mapping agent
                    mapping_result = map_requirement(
                        mapping_agent=mapping_agent,
                        requirement=current_req,
                        kyc_policies=kyc_policies
                    )

                    # Update mapped requirements
                    if current_idx < len(st.session_state.mapped_requirements):
                        st.session_state.mapped_requirements[current_idx] = mapping_result
                    else:
                        st.session_state.mapped_requirements.append(mapping_result)

                    thinking_container.success("Mapping analysis complete!")
                    st.rerun()

                except Exception as e:
                    thinking_container.error(f"Error mapping requirement: {str(e)}")
                    logger.error(f"Mapping error: {str(e)}", exc_info=True)
