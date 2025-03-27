# File: ui/export_page.py (Temporary Version)

import streamlit as st
import pandas as pd
import tempfile
import os
import logging

# Configure logging
logger = logging.getLogger('export_page')

def display_export_page():
    """Display a temporary export page until mapping functionality is implemented"""
    st.header("Step 5: Export Classifications (Interim)")

    if "classified_requirements" not in st.session_state or not st.session_state.classified_requirements:
        st.error("No classified requirements found. Please go back and classify the requirements first.")
        if st.button("Back to Classification"):
            st.session_state.current_step = 3
            st.rerun()
        st.stop()

    st.write("""
    ### Classification Complete
    
    This is a temporary export page. In the future, this will be replaced with a full mapping step 
    that will match each requirement to specific policies based on its classification.
    
    Currently, you can export the classification results to Excel for review.
    """)

    # Create a DataFrame with the classification results
    classification_rows = []

    for req in st.session_state.classified_requirements:
        # Extract requirement info
        req_text = req["requirement"]["text"]
        req_metadata = req["requirement"]["metadata"]
        classification = req["classification"]

        # Create a row with original data and classification information
        row = {}

        # Add all original metadata
        if "original_data" in req_metadata:
            row.update(req_metadata["original_data"])

        # Add requirement text
        row["Requirement Text"] = req_text

        # Add classification information
        row["Type"] = classification.get("type", "")
        row["Confidence"] = classification.get("confidence", 0.0)
        row["Explanation"] = classification.get("explanation", "")
        row["Classification Method"] = "Manual" if req.get("manual_classification", False) else "LLM"

        classification_rows.append(row)

    # Create DataFrame
    if classification_rows:
        classification_df = pd.DataFrame(classification_rows)

        # Remove any duplicate columns
        classification_df = classification_df.loc[:, ~classification_df.columns.duplicated()]

        # Display the classification results
        st.write("### Classification Results")
        st.dataframe(classification_df, use_container_width=True)

        # Export options
        st.write("### Export Options")

        # Create Excel file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            with pd.ExcelWriter(tmp_file.name, engine="openpyxl") as writer:
                classification_df.to_excel(writer, index=False, sheet_name="Classified Requirements")

            excel_data = open(tmp_file.name, "rb").read()

            # Download button
            st.download_button(
                label="📥 Download Classification Results (Excel)",
                data=excel_data,
                file_name="regulatory_classification_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_classification",
                help="Download the classification results as an Excel file",
                use_container_width=True
            )

            # Cleanup temp file
            os.unlink(tmp_file.name)
    else:
        st.warning("No classification results to display.")

    # Information about the next development step
    st.info("""
    🔍 **Coming Next: Mapping Agent**
    
    The next development step will be implementing a mapping agent that will:
    1. Take these classified requirements as input
    2. For each requirement, determine:
       - Coverage level (Equivalent, Partial Uplift, Full Uplift)
       - Specific policy references (Master Sheet, Minor Sheet)
       - Detailed explanations for matches or gaps
    3. Generate comprehensive mapping information for export
    """)

    # Navigation
    st.markdown("---")
    if st.button("⬅️ Back to Classification"):
        st.session_state.current_step = 3
        st.rerun()

    if st.button("Start New Project"):
        # Reset session state
        for key in list(st.session_state.keys()):
            # Keep only the embedding agent and classification agent to avoid re-initialization
            if key not in ["embedding_agent", "classification_agent"]:
                del st.session_state[key]

        st.session_state.current_step = 0
        st.rerun()
