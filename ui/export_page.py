# File: ui/export_page.py

import streamlit as st
import pandas as pd
import tempfile
import os
import logging

# Configure logging
logger = logging.getLogger('export_page')

def display_export_page():
    """Display the final mapping and export interface"""
    st.header("Step 5: Final Mapping and Export")

    if "processed_requirements" not in st.session_state or not st.session_state.processed_requirements:
        st.error("No processed requirements found. Please go back and process the requirements first.")
        if st.button("Back to Classification"):
            st.session_state.current_step = 3
            st.rerun()
        st.stop()

    st.write("""
    Review the final mappings and export the results to Excel.
    """)

    # Create a DataFrame with the mapping results
    mapping_rows = []

    for req in st.session_state.processed_requirements:
        # Extract requirement info
        req_text = req["requirement"]["text"]
        req_metadata = req["requirement"]["metadata"]
        classification = req["classification"]

        # Create a row with original data and mapping information
        row = {}

        # Add all original metadata
        if "original_data" in req_metadata:
            row.update(req_metadata["original_data"])

        # Add classification information
        row["Type"] = classification.get("type", "")
        row["Coverage"] = classification.get("coverage", "")
        row["Mapped Policies"] = ", ".join(classification.get("mapped_policies", []))
        row["Explanation"] = classification.get("explanation", "")
        row["Confidence"] = classification.get("confidence", 0.0)

        # Add KYC Standards Impact column
        if classification.get("coverage") == "Equivalent":
            row["KYC Standards Impact"] = "Equivalent"
        else:
            row["KYC Standards Impact"] = "Uplift"

        # Add mappings for specific columns based on B1 format
        row["If equivalent, provide KYC Standard, including section"] = row["Mapped Policies"]

        # For Program Level requirements
        if classification.get("type") == "Program-Level":
            # Extract IDs if available
            ids = []
            for policy_id in classification.get("mapped_policies", []):
                if policy_id.strip():
                    ids.append(policy_id)

            if ids:
                row["If Program Level requirement, select Appendix Serial #"] = ", ".join(ids)

        # For Uplifts
        if "Uplift" in classification.get("coverage", ""):
            row["If Uplift, provide Local Guidance"] = classification.get("explanation", "")

        mapping_rows.append(row)

    # Create DataFrame
    if mapping_rows:
        mapping_df = pd.DataFrame(mapping_rows)

        # Remove any duplicate columns
        mapping_df = mapping_df.loc[:, ~mapping_df.columns.duplicated()]

        # Display the mapping results
        st.write("### Mapping Results")
        st.dataframe(mapping_df, use_container_width=True)

        # Export options
        st.write("### Export Options")

        # Create Excel file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            with pd.ExcelWriter(tmp_file.name, engine="openpyxl") as writer:
                mapping_df.to_excel(writer, index=False, sheet_name="Regulatory Mapping")

            excel_data = open(tmp_file.name, "rb").read()

            # Download button
            st.download_button(
                label="📥 Download Mapping Results (Excel)",
                data=excel_data,
                file_name="regulatory_mapping_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_mapping",
                help="Download the complete mapping results as an Excel file",
                use_container_width=True
            )

            # Cleanup temp file
            os.unlink(tmp_file.name)
    else:
        st.warning("No mapping results to display.")

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
