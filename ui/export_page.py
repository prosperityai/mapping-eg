# File: ui/export_page.py

import streamlit as st
import pandas as pd
import tempfile
import os
import logging

# Configure logging
logger = logging.getLogger('export_page')

def display_export_page():
    """Display the export interface for mapped requirements"""
    st.header("Step 6: Export Final Results")

    # Determine which type of data to export based on what's available
    has_classified_requirements = "classified_requirements" in st.session_state and st.session_state.classified_requirements
    has_mapped_requirements = "mapped_requirements" in st.session_state and st.session_state.mapped_requirements

    if not has_classified_requirements and not has_mapped_requirements:
        st.error("No requirements data found. Please go back and process requirements first.")
        if st.button("Back to Classification"):
            st.session_state.current_step = 3
            st.rerun()
        st.stop()

    # Determine which data to show based on availability
    if has_mapped_requirements:
        st.write("""
        Excellent! You've completed the full mapping process.
        
        You can now export the comprehensive mapping results, including:
        - Requirement details and classifications
        - Coverage assessments (Equivalent, Partial Uplift, Full Uplift)
        - Specific policy references (Master Sheet, Minor Sheet, etc.)
        - Gap analysis for items requiring uplift
        """)

        # Create a DataFrame with the complete mapping results
        mapping_rows = []

        for req in st.session_state.mapped_requirements:
            # Extract data
            requirement_data = req.get("requirement", {})
            req_text = requirement_data.get("text", "")
            req_metadata = requirement_data.get("metadata", {})
            req_type = requirement_data.get("type", "Unknown")

            coverage = req.get("coverage", "Unknown")
            confidence = req.get("confidence", 0.0)
            coverage_explanation = req.get("coverage_explanation", "")
            gap_analysis = req.get("gap_analysis", "")
            mapped_policies = req.get("mapped_policies", [])

            # Create a row with all the mapping information
            row = {}

            # Add original metadata if available
            if "original_data" in req_metadata:
                row.update(req_metadata["original_data"])

            # Add requirement text and classification
            row["Requirement Text"] = req_text
            row["Requirement Type"] = req_type

            # Add mapping details
            row["Coverage"] = coverage
            row["Confidence"] = confidence
            row["Coverage Explanation"] = coverage_explanation
            row["Gap Analysis"] = gap_analysis

            # Add KYC Standards Impact column
            if coverage == "Equivalent":
                row["KYC Standards Impact"] = "Equivalent"
            else:
                row["KYC Standards Impact"] = "Uplift"

            # Extract Master Sheet and Minor Sheet details from mapped policies
            # If there are multiple mapped policies, use the first one with highest relevance
            master_sheets = []
            minor_sheets = []
            sections = []
            titles = []
            policy_refs = []

            # Sort policies by relevance score in descending order
            sorted_policies = sorted(mapped_policies, key=lambda x: x.get('relevance_score', 0), reverse=True)

            for policy in sorted_policies:
                master_sheet = policy.get('master_sheet', '')
                minor_sheet = policy.get('minor_sheet', '')
                section = policy.get('section', '')
                title = policy.get('title', '')

                if master_sheet:
                    master_sheets.append(master_sheet)
                if minor_sheet:
                    minor_sheets.append(minor_sheet)
                if section:
                    sections.append(section)
                if title:
                    titles.append(title)

                policy_refs.append(f"{master_sheet}-{minor_sheet}")

            # Add separate columns for Master Sheet and Minor Sheet
            row["Master Sheets"] = ", ".join(master_sheets)
            row["Minor Sheets"] = ", ".join(minor_sheets)
            row["Sections"] = ", ".join(sections)
            row["Titles"] = ", ".join(titles)
            row["Policy References"] = ", ".join(policy_refs)

            # Add mappings for specific B1 format columns
            row["If equivalent, provide KYC Standard, including section"] = ", ".join(policy_refs) if coverage == "Equivalent" else ""

            # For Program Level requirements
            if req_type == "Program-Level":
                row["If Program Level requirement, select Appendix Serial #"] = ", ".join(policy_refs) if policy_refs else ""

            # For Uplifts
            if coverage in ["Partial Uplift", "Full Uplift"]:
                row["If Uplift, provide Local Guidance"] = gap_analysis

            mapping_rows.append(row)

        # Create DataFrame
        if mapping_rows:
            mapping_df = pd.DataFrame(mapping_rows)

            # Remove any duplicate columns
            mapping_df = mapping_df.loc[:, ~mapping_df.columns.duplicated()]

            # Reorder columns to prioritize Master Sheet and Minor Sheet
            priority_columns = [
                "Requirement Text",
                "Requirement Type",
                "Coverage",
                "KYC Standards Impact",
                "Master Sheets",
                "Minor Sheets",
                "Sections",
                "Titles"
            ]

            # Get remaining columns excluding priority ones
            remaining_columns = [col for col in mapping_df.columns if col not in priority_columns]

            # Create new column order
            new_column_order = priority_columns + [col for col in remaining_columns if col in mapping_df.columns]

            # Reorder DataFrame
            mapping_df = mapping_df[new_column_order]

            # Display the mapping results
            st.write("### Comprehensive Mapping Results")
            st.dataframe(mapping_df, use_container_width=True)

            # Create Excel file for download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                with pd.ExcelWriter(tmp_file.name, engine="openpyxl") as writer:
                    mapping_df.to_excel(writer, index=False, sheet_name="Regulatory Mapping")

                    # Add a summary sheet
                    summary_data = {
                        "Metric": [
                            "Total Requirements",
                            "CDD Requirements",
                            "Program-Level Requirements",
                            "Equivalent Coverage",
                            "Partial Uplift Needed",
                            "Full Uplift Needed",
                            "Average Confidence"
                        ]
                    }

                    # Calculate metrics
                    total = len(st.session_state.mapped_requirements)
                    cdd_count = sum(1 for r in st.session_state.mapped_requirements if r.get("requirement", {}).get("type") == "CDD")
                    program_count = sum(1 for r in st.session_state.mapped_requirements if r.get("requirement", {}).get("type") == "Program-Level")

                    equivalent_count = sum(1 for r in st.session_state.mapped_requirements if r.get("coverage") == "Equivalent")
                    partial_count = sum(1 for r in st.session_state.mapped_requirements if r.get("coverage") == "Partial Uplift")
                    full_count = sum(1 for r in st.session_state.mapped_requirements if r.get("coverage") == "Full Uplift")

                    avg_confidence = sum(r.get("confidence", 0) for r in st.session_state.mapped_requirements) / total if total > 0 else 0

                    summary_data["Value"] = [
                        total,
                        cdd_count,
                        program_count,
                        equivalent_count,
                        partial_count,
                        full_count,
                        f"{avg_confidence:.2f}"
                    ]

                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, index=False, sheet_name="Summary")

                    # Add a policies sheet with detailed mapping of each policy
                    policy_rows = []
                    for i, req in enumerate(st.session_state.mapped_requirements):
                        requirement_data = req.get("requirement", {})
                        req_text = requirement_data.get("text", "")
                        req_type = requirement_data.get("type", "Unknown")
                        coverage = req.get("coverage", "Unknown")

                        # Process each mapped policy separately
                        for policy in req.get("mapped_policies", []):
                            policy_row = {
                                "Requirement Number": i + 1,
                                "Requirement Text": req_text,
                                "Requirement Type": req_type,
                                "Coverage": coverage,
                                "Master Sheet": policy.get("master_sheet", ""),
                                "Minor Sheet": policy.get("minor_sheet", ""),
                                "Section": policy.get("section", ""),
                                "Title": policy.get("title", ""),
                                "Relevance Score": policy.get("relevance_score", 0),
                                "Policy Explanation": policy.get("explanation", "")
                            }
                            policy_rows.append(policy_row)

                    if policy_rows:
                        policy_df = pd.DataFrame(policy_rows)
                        policy_df.to_excel(writer, index=False, sheet_name="Detailed Policy Mappings")

                excel_data = open(tmp_file.name, "rb").read()

                # Download button
                st.download_button(
                    label="📥 Download Complete Mapping Results (Excel)",
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

    # If we only have classification data but not mapping data
    elif has_classified_requirements:
        st.write("""
        You've completed the classification step but haven't performed the mapping yet.
        
        You can export the classification results now, or go back to complete the mapping process
        for more comprehensive results.
        """)

        # Create a DataFrame with the classification results
        classification_rows = []

        for req in st.session_state.classified_requirements:
            # Extract requirement info
            req_text = req.get("requirement", {}).get("text", "")
            req_metadata = req.get("requirement", {}).get("metadata", {})
            classification = req.get("classification", {})

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

            # Add button to go to mapping
            st.info("For more comprehensive results, complete the mapping process.")
            if st.button("Continue to Mapping"):
                st.session_state.current_step = 4
                st.rerun()
        else:
            st.warning("No classification results to display.")

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if has_mapped_requirements:
            if st.button("⬅️ Back to Mapping"):
                st.session_state.current_step = 4
                st.rerun()
        else:
            if st.button("⬅️ Back to Classification"):
                st.session_state.current_step = 3
                st.rerun()

    with col2:
        if st.button("Start New Project"):
            # Reset session state
            for key in list(st.session_state.keys()):
                # Keep only the embedding agent and classification agent to avoid re-initialization
                if key not in ["embedding_agent", "classification_agent", "mapping_agent"]:
                    del st.session_state[key]

            st.session_state.current_step = 0
            st.rerun()
