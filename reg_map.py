import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Regulatory Mapping Tool",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more beautiful UI
def apply_custom_css():
    st.markdown("""
    <style>
        /* Main colors */
        :root {
            --primary-color: #4F8BF9;
            --secondary-color: #5EEAD4;
            --background-color: #F8F9FA;
            --text-color: #333333;
            --success-color: #4CAF50;
            --warning-color: #ff9800;
            --error-color: #f44336;
        }
        
        /* Card-like containers */
        .stApp {
            background-color: var(--background-color);
        }
        
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Headers and text */
        h1 {
            color: var(--primary-color);
            font-weight: 700;
            margin-bottom: 20px;
        }
        
        h2, h3 {
            color: var(--primary-color);
            font-weight: 600;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-equivalent {
            background-color: var(--success-color);
        }
        
        .status-uplift {
            background-color: var(--warning-color);
        }
        
        .status-na {
            background-color: var(--error-color);
        }
        
        /* Custom button styles */
        .stButton>button {
            border-radius: 5px;
            font-weight: 600;
            padding: 8px 16px;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Progress bars */
        .stProgress>div>div {
            background-color: var(--primary-color);
        }
        
        /* Metrics row */
        .metrics-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            flex: 1;
            margin: 0 10px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .metric-label {
            color: #666;
            font-size: 14px;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #2C3E50;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

def card_container(title, content_function):
    st.markdown(f"<div class='card'><h3>{title}</h3>", unsafe_allow_html=True)
    content_function()
    st.markdown("</div>", unsafe_allow_html=True)

def show_metrics(equivalent, uplift, not_applicable, total):
    metrics_html = f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-label">Equivalent</div>
            <div class="metric-value" style="color: #4CAF50;">{equivalent}</div>
            <div class="metric-label">{int(equivalent/total*100)}% of total</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Uplift</div>
            <div class="metric-value" style="color: #ff9800;">{uplift}</div>
            <div class="metric-label">{int(uplift/total*100)}% of total</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Not Applicable</div>
            <div class="metric-value" style="color: #f44336;">{not_applicable}</div>
            <div class="metric-label">{int(not_applicable/total*100)}% of total</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Requirements</div>
            <div class="metric-value" style="color: #4F8BF9;">{total}</div>
            <div class="metric-label">100%</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

def show_spinner_with_progress(message, duration=1.0):
    progress_bar = st.progress(0)
    with st.spinner(message):
        for i in range(101):
            progress_bar.progress(i/100)
            time.sleep(duration/100)
    progress_bar.empty()

def main():
    apply_custom_css()

    # Sidebar for navigation and info
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1024px-ChatGPT_logo.svg.png", width=100)
        st.title("Regulatory Mapping")
        st.markdown("### Navigation")

        # Display current step indicator
        if "current_step" in st.session_state:
            current_step = st.session_state.current_step
            steps = ["Upload", "Review", "Classification", "Mapping", "Export"]

            for i, step in enumerate(steps):
                if i < current_step:
                    st.markdown(f"✅ Step {i+1}: {step}")
                elif i == current_step:
                    st.markdown(f"🔵 **Step {i+1}: {step}**")
                else:
                    st.markdown(f"⚪ Step {i+1}: {step}")

        st.markdown("---")
        st.markdown("### Session Info")
        if "requirements_df" in st.session_state and st.session_state.requirements_df is not None:
            st.success("✅ Requirements loaded")
            row_count = len(st.session_state.requirements_df)
            st.markdown(f"📊 {row_count} requirements")
        else:
            st.warning("⚠️ No requirements loaded")

        if "kb_df" in st.session_state and st.session_state.kb_df is not None:
            st.success("✅ Knowledge base loaded")
        else:
            st.info("ℹ️ No knowledge base")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("This tool helps map regulatory requirements to existing controls and policies.")
        st.markdown("v1.0.0 - " + datetime.now().strftime("%Y-%m-%d"))

    # --- Session states to store intermediate data ---
    if "requirements_df" not in st.session_state:
        st.session_state.requirements_df = None  # will hold the Excel data
    if "kb_df" not in st.session_state:
        st.session_state.kb_df = None  # knowledge base placeholder
    if "classification_logs" not in st.session_state:
        st.session_state.classification_logs = pd.DataFrame()
    if "mapping_df" not in st.session_state:
        st.session_state.mapping_df = pd.DataFrame()
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0  # track the wizard step

    # --- Step 0: File upload interface ---
    if st.session_state.current_step == 0:
        st.title("Regulatory Mapping Tool")
        st.markdown("### Upload your regulatory requirements and knowledge base")

        col1, col2 = st.columns([1, 1])

        with col1:
            card_container("Regulatory Requirements", lambda: upload_regulatory_requirements())

        with col2:
            card_container("Knowledge Base (Optional)", lambda: upload_knowledge_base())

        if st.session_state.requirements_df is not None:
            st.button("Next: Review Data", on_click=lambda: setattr(st.session_state, "current_step", 1),
                      use_container_width=True)

    # --- Step 1: Show user a quick preview, pick a classification scope ---
    elif st.session_state.current_step == 1:
        st.title("Review and Configure")

        # Show a short preview of the data
        req_df = st.session_state.requirements_df

        card_container("Data Preview", lambda: preview_data(req_df))

        card_container("Classification Settings", lambda: classification_settings())

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back", key="back_to_upload"):
                st.session_state.current_step = 0
                st.experimental_rerun()

        with col2:
            if st.button("Start Classification ➡️", key="start_classification"):
                st.session_state.current_step = 2
                st.experimental_rerun()

    # --- Step 2: Simulate agent calls ---
    elif st.session_state.current_step == 2:
        st.title("Classification in Progress")

        # Run the simulation/classification
        run_classification_simulation()

        # Display metrics
        total_requirements = len(st.session_state.requirements_df)
        classification_df = st.session_state.classification_logs
        eq_count = (classification_df["Classification"] == "Equivalent").sum()
        up_count = (classification_df["Classification"] == "Uplift").sum()
        na_count = (classification_df["Classification"] == "Not Applicable").sum()

        show_metrics(eq_count, up_count, na_count, total_requirements)

        # Display classification logs
        card_container("Classification Results", lambda: st.dataframe(classification_df, use_container_width=True))

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back", key="back_to_review"):
                st.session_state.current_step = 1
                st.experimental_rerun()

        with col2:
            if st.button("Generate Mappings ➡️", key="generate_mappings"):
                st.session_state.current_step = 3
                st.experimental_rerun()

    # --- Step 3: Generate mapping table & let user Approve/Change ---
    elif st.session_state.current_step == 3:
        st.title("Mapping Approval")

        # Generate and display the mapping table
        mapping_table = generate_mapping_table()

        # Provide summary metrics
        total_requirements = len(mapping_table)
        eq_count = (mapping_table["Classification"] == "Equivalent").sum()
        up_count = (mapping_table["Classification"] == "Uplift").sum()
        na_count = (mapping_table["Classification"] == "Not Applicable").sum()

        show_metrics(eq_count, up_count, na_count, total_requirements)

        # Show the mapping table with data editor
        card_container("Review and Approve Mappings", lambda: show_data_editor(mapping_table))

        st.markdown("""
        <div style="margin: 30px 0; padding: 15px; background-color: #e7f3fe; border-left: 6px solid #2196F3; border-radius: 5px;">
            <p><strong>How to use:</strong> Review each mapping and make any necessary changes. You can approve or modify the classifications and recommended actions.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back", key="back_to_classification"):
                st.session_state.current_step = 2
                st.experimental_rerun()

        with col2:
            if st.button("Finalize and Export ➡️", key="finalize_export"):
                # Store the edited dataframe
                st.session_state.mapping_df = st.session_state.edited_df
                st.session_state.current_step = 4
                st.experimental_rerun()

    # --- Step 4: Final output ---
    elif st.session_state.current_step == 4:
        st.title("Export Results")

        # Show completion message with a celebration
        st.balloons()

        st.success("🎉 Congratulations! Mappings have been finalized and are ready for export.")

        # Final metrics
        mapping_df = st.session_state.mapping_df
        total_requirements = len(mapping_df)
        eq_count = (mapping_df["Classification"] == "Equivalent").sum()
        up_count = (mapping_df["Classification"] == "Uplift").sum()
        na_count = (mapping_df["Classification"] == "Not Applicable").sum()

        show_metrics(eq_count, up_count, na_count, total_requirements)

        # Show the final table
        card_container("Final Mapping Results", lambda: st.dataframe(mapping_df, use_container_width=True))

        # Export options
        col1, col2 = st.columns([1, 1])

        with col1:
            card_container("Download Options", lambda: download_options(mapping_df))

        with col2:
            card_container("Next Steps", lambda: next_steps())

        if st.button("⬅️ Back to Mapping", key="back_to_mapping"):
            st.session_state.current_step = 3
            st.experimental_rerun()

        if st.button("Start New Project", key="new_project"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.current_step = 0
            st.experimental_rerun()

def upload_regulatory_requirements():
    uploaded_file = st.file_uploader("Upload Excel or CSV file with regulatory requirements", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            # Show loading spinner
            with st.spinner("Processing file..."):
                # We'll assume user can also upload CSV for simplicity
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Fix for Arrow serialization error - convert 'Unnamed: 0' column if exists
                if 'Unnamed: 0' in df.columns:
                    df['Unnamed: 0'] = df['Unnamed: 0'].astype(str)

                # Ensure all object columns are string types for better compatibility
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str)

                st.session_state.requirements_df = df
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                st.markdown(f"📊 Found {len(df)} rows of data")

                # Show a preview
                if not df.empty:
                    st.dataframe(df.head(3), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Please upload an Excel (.xlsx, .xls) or CSV (.csv) file containing your regulatory requirements.")

        # Add a demo data button for easier testing
        if st.button("Load Demo Data"):
            # Create a simple demo dataframe
            demo_df = pd.DataFrame({
                'RequirementID': [f'REQ-{i:03d}' for i in range(1, 51)],
                'Description': [f'Requirement description {i}' for i in range(1, 51)],
                'Source': np.random.choice(['ISO 27001', 'GDPR', 'PCI DSS', 'SOC 2'], 50),
                'Category': np.random.choice(['Security', 'Privacy', 'Compliance', 'Operational'], 50),
                'Priority': np.random.choice(['High', 'Medium', 'Low'], 50),
            })
            st.session_state.requirements_df = demo_df
            st.success("Demo data loaded successfully!")
            st.dataframe(demo_df.head(3), use_container_width=True)

def upload_knowledge_base():
    kb_file = st.file_uploader("Upload knowledge base of existing policies/controls", type=["xlsx", "xls", "csv"], key="kb_uploader")

    if kb_file is not None:
        try:
            with st.spinner("Processing knowledge base..."):
                if kb_file.name.endswith(".csv"):
                    kb_df = pd.read_csv(kb_file)
                else:
                    kb_df = pd.read_excel(kb_file)

                # Fix for Arrow serialization error
                if 'Unnamed: 0' in kb_df.columns:
                    kb_df['Unnamed: 0'] = kb_df['Unnamed: 0'].astype(str)

                # Ensure all object columns are string types
                for col in kb_df.select_dtypes(include=['object']).columns:
                    kb_df[col] = kb_df[col].astype(str)

                st.session_state.kb_df = kb_df
                st.success(f"Knowledge base '{kb_file.name}' uploaded successfully!")

                # Show a preview
                if not kb_df.empty:
                    st.dataframe(kb_df.head(3), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading KB file: {e}")
    else:
        st.info("(Optional) Upload a knowledge base of existing policies and controls for better mapping results.")

        # Add a demo data button
        if st.button("Load Demo KB"):
            # Create a simple demo KB dataframe
            demo_kb = pd.DataFrame({
                'ControlID': [f'CTL-{i:03d}' for i in range(1, 31)],
                'ControlName': [f'Control {i}' for i in range(1, 31)],
                'Description': [f'This control ensures that {np.random.choice(["data is protected", "systems are secured", "privacy is maintained", "compliance is achieved"])}' for _ in range(30)],
                'Type': np.random.choice(['Technical', 'Administrative', 'Physical'], 30),
                'Status': np.random.choice(['Implemented', 'Planned', 'Under Review'], 30),
            })
            st.session_state.kb_df = demo_kb
            st.success("Demo knowledge base loaded successfully!")
            st.dataframe(demo_kb.head(3), use_container_width=True)

def preview_data(df):
    # Show column information
    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count().values,
        'Sample Values': [", ".join(df[col].astype(str).unique()[:3]) + "..." for col in df.columns]
    })

    st.write("Data structure:")
    st.dataframe(col_info, use_container_width=True)

    # Show data preview
    st.write("Data preview:")
    st.dataframe(df.head(5), use_container_width=True)

    # Show basic statistics
    st.write(f"Total rows: {len(df)}")

    # Add some data quality metrics if appropriate
    missing_data = df.isnull().sum().sum()
    if missing_data > 0:
        st.warning(f"Found {missing_data} missing values in the dataset")

def classification_settings():
    # Let user choose a classification bucket (like B1)
    st.write("Select regulatory framework and classification parameters:")

    col1, col2 = st.columns(2)

    with col1:
        category_selection = st.selectbox(
            "Regulatory Framework:",
            ["B1", "Program-Level", "Local Definition", "GDPR", "PCI DSS", "ISO 27001", "SOC 2", "Other"],
            help="Select the regulatory framework you're mapping against"
        )

    with col2:
        classification_method = st.selectbox(
            "Classification Method:",
            ["AI-Assisted", "Rule-Based", "Manual"],
            help="Select how requirements should be classified"
        )

    # Additional settings
    threshold = st.slider(
        "Confidence Threshold for Auto-Classification",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Minimum confidence level required for automatic classification"
    )

    # Store these settings
    if "settings" not in st.session_state:
        st.session_state.settings = {}

    st.session_state.settings.update({
        "category": category_selection,
        "method": classification_method,
        "threshold": threshold
    })

def run_classification_simulation():
    if "classification_completed" not in st.session_state or not st.session_state.classification_completed:
        # A. Embedding the file
        show_spinner_with_progress("Embedding document for analysis...", 1.5)

        # B. Identifying regulatory requirements
        total_requirements = len(st.session_state.requirements_df)
        st.info(f"Found **{total_requirements}** total requirements for classification")

        # C. Classification simulation
        show_spinner_with_progress(f"Classifying {total_requirements} requirements...", 3.0)

        # Simulate classification results
        simulated_results = []
        for i in range(total_requirements):
            # Random classification - in a real app, this would use actual classification logic
            classification = np.random.choice(["Equivalent", "Uplift", "Not Applicable"],
                                              p=[0.6, 0.3, 0.1])  # weighted probabilities

            confidence = np.random.uniform(0.65, 0.98)
            reason = f"This requirement appears to be {classification.lower()} to existing controls because "

            if classification == "Equivalent":
                reason += f"it matches control CTL-{np.random.randint(100, 999)} with {confidence:.2f} confidence."
            elif classification == "Uplift":
                reason += f"it requires additional implementation beyond existing controls with {confidence:.2f} confidence."
            else:
                reason += f"it does not apply to our organization's context with {confidence:.2f} confidence."

            simulated_results.append({
                "ReqIndex": i,
                "RequirementID": f"REQ-{i+1:03d}" if "RequirementID" not in st.session_state.requirements_df.columns else st.session_state.requirements_df.iloc[i].get("RequirementID", f"REQ-{i+1:03d}"),
                "Classification": classification,
                "Confidence": confidence,
                "Reason": reason
            })

        classification_df = pd.DataFrame(simulated_results)
        st.session_state.classification_logs = classification_df
        st.session_state.classification_completed = True

    return st.session_state.classification_logs

def generate_mapping_table():
    # Combine the classification logs with the original data
    req_df = st.session_state.requirements_df.copy()
    class_df = st.session_state.classification_logs.copy()

    # For demonstration, we'll create a mapping table by merging on index
    # In a real app, you'd have a more robust joining mechanism

    # First ensure we have a column to join on
    if "RequirementID" in req_df.columns and "RequirementID" in class_df.columns:
        joined_df = pd.merge(req_df, class_df, on="RequirementID", how="left")
    else:
        # If no common ID, use the index
        class_df_indexed = class_df.set_index("ReqIndex")
        # Add classification info to original dataframe
        for col in ["Classification", "Confidence", "Reason"]:
            req_df[col] = req_df.index.map(lambda x: class_df_indexed.loc[x, col] if x in class_df_indexed.index else None)
        joined_df = req_df

    # Add recommended action based on classification
    def recommended_action(row):
        classification = row.get("Classification")
        if classification == "Equivalent":
            # In a real app, you'd look up actual control IDs
            control_id = f"CTL-{np.random.randint(100, 999)}"
            return f"Map to existing control {control_id}"
        elif classification == "Uplift":
            return "Create new control requirement"
        else:
            return "No action required"

    joined_df["RecommendedAction"] = joined_df.apply(recommended_action, axis=1)

    # Add approval status column for user input
    joined_df["ApprovalStatus"] = "Pending"

    return joined_df

def show_data_editor(mapping_table):
    # Configure the columns to show in the data editor
    display_cols = mapping_table.columns.tolist()

    # We'll use st.data_editor (the replacement for experimental_data_editor)
    edited_df = st.data_editor(
        mapping_table,
        column_config={
            "ApprovalStatus": st.column_config.SelectboxColumn(
                "Approval Status",
                help="Approve or reject the suggested mapping",
                width="medium",
                options=["Approved", "Rejected", "Pending", "Needs Review"],
                required=True,
            ),
            "Classification": st.column_config.SelectboxColumn(
                "Classification",
                help="The classification of this requirement",
                width="medium",
                options=["Equivalent", "Uplift", "Not Applicable"],
                required=True,
            ),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="AI confidence in this classification",
                width="small",
                format="%.0f%%",
                min_value=0,
                max_value=1,
            ),
            "RecommendedAction": st.column_config.TextColumn(
                "Recommended Action",
                help="The suggested action to take",
                width="large",
            ),
        },
        use_container_width=True,
        num_rows="dynamic",
        key="mapping_editor"
    )

    # Store the edited dataframe in session state
    st.session_state.edited_df = edited_df

    # Show approval statistics
    if "edited_df" in st.session_state:
        approved = (st.session_state.edited_df["ApprovalStatus"] == "Approved").sum()
        rejected = (st.session_state.edited_df["ApprovalStatus"] == "Rejected").sum()
        pending = (st.session_state.edited_df["ApprovalStatus"] == "Pending").sum()
        needs_review = (st.session_state.edited_df["ApprovalStatus"] == "Needs Review").sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Approved", approved)
        col2.metric("Rejected", rejected)
        col3.metric("Pending", pending)
        col4.metric("Needs Review", needs_review)

def download_options(mapping_df):
    st.download_button(
        label="📥 Download Mapping (CSV)",
        data=mapping_df.to_csv(index=False).encode("utf-8"),
        file_name="regulatory_mapping_results.csv",
        mime="text/csv",
        help="Download the complete mapping results as a CSV file",
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Mapping (Excel)",
        data=mapping_df.to_excel(index=False).encode("utf-8"),
        file_name="regulatory_mapping_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download the complete mapping results as an Excel file",
        use_container_width=True
    )

    # Add an option for a summary report
    if st.button("📊 Generate Summary Report", use_container_width=True):
        st.session_state.show_summary = True

    if "show_summary" in st.session_state and st.session_state.show_summary:
        st.markdown("### Summary Report")

        # Generate some summary stats
        total = len(mapping_df)
        by_classification = mapping_df["Classification"].value_counts()
        by_approval = mapping_df["ApprovalStatus"].value_counts()

        st.write("#### Classification Summary")
        st.write(f"- **Equivalent:** {by_classification.get('Equivalent', 0)} ({by_classification.get('Equivalent', 0)/total*100:.1f}%)")
        st.write(f"- **Uplift:** {by_classification.get('Uplift', 0)} ({by_classification.get('Uplift', 0)/total*100:.1f}%)")
        st.write(f"- **Not Applicable:** {by_classification.get('Not Applicable', 0)} ({by_classification.get('Not Applicable', 0)/total*100:.1f}%)")

        st.write("#### Approval Status")
        st.write(f"- **Approved:** {by_approval.get('Approved', 0)} ({by_approval.get('Approved', 0)/total*100:.1f}%)")
        st.write(f"- **Rejected:** {by_approval.get('Rejected', 0)} ({by_approval.get('Rejected', 0)/total*100:.1f}%)")
        st.write(f"- **Pending:** {by_approval.get('Pending', 0)} ({by_approval.get('Pending', 0)/total*100:.1f}%)")
        st.write(f"- **Needs Review:** {by_approval.get('Needs Review', 0)} ({by_approval.get('Needs Review', 0)/total*100:.1f}%)")

def next_steps():
    st.markdown("""
    ### What happens next?
    
    1. **Implementation Planning**: For "Uplift" requirements, create an implementation plan with timelines and ownership.
    
    2. **Documentation Update**: Update your compliance documentation to reference the newly mapped controls.
    
    3. **Gap Remediation**: Address any identified gaps between requirements and existing controls.
    
    4. **Validation**: Conduct validation to ensure that controls effectively meet the requirements.
    
    5. **Review Cycle**: Schedule a regular review cycle to keep mappings up-to-date with regulatory changes.
    """)

    st.info("💡 Tip: Use the exported file to update your Governance, Risk, and Compliance (GRC) system.")

if __name__ == "__main__":
    main()
