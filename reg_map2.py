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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': 'https://www.example.com/bug',
        'About': 'Regulatory Mapping Tool v1.0.0'
    }
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
            --text-color: #1E1E1E;
            --success-color: #4CAF50;
            --warning-color: #ff9800;
            --error-color: #f44336;
            --sidebar-bg: #1E1E2E;
            --card-bg: #FFFFFF;
        }
        
        /* Overall layout */
        .stApp {
            background-color: var(--background-color);
        }
        
        /* Card-like containers */
        .card {
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
            border: 1px solid #EAEAEA;
        }
        
        /* Headers and text */
        h1 {
            color: var(--text-color);
            font-weight: 700;
            margin-bottom: 24px;
            font-size: 24px;
        }
        
        h2, h3 {
            color: var(--text-color);
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
            border-radius: 4px;
            font-weight: 500;
            padding: 10px 16px;
            transition: all 0.2s;
            border: none;
            background-color: #1E1E2E;
            color: white;
        }
        
        .stButton>button:hover {
            opacity: 0.9;
        }
        
        /* File uploader styling */
        .stFileUploader {
            padding: 30px;
            border: 2px dashed #CCCCCC;
            border-radius: 8px;
            background-color: #F8FAFC;
            margin-bottom: 20px;
        }
        
        .stFileUploader:hover {
            border-color: var(--primary-color);
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
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            flex: 1;
            margin: 0 8px;
            text-align: center;
            border: 1px solid #EAEAEA;
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
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            padding-top: 2rem;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #FFFFFF;
        }
        
        [data-testid="stSidebar"] hr {
            margin-top: 20px;
            margin-bottom: 20px;
            border: none;
            height: 1px;
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        /* Download button styling */
        .download-button {
            background-color: #1E1E2E;
            color: white;
            padding: 12px 16px;
            border-radius: 4px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin: 10px 0;
            display: block;
            width: 100%;
        }
        
        .download-button:hover {
            opacity: 0.9;
        }
        
        /* For file upload areas */
        .upload-area {
            background-color: rgba(240, 242, 246, 0.4);
            border-radius: 8px;
            padding: 40px 20px;
            text-align: center;
            border: 1px dashed #CED4DA;
            margin-bottom: 20px;
        }
        
        /* For the cloud icon */
        .upload-icon {
            font-size: 48px;
            color: #ADB5BD;
            margin-bottom: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

def card_container(title, content_function):
    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top: 0; margin-bottom: 16px; color: #1E1E2E;">{title}</h3>
        <div id="card-content-{title.replace(' ', '-').lower()}"></div>
    </div>
    """, unsafe_allow_html=True)
    content_function()

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
            <div class="metric-value" style="color: #1E1E2E;">{total}</div>
            <div class="metric-label">100%</div>
        </div>
    </div>
    
    <div style="margin-top: 20px; margin-bottom: 30px;">
        <div style="height: 24px; background-color: #f1f1f1; border-radius: 12px; overflow: hidden; display: flex;">
            <div style="width: {int(equivalent/total*100)}%; height: 100%; background-color: #4CAF50;"></div>
            <div style="width: {int(uplift/total*100)}%; height: 100%; background-color: #ff9800;"></div>
            <div style="width: {int(not_applicable/total*100)}%; height: 100%; background-color: #f44336;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: #666;">
            <div>
                <span style="display: inline-block; width: 10px; height: 10px; background-color: #4CAF50; border-radius: 50%; margin-right: 5px;"></span>
                Equivalent
            </div>
            <div>
                <span style="display: inline-block; width: 10px; height: 10px; background-color: #ff9800; border-radius: 50%; margin-right: 5px;"></span>
                Uplift
            </div>
            <div>
                <span style="display: inline-block; width: 10px; height: 10px; background-color: #f44336; border-radius: 50%; margin-right: 5px;"></span>
                Not Applicable
            </div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

def show_spinner_with_progress(message, duration=1.0, steps=100):
    progress_bar = st.progress(0)
    with st.spinner(message):
        step_size = 1 / steps
        for i in range(steps + 1):
            progress_bar.progress(i * step_size)
            time.sleep(duration / steps)
    progress_bar.empty()

def main():
    apply_custom_css()

    # Sidebar for navigation and info
    with st.sidebar:
        # App logo and title
        st.image("https://share.hsforms.com/1HTeegIJpS_GMdimGKwK3MQclp5j", width=72)
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
    if "theme_color" not in st.session_state:
        st.session_state.theme_color = "#1E1E2E"  # Default theme color

    # --- Step 0: File upload interface ---
    if st.session_state.current_step == 0:
        st.title("Regulatory Mapping Tool")
        st.markdown("### Step 1: Upload your regulatory requirements and knowledge base")

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
        st.markdown("### Step 2: Confirm data and choose classification scope")

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
        st.markdown("### Step 3: AI Classification")

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
        st.markdown("### Step 4: Review and approve mappings")

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
        <div style="margin: 30px 0; padding: 15px; background-color: #f8f8f8; border-left: 4px solid #1E1E2E; border-radius: 4px;">
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
        st.markdown("### Step 5: Final Output")

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
        st.markdown("<h3 style='margin-top:30px;'>Download Results</h3>", unsafe_allow_html=True)
        download_options(mapping_df)

        card_container("Next Steps", lambda: next_steps())

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Mapping", key="back_to_mapping", use_container_width=True):
                st.session_state.current_step = 3
                st.experimental_rerun()

        with col2:
            if st.button("Start New Project", key="new_project", use_container_width=True):
                # Reset session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.current_step = 0
                st.experimental_rerun()

def upload_regulatory_requirements():
    st.markdown("""
    <div class="upload-area">
        <div class="upload-icon">📤</div>
        <h3>Drag and drop file here</h3>
        <p>Limit 200MB per file • XLSX, XLS, CSV</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Excel or CSV file with regulatory requirements",
                                     type=["xlsx", "xls", "csv"],
                                     label_visibility="collapsed")

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
        # Add a demo data button for easier testing
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Load Demo Data", use_container_width=True):
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
    st.markdown("""
    <div class="upload-area">
        <div class="upload-icon">📤</div>
        <h3>Drag and drop file here</h3>
        <p>Limit 200MB per file • XLSX, XLS, CSV</p>
    </div>
    """, unsafe_allow_html=True)

    kb_file = st.file_uploader("Upload knowledge base of existing policies/controls",
                               type=["xlsx", "xls", "csv"],
                               key="kb_uploader",
                               label_visibility="collapsed")

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
        # Add a demo data button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Load Demo KB", use_container_width=True):
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
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Column Info", "Statistics"])

    with tab1:
        # Show data preview
        st.dataframe(df.head(5), use_container_width=True)

    with tab2:
        # Show column information
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Sample Values': [", ".join(df[col].astype(str).unique()[:3]) + "..." for col in df.columns]
        })

        st.dataframe(col_info, use_container_width=True)

    with tab3:
        # Show basic statistics
        col1, col2, col3 = st.columns(3)

        # Total rows
        col1.metric("Total Rows", len(df))

        # Add some data quality metrics if appropriate
        missing_data = df.isnull().sum().sum()
        col2.metric("Missing Values", missing_data)

        # Count unique values in the first column as an example stat
        if len(df.columns) > 0:
            first_col = df.columns[0]
            unique_vals = df[first_col].nunique()
            col3.metric(f"Unique {first_col} Values", unique_vals)

        # Display a summary of common columns if they exist
        common_cols = ["Source", "Category", "Priority", "Status", "Type"]
        existing_cols = [col for col in common_cols if col in df.columns]

        if existing_cols:
            st.markdown("### Value Distribution")

            for col in existing_cols:
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'Count']

                # Calculate percentage
                total = value_counts['Count'].sum()
                value_counts['Percentage'] = (value_counts['Count'] / total * 100).round(1).astype(str) + '%'

                # Display as a small table
                st.markdown(f"**{col} Distribution:**")
                st.dataframe(value_counts, use_container_width=True, height=150)

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

    # Additional settings with a more modern slider design
    st.markdown("""
    <style>
    .threshold-slider-container {
        margin-top: 20px;
    }
    .threshold-slider-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 8px;
    }
    .threshold-slider-value {
        font-size: 20px;
        font-weight: 600;
        color: #1E1E2E;
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='threshold-slider-container'>", unsafe_allow_html=True)
    st.markdown("<div class='threshold-slider-label'>Confidence Threshold for Auto-Classification</div>", unsafe_allow_html=True)

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Minimum confidence level required for automatic classification",
        label_visibility="collapsed"
    )

    st.markdown(f"<div class='threshold-slider-value'>{threshold:.1f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
        # Progress bar container
        progress_container = st.empty()
        with progress_container.container():
            st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <h3>Classification in Progress</h3>
                <p>Please wait while we analyze your requirements</p>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            progress_bar = st.progress(0)

            # A. Embedding the file
            progress_bar.progress(10)
            st.markdown("⏳ **Step 1/3:** Embedding document for analysis...")
            time.sleep(1)

            progress_bar.progress(30)

            # B. Identifying regulatory requirements
            total_requirements = len(st.session_state.requirements_df)
            st.markdown(f"⏳ **Step 2/3:** Identifying {total_requirements} requirements...")
            time.sleep(1)

            progress_bar.progress(60)

            # C. Classification simulation
            st.markdown(f"⏳ **Step 3/3:** Classifying requirements...")

            # Simulate step-by-step progress
            for i in range(60, 101, 10):
                progress_bar.progress(i)
                time.sleep(0.5)

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

            # Display completion message
            st.success("✅ Classification completed successfully!")
            time.sleep(1)

        # Clear the progress container after completion
        progress_container.empty()

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
        height=400,  # Set a fixed height to prevent the table from being too long
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

        st.markdown("<h4>Approval Status</h4>", unsafe_allow_html=True)

        metrics_html = f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">Approved</div>
                <div class="metric-value" style="color: #4CAF50;">{approved}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Rejected</div>
                <div class="metric-value" style="color: #f44336;">{rejected}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Pending</div>
                <div class="metric-value" style="color: #ff9800;">{pending}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Needs Review</div>
                <div class="metric-value" style="color: #1E1E2E;">{needs_review}</div>
            </div>
        </div>
        
        <div style="margin-top: 20px; margin-bottom: 30px;">
            <div style="height: 8px; background-color: #f1f1f1; border-radius: 4px; overflow: hidden; display: flex;">
                <div style="width: {int(approved/(approved+rejected+pending+needs_review)*100)}%; height: 100%; background-color: #4CAF50;"></div>
                <div style="width: {int(rejected/(approved+rejected+pending+needs_review)*100)}%; height: 100%; background-color: #f44336;"></div>
                <div style="width: {int(pending/(approved+rejected+pending+needs_review)*100)}%; height: 100%; background-color: #ff9800;"></div>
                <div style="width: {int(needs_review/(approved+rejected+pending+needs_review)*100)}%; height: 100%; background-color: #1E1E2E;"></div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

def download_options(mapping_df):
    st.markdown("""
    <a href="#" class="download-button">
        <span style="display: flex; align-items: center; justify-content: center">
            <span style="margin-right: 8px;">📥</span> Download Mapping (CSV)
        </span>
    </a>
    """, unsafe_allow_html=True)

    st.download_button(
        label="📥 Download Mapping (CSV)",
        data=mapping_df.to_csv(index=False).encode("utf-8"),
        file_name="regulatory_mapping_results.csv",
        mime="text/csv",
        help="Download the complete mapping results as a CSV file",
        use_container_width=True,
        key="download_csv"
    )

    st.download_button(
        label="📥 Download Mapping (Excel)",
        data=mapping_df.to_excel(index=False).encode("utf-8"),
        file_name="regulatory_mapping_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download the complete mapping results as an Excel file",
        use_container_width=True,
        key="download_excel"
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
    <div style="padding: 0 10px;">
        <h4>What happens next?</h4>
        
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: #1E1E2E; color: white; width: 28px; height: 28px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">1</div>
            <div>
                <strong>Implementation Planning</strong>
                <p style="margin: 0; color: #666; font-size: 14px;">For "Uplift" requirements, create an implementation plan with timelines and ownership.</p>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: #1E1E2E; color: white; width: 28px; height: 28px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">2</div>
            <div>
                <strong>Documentation Update</strong>
                <p style="margin: 0; color: #666; font-size: 14px;">Update your compliance documentation to reference the newly mapped controls.</p>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: #1E1E2E; color: white; width: 28px; height: 28px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">3</div>
            <div>
                <strong>Gap Remediation</strong>
                <p style="margin: 0; color: #666; font-size: 14px;">Address any identified gaps between requirements and existing controls.</p>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: #1E1E2E; color: white; width: 28px; height: 28px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">4</div>
            <div>
                <strong>Validation</strong>
                <p style="margin: 0; color: #666; font-size: 14px;">Conduct validation to ensure that controls effectively meet the requirements.</p>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: #1E1E2E; color: white; width: 28px; height: 28px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">5</div>
            <div>
                <strong>Review Cycle</strong>
                <p style="margin: 0; color: #666; font-size: 14px;">Schedule a regular review cycle to keep mappings up-to-date with regulatory changes.</p>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 20px; padding: 15px; background-color: #f8f8f8; border-left: 4px solid #1E1E2E; border-radius: 4px;">
        <p><strong>💡 Tip:</strong> Use the exported file to update your Governance, Risk, and Compliance (GRC) system.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
