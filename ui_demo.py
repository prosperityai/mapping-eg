import gradio as gr
import pandas as pd
import numpy as np
import time
from datetime import datetime

# For file handling
import tempfile
import os

class RegMappingApp:
    def __init__(self):
        self.requirements_df = None
        self.kb_df = None
        self.classification_logs = pd.DataFrame()
        self.mapping_df = pd.DataFrame()
        self.settings = {}
        self.current_step = 0

    def upload_requirements(self, file):
        """Process uploaded requirements file"""
        if file is None:
            return None, "No file uploaded"

        try:
            # Handle file upload
            temp_path = tempfile.NamedTemporaryFile(delete=False).name
            with open(temp_path, "wb") as f:
                f.write(file)

            # Read based on file type
            if file.name.endswith(".csv"):
                df = pd.read_csv(temp_path)
            else:
                df = pd.read_excel(temp_path)

            # Clean up temp file
            os.unlink(temp_path)

            # Fix for any problematic columns
            if 'Unnamed: 0' in df.columns:
                df['Unnamed: 0'] = df['Unnamed: 0'].astype(str)

            # Save dataframe
            self.requirements_df = df

            # Return preview
            return df.head(5).to_html(), f"Successfully loaded {len(df)} requirements"
        except Exception as e:
            return None, f"Error processing file: {str(e)}"

    def load_demo_data(self):
        """Load demo data for requirements"""
        demo_df = pd.DataFrame({
            'RequirementID': [f'REQ-{i:03d}' for i in range(1, 51)],
            'Description': [f'Requirement description {i}' for i in range(1, 51)],
            'Source': np.random.choice(['ISO 27001', 'GDPR', 'PCI DSS', 'SOC 2'], 50),
            'Category': np.random.choice(['Security', 'Privacy', 'Compliance', 'Operational'], 50),
            'Priority': np.random.choice(['High', 'Medium', 'Low'], 50),
        })
        self.requirements_df = demo_df
        return demo_df.head(5).to_html(), "Demo data loaded successfully"

    def upload_kb(self, file):
        """Process uploaded knowledge base file"""
        if file is None:
            return None, "No file uploaded"

        try:
            # Handle file upload
            temp_path = tempfile.NamedTemporaryFile(delete=False).name
            with open(temp_path, "wb") as f:
                f.write(file)

            # Read based on file type
            if file.name.endswith(".csv"):
                df = pd.read_csv(temp_path)
            else:
                df = pd.read_excel(temp_path)

            # Clean up temp file
            os.unlink(temp_path)

            # Fix for any problematic columns
            if 'Unnamed: 0' in df.columns:
                df['Unnamed: 0'] = df['Unnamed: 0'].astype(str)

            # Save dataframe
            self.kb_df = df

            # Return preview
            return df.head(5).to_html(), f"Successfully loaded knowledge base with {len(df)} entries"
        except Exception as e:
            return None, f"Error processing file: {str(e)}"

    def load_demo_kb(self):
        """Load demo data for knowledge base"""
        demo_kb = pd.DataFrame({
            'ControlID': [f'CTL-{i:03d}' for i in range(1, 31)],
            'ControlName': [f'Control {i}' for i in range(1, 31)],
            'Description': [f'This control ensures that {np.random.choice(["data is protected", "systems are secured", "privacy is maintained", "compliance is achieved"])}' for _ in range(30)],
            'Type': np.random.choice(['Technical', 'Administrative', 'Physical'], 30),
            'Status': np.random.choice(['Implemented', 'Planned', 'Under Review'], 30),
        })
        self.kb_df = demo_kb
        return demo_kb.head(5).to_html(), "Demo knowledge base loaded successfully"

    def save_settings(self, framework, method, threshold):
        """Save classification settings"""
        self.settings = {
            "framework": framework,
            "method": method,
            "threshold": threshold
        }
        return f"Settings saved: {framework}, {method}, threshold={threshold}"

    def run_classification(self):
        """Run the classification simulation"""
        if self.requirements_df is None:
            return "Error: No requirements data loaded", None

        total = len(self.requirements_df)

        # Simulate classification process
        time.sleep(1)

        # Classification simulation
        simulated_results = []

        for i in range(total):
            # Random classification
            classification = np.random.choice(["Equivalent", "Uplift", "Not Applicable"],
                                              p=[0.6, 0.3, 0.1])

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
                "RequirementID": f'REQ-{i:03d}' if "RequirementID" not in self.requirements_df.columns
                else self.requirements_df.iloc[i].get("RequirementID", f'REQ-{i:03d}'),
                "Classification": classification,
                "Confidence": confidence,
                "Reason": reason
            })

            # Simulate processing time
            if i % 10 == 0:
                time.sleep(0.05)

        # Create classification dataframe
        classification_df = pd.DataFrame(simulated_results)
        self.classification_logs = classification_df

        # Summary stats
        eq_count = (classification_df["Classification"] == "Equivalent").sum()
        up_count = (classification_df["Classification"] == "Uplift").sum()
        na_count = (classification_df["Classification"] == "Not Applicable").sum()

        summary = f"""
        Classification Summary:
        - Total: {total}
        - Equivalent: {eq_count} ({eq_count/total*100:.1f}%)
        - Uplift: {up_count} ({up_count/total*100:.1f}%)
        - Not Applicable: {na_count} ({na_count/total*100:.1f}%)
        """

        return summary, classification_df.head(10).to_html()

    def generate_mapping_table(self):
        """Generate mapping table for review"""
        if self.requirements_df is None or self.classification_logs.empty:
            return "Error: Classification not completed", None

        # Create mapping table
        req_df = self.requirements_df.copy()
        class_df = self.classification_logs.copy()

        # Join datasets
        if "RequirementID" in req_df.columns and "RequirementID" in class_df.columns:
            joined_df = pd.merge(req_df, class_df, on="RequirementID", how="left")
        else:
            # If no common ID, use the index
            class_df_indexed = class_df.set_index("ReqIndex")
            # Add classification info to original dataframe
            for col in ["Classification", "Confidence", "Reason"]:
                req_df[col] = req_df.index.map(lambda x: class_df_indexed.loc[x, col] if x in class_df_indexed.index else None)
            joined_df = req_df

        # Add recommended action
        def recommended_action(row):
            classification = row.get("Classification")
            if classification == "Equivalent":
                control_id = f"CTL-{np.random.randint(100, 999)}"
                return f"Map to existing control {control_id}"
            elif classification == "Uplift":
                return "Create new control requirement"
            else:
                return "No action required"

        joined_df["RecommendedAction"] = joined_df.apply(recommended_action, axis=1)
        joined_df["Comments"] = ""
        joined_df["MappedControlID"] = ""
        joined_df["ApprovalStatus"] = "Pending"

        # Save mapping table
        self.mapping_df = joined_df

        # Exportable data (first few rows) as CSV
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        joined_df.to_csv(temp_csv.name, index=False)
        temp_csv.close()

        return joined_df.head(10).to_html(), temp_csv.name

    def finalize_mapping(self, edited_data):
        """Process the edited mapping data"""
        if edited_data is None:
            return "No edited data provided", None

        # In a real app, we would parse the edited data
        # For demo purposes, we'll just return the mapping data

        # Generate download file
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.mapping_df.to_csv(temp_csv.name, index=False)
        temp_csv.close()

        return "Mapping finalized successfully!", temp_csv.name

    def reset_app(self):
        """Reset the application state"""
        self.requirements_df = None
        self.kb_df = None
        self.classification_logs = pd.DataFrame()
        self.mapping_df = pd.DataFrame()
        self.settings = {}
        self.current_step = 0

        return (
            None,  # req_preview
            "Application reset. Please start from Step 1.",  # req_status
            None,  # kb_preview
            "Application reset. Please start from Step 1.",  # kb_status
            "B1",  # framework
            "AI-Assisted",  # method
            0.7,  # threshold
            "Settings reset.",  # settings_status
            "Classification reset.",  # classification_status
            None,  # classification_results
            None,  # mapping_preview
            None,  # mapping_file
            "",    # mapping_comments
            "Mapping reset.",  # finalize_status
            None,  # download_file
        )

# Create the Gradio interface
def create_interface():
    app = RegMappingApp()

    with gr.Blocks(title="Regulatory Mapping Tool", css="""
        .container { max-width: 1200px; margin: auto; }
        .header { margin-bottom: 20px; }
        .footer { margin-top: 30px; text-align: center; }
        .step-container { border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .success-msg { color: green; }
        .error-msg { color: red; }
        .nav-buttons { display: flex; justify-content: space-between; margin-top: 20px; }
        .dark-button { background-color: #1E1E2E; color: white; }
        table { width: 100%; border-collapse: collapse; }
        th { background-color: #1E1E2E; color: white; text-align: left; padding: 8px; }
        td { padding: 8px; border-bottom: 1px solid #ddd; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    """) as interface:
        gr.Markdown("# Regulatory Mapping Tool")

        # Step navigation tabs
        with gr.Tabs() as tabs:
            # Step 1: Upload Data
            with gr.TabItem("Step 1: Upload Data"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Regulatory Requirements")
                        req_file = gr.File(label="Upload Requirements (XLSX, CSV)", file_types=[".xlsx", ".xls", ".csv"])
                        load_demo_btn = gr.Button("Load Demo Data", variant="secondary")
                        req_preview = gr.HTML(label="Preview")
                        req_status = gr.Textbox(label="Status")

                    with gr.Column():
                        gr.Markdown("### Knowledge Base (Optional)")
                        kb_file = gr.File(label="Upload Knowledge Base (XLSX, CSV)", file_types=[".xlsx", ".xls", ".csv"])
                        load_kb_btn = gr.Button("Load Demo KB", variant="secondary")
                        kb_preview = gr.HTML(label="Preview")
                        kb_status = gr.Textbox(label="Status")

                next_step1 = gr.Button("Next: Review Data", variant="primary")

            # Step 2: Configure
            with gr.TabItem("Step 2: Review & Configure"):
                gr.Markdown("### Classification Settings")

                with gr.Row():
                    framework = gr.Dropdown(
                        ["B1", "Program-Level", "Local Definition", "GDPR", "PCI DSS", "ISO 27001", "SOC 2", "Other"],
                        label="Regulatory Framework",
                        value="B1"
                    )

                    method = gr.Dropdown(
                        ["AI-Assisted", "Rule-Based", "Manual"],
                        label="Classification Method",
                        value="AI-Assisted"
                    )

                threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Confidence Threshold"
                )

                save_settings_btn = gr.Button("Save Settings")
                settings_status = gr.Textbox(label="Settings Status")

                with gr.Row():
                    back_step2 = gr.Button("← Back", variant="secondary")
                    next_step2 = gr.Button("Start Classification →", variant="primary")

            # Step 3: Classification
            with gr.TabItem("Step 3: Classification"):
                start_classify_btn = gr.Button("Start Classification Process", variant="primary")
                classification_status = gr.Textbox(label="Classification Status")
                classification_results = gr.HTML(label="Classification Results")

                with gr.Row():
                    back_step3 = gr.Button("← Back", variant="secondary")
                    next_step3 = gr.Button("Generate Mappings →", variant="primary")

            # Step 4: Mapping Approval
            with gr.TabItem("Step 4: Mapping Approval"):
                generate_mapping_btn = gr.Button("Generate Mapping Table", variant="primary")
                mapping_preview = gr.HTML(label="Mapping Table")
                mapping_file = gr.File(label="Mapping Data", visible=False)

                # In a real app, we would use a DataFrame editor component
                # For this demo, we'll simulate with a textbox for comments
                mapping_comments = gr.Textbox(
                    label="Mapping Comments & Edits",
                    placeholder="Enter any comments or edits for the mapping table..."
                )

                with gr.Row():
                    back_step4 = gr.Button("← Back", variant="secondary")
                    next_step4 = gr.Button("Finalize and Export →", variant="primary")

            # Step 5: Export Results
            with gr.TabItem("Step 5: Export Results"):
                gr.Markdown("### Final Results")
                finalize_btn = gr.Button("Finalize Mapping", variant="primary")
                finalize_status = gr.Textbox(label="Status")
                download_file = gr.File(label="Download Mapping")

                with gr.Row():
                    back_step5 = gr.Button("← Back", variant="secondary")
                    reset_btn = gr.Button("Start New Project", variant="secondary")

        # Connect components with functions
        req_file.upload(app.upload_requirements, req_file, [req_preview, req_status])
        load_demo_btn.click(app.load_demo_data, None, [req_preview, req_status])

        kb_file.upload(app.upload_kb, kb_file, [kb_preview, kb_status])
        load_kb_btn.click(app.load_demo_kb, None, [kb_preview, kb_status])

        save_settings_btn.click(app.save_settings, [framework, method, threshold], settings_status)

        start_classify_btn.click(app.run_classification, None, [classification_status, classification_results])

        generate_mapping_btn.click(app.generate_mapping_table, None, [mapping_preview, mapping_file])

        finalize_btn.click(app.finalize_mapping, mapping_comments, [finalize_status, download_file])

        # Connect reset button with the reset function
        reset_btn.click(
            app.reset_app,
            None,
            [
                req_preview, req_status, kb_preview, kb_status,
                framework, method, threshold, settings_status,
                classification_status, classification_results,
                mapping_preview, mapping_file, mapping_comments,
                finalize_status, download_file
            ]
        )

        # Navigation between tabs
        next_step1.click(lambda: gr.update(selected=1), None, tabs)
        back_step2.click(lambda: gr.update(selected=0), None, tabs)
        next_step2.click(lambda: gr.update(selected=2), None, tabs)
        back_step3.click(lambda: gr.update(selected=1), None, tabs)
        next_step3.click(lambda: gr.update(selected=3), None, tabs)
        back_step4.click(lambda: gr.update(selected=2), None, tabs)
        next_step4.click(lambda: gr.update(selected=4), None, tabs)
        back_step5.click(lambda: gr.update(selected=3), None, tabs)
        reset_btn.click(lambda: gr.update(selected=0), None, tabs)

    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
