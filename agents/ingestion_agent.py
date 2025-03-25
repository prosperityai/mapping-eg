
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import pandas as pd
import numpy as np
from langchain.schema import Document
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_transformers import LongContextReorder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import tempfile
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ingestion_agent')

class IngestionAgent:
    """
    Agent responsible for reading regulatory obligations and KYC policies from Excel files,
    and structuring each row as a Document (text + metadata).

    This agent handles:
    1. Loading Excel files (both regulatory requirements and KYC policies)
    2. Converting Excel data into structured document objects
    3. Error handling and validation for all ingestion operations
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the ingestion agent.

        Args:
            chunk_size: The size of text chunks for large documents (default: 1000)
            chunk_overlap: The overlap between chunks (default: 200)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        logger.info(f"Initialized IngestionAgent with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def validate_excel_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validates that the given file is a valid Excel file.

        Args:
            file_path: Path to the Excel file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"

        if not file_path.endswith(('.xlsx', '.xls')):
            return False, f"File is not an Excel file: {file_path}"

        try:
            # Try to read the file to validate it's a proper Excel file
            _ = pd.read_excel(file_path, nrows=1)
            return True, ""
        except Exception as e:
            error_msg = f"Invalid Excel file: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def save_uploaded_file(self, uploaded_file) -> Tuple[str, Optional[str]]:
        """
        Saves an uploaded file to a temporary location and returns the path.

        Args:
            uploaded_file: The uploaded file object (from Streamlit)

        Returns:
            Tuple of (path, error_message)
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            logger.info(f"Saved uploaded file to temporary location: {temp_path}")
            return temp_path, None
        except Exception as e:
            error_msg = f"Error saving uploaded file: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return "", error_msg

    def ingest_regulations(self, file_path: str) -> Tuple[List[Document], Optional[str]]:
        """
        Ingests an Excel file containing regulatory obligations and converts
        each row into a Document (text + metadata).

        Args:
            file_path: Path to the Excel file with regulatory obligations

        Returns:
            Tuple of (list of documents, error_message if any)
        """
        logger.info(f"Starting ingestion of regulatory file: {file_path}")

        # Validate the file
        is_valid, error_msg = self.validate_excel_file(file_path)
        if not is_valid:
            return [], error_msg

        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")

            if df.empty:
                return [], "Excel file is empty"

            # Determine the column containing the requirement text
            # Typically this is the first column or a column with a specific name
            text_column = df.columns[0]  # Default to first column

            # Check if there's a column that might contain the requirement text
            likely_columns = [col for col in df.columns if 'requirement' in col.lower()]
            if likely_columns:
                text_column = likely_columns[0]

            logger.info(f"Using column '{text_column}' as the requirement text source")

            # Convert each row to a document
            documents = []
            for idx, row in df.iterrows():
                try:
                    # Extract the requirement text
                    requirement_text = str(row[text_column])
                    if pd.isna(requirement_text) or not requirement_text.strip():
                        logger.warning(f"Empty requirement text in row {idx+1}, skipping")
                        continue

                    # Create the metadata
                    metadata = {
                        "source": "regulatory_requirement",
                        "row_index": idx,
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                    }

                    # Add all other columns as metadata
                    for col, val in row.items():
                        # Convert to string if not null, otherwise use empty string
                        if pd.notna(val):
                            if isinstance(val, (int, float, bool)):
                                metadata[str(col)] = str(val)
                            else:
                                metadata[str(col)] = val
                        else:
                            metadata[str(col)] = ""

                    # Create the document
                    doc = Document(
                        page_content=requirement_text,
                        metadata=metadata
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    # Continue with next row instead of failing the entire process

            if not documents:
                return [], "No valid requirements found in the Excel file"

            logger.info(f"Successfully created {len(documents)} documents from regulatory requirements")
            return documents, None

        except Exception as e:
            error_msg = f"Error processing Excel file: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [], error_msg

    def ingest_kyc_policy(self, file_path: str) -> Tuple[List[Document], Optional[str]]:
        """
        Ingests the B1 Mastersheet (KYC policy) Excel file and converts each row
        into a Document (text + metadata).

        Args:
            file_path: Path to the Excel file with KYC policy

        Returns:
            Tuple of (list of documents, error_message if any)
        """
        logger.info(f"Starting ingestion of KYC policy file: {file_path}")

        # Validate the file
        is_valid, error_msg = self.validate_excel_file(file_path)
        if not is_valid:
            return [], error_msg

        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            logger.info(f"Successfully loaded {len(df)} rows from KYC policy file")

            if df.empty:
                return [], "KYC policy Excel file is empty"

            # Expected columns for KYC policy (B1 Mastersheet)
            expected_columns = [
                "Master Sheet", "Minor Sheet", "Section", "Title",
                "Requirement", "Rule", "Guidance", "KYC Standards Impact"
            ]

            # Check if at least some of the expected columns exist
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if len(missing_columns) > len(expected_columns) // 2:
                logger.warning(f"Many expected columns are missing: {missing_columns}")

            # Convert each row to a document
            documents = []
            for idx, row in df.iterrows():
                try:
                    # Combine relevant fields to create a rich text representation
                    text_parts = []

                    # Add Master Sheet, Minor Sheet, and Section info
                    master_sheet = str(row.get("Master Sheet", "")) if pd.notna(row.get("Master Sheet", "")) else ""
                    minor_sheet = str(row.get("Minor Sheet", "")) if pd.notna(row.get("Minor Sheet", "")) else ""
                    section = str(row.get("Section", "")) if pd.notna(row.get("Section", "")) else ""

                    if master_sheet and minor_sheet:
                        text_parts.append(f"Section: {master_sheet}-{minor_sheet} {section}")

                    # Add Title
                    if "Title" in row and pd.notna(row["Title"]):
                        text_parts.append(f"Title: {row['Title']}")

                    # Add Rule
                    if "Rule" in row and pd.notna(row["Rule"]):
                        text_parts.append(f"Rule: {row['Rule']}")

                    # Add Guidance
                    if "Guidance" in row and pd.notna(row["Guidance"]):
                        text_parts.append(f"Guidance: {row['Guidance']}")

                    # Add Impact
                    if "KYC Standards Impact" in row and pd.notna(row["KYC Standards Impact"]):
                        text_parts.append(f"Impact: {row['KYC Standards Impact']}")

                    # Skip if no meaningful text was extracted
                    if not text_parts:
                        logger.warning(f"No meaningful text extracted from row {idx+1}, skipping")
                        continue

                    # Create the combined text
                    text = "\n".join(text_parts)

                    # Create the metadata
                    metadata = {
                        "source": "kyc_policy",
                        "row_index": idx,
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                    }

                    # Add specific metadata fields
                    metadata["master_sheet"] = master_sheet
                    metadata["minor_sheet"] = minor_sheet
                    metadata["section"] = section
                    metadata["title"] = str(row.get("Title", "")) if pd.notna(row.get("Title", "")) else ""

                    # Add all other columns as metadata
                    for col, val in row.items():
                        col_name = str(col)
                        # Don't overwrite already added metadata
                        if col_name not in metadata and pd.notna(val):
                            if isinstance(val, (int, float, bool)):
                                metadata[col_name] = str(val)
                            else:
                                metadata[col_name] = val
                        elif col_name not in metadata:
                            metadata[col_name] = ""

                    # Create the document
                    doc = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error processing policy row {idx}: {str(e)}")
                    # Continue with next row

            if not documents:
                return [], "No valid policy entries found in the KYC policy Excel file"

            logger.info(f"Successfully created {len(documents)} documents from KYC policy")
            return documents, None

        except Exception as e:
            error_msg = f"Error processing KYC policy Excel file: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [], error_msg

    def ingest_full_policy_document(self, file_path: str) -> Tuple[List[Document], Optional[str]]:
        """
        Ingests a full KYC policy document (not row-based, but the actual long document).
        This method is useful for ingesting large policy documents that need to be chunked.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (list of documents, error_message if any)
        """
        logger.info(f"Starting ingestion of full policy document: {file_path}")

        try:
            # Use Langchain's UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(file_path)
            documents = loader.load()

            if not documents:
                return [], "No content extracted from the document"

            logger.info(f"Loaded {len(documents)} sections from document, chunking...")

            # Process with text splitter for large documents
            chunked_documents = self.text_splitter.split_documents(documents)

            # Reorder for better context
            reordered = LongContextReorder().transform_documents(chunked_documents)

            logger.info(f"Successfully created {len(reordered)} chunks from policy document")
            return reordered, None

        except Exception as e:
            error_msg = f"Error processing full policy document: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [], error_msg

    def process_excel_file(self, file_path: str, file_type: str = "regulations") -> Tuple[List[Document], Optional[str]]:
        """
        Process an Excel file based on its type.

        Args:
            file_path: Path to the Excel file
            file_type: Type of file ("regulations" or "kyc_policy")

        Returns:
            Tuple of (list of documents, error_message if any)
        """
        if file_type.lower() == "regulations":
            return self.ingest_regulations(file_path)
        elif file_type.lower() == "kyc_policy":
            return self.ingest_kyc_policy(file_path)
        else:
            return [], f"Unsupported file type: {file_type}"
