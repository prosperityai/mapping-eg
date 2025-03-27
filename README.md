# Regulatory Mapping Tool

A comprehensive solution for mapping regulatory obligations to internal KYC policies using large language models and semantic search.

## Overview

The Regulatory Mapping Tool automates the process of analyzing regulatory requirements and mapping them to existing internal KYC policies. It uses vector embeddings and large language models to find semantic similarities and provide intelligent classification and mapping recommendations.

## Features

- **Document Ingestion**: Process regulatory requirements, KYC policies, and knowledge base documents  
- **Vector Embeddings**: Create searchable vector representations of documents using OpenAI embeddings  
- **Semantic Search**: Find similar policies and knowledge base documents for each requirement  
- **Intelligent Classification**: Classify requirements as CDD or Program-Level and determine coverage level  
- **Mapping Recommendations**: Generate mapping recommendations with explanations  
- **Export Functionality**: Export mapping results to Excel for further analysis and documentation

## System Architecture

The system uses an agent-based architecture with the following components:

- **Ingestion Agent**: Handles document loading and processing  
- **Embedding Agent**: Creates vector embeddings for semantic search  
- **Classification Agent**: Classifies requirements and generates mapping recommendations  

## Installation

### Prerequisites

- Python 3.10+  
- Streamlit  
- OpenAI API key  

### Setup

1. Clone the repository:
   ```bash
   git clone 
   cd regulatory-mapping
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API key:
   ```bash
   
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run reg_map.py
   ```
2. Follow the step-by-step workflow in the web interface:
    - Upload regulatory requirements (Excel file)
    - Upload KYC policy document (B1 Mastersheet)
    - Upload knowledge base documents (PDF files)
    - Review and approve document processing
    - Build vector indices for semantic search
    - Classify 
    - map regulatory requirements
    - Review and export mapping results

## Project Structure

```plaintext
reg_map/
├── agents/
│   ├── __init__.py
│   ├── ingestion_agent.py
│   ├── embedding_agent.py
│   └── classification_agent.py
    ---mapping_agent.py

├── data/
│   ├── documents/
│   ├── knowledge_base/
│   ├── temp/
│   └── vectors/
├── reg_map.py
└── requirements.txt
```

## Troubleshooting

- **OpenAI API errors**: Ensure your API key has the correct permissions and sufficient quota
- **Memory issues with large documents**: Adjust chunk size and overlap in the `IngestionAgent`


```
