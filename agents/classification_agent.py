# File: agents/classification_agent.py

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import logging
import traceback
import json
import time
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('classification_agent')

# Define Pydantic model for structured output
class RequirementClassification(BaseModel):
    """Classification result for a regulatory requirement."""
    type: str = Field(description="Type of requirement: 'CDD' (Customer Due Diligence) or 'Program-Level'")
    confidence: float = Field(description="Confidence score between 0 and 1")
    explanation: str = Field(description="Explanation for the classification")

    @validator('type')
    def type_must_be_valid(cls, v):
        valid_types = ['CDD', 'Program-Level']
        if v not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
        return v

    @validator('confidence')
    def confidence_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

class ClassificationAgent:
    """
    Agent responsible for classifying regulatory requirements as
    either CDD (Customer Due Diligence) or Program-Level.

    This agent:
    1. Takes a regulatory requirement and relevant knowledge base documents
    2. Uses an LLM to deeply analyze and understand the requirement
    3. Classifies the requirement as CDD or Program-Level with explanation
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4o"):
        """
        Initialize the classification agent.

        Args:
            openai_api_key: API key for OpenAI (optional, will use environment variable if not provided)
            model_name: Name of the LLM model to use for classification
        """
        self.model_name = model_name
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

        # Set up LLM
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=self.openai_api_key,
            temperature=0.1,  # Low temperature for consistent outputs
        )

        # Set up output parser
        self.parser = PydanticOutputParser(pydantic_object=RequirementClassification)

        logger.info(f"Initialized ClassificationAgent with model {model_name}")

    def classify_requirement(self,
                             requirement: Union[str, Document],
                             knowledge_base_docs: List[Document]) -> Dict[str, Any]:
        """
        Classify a regulatory requirement as CDD or Program-Level.

        Args:
            requirement: The regulatory requirement text or Document
            knowledge_base_docs: Relevant knowledge base documents

        Returns:
            Classification result as a dictionary
        """
        # Get requirement text
        if isinstance(requirement, Document):
            requirement_text = requirement.page_content
            requirement_metadata = requirement.metadata
        else:
            requirement_text = requirement
            requirement_metadata = {}

        logger.info(f"Classifying requirement: {requirement_text[:100]}...")

        # Create context from knowledge base documents
        kb_context = self._prepare_kb_context(knowledge_base_docs)

        # Create the prompt for the LLM
        prompt = self._create_classification_prompt(
            requirement_text,
            kb_context,
            self.parser.get_format_instructions()
        )

        try:
            # Call the LLM for classification
            start_time = time.time()
            response = self.llm.invoke(prompt)
            elapsed_time = time.time() - start_time

            logger.info(f"LLM responded in {elapsed_time:.2f} seconds")

            # Parse the response
            classification_str = response.content
            classification = self.parser.parse(classification_str)

            # Convert to dictionary
            result = classification.dict()

            # Add the requirement and context to the result
            result["requirement"] = {
                "text": requirement_text,
                "metadata": requirement_metadata
            }
            result["knowledge_base_docs"] = [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                } for doc in knowledge_base_docs
            ]

            logger.info(f"Classification result: Type={result['type']}, Confidence={result['confidence']}")

            return result

        except Exception as e:
            error_msg = f"Error during classification: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            # Return error result
            return {
                "error": str(e),
                "requirement": {
                    "text": requirement_text,
                    "metadata": requirement_metadata
                },
                "type": "Unknown",
                "confidence": 0.0,
                "explanation": f"Error during classification: {str(e)}"
            }

    def _prepare_kb_context(self, kb_docs: List[Document]) -> str:
        """
        Prepare context from knowledge base documents.

        Args:
            kb_docs: List of knowledge base documents

        Returns:
            Formatted context string
        """
        if not kb_docs:
            return "No knowledge base documents provided."

        context_parts = []

        for i, doc in enumerate(kb_docs):
            metadata = doc.metadata

            # Format document identifiers
            file_name = metadata.get("file_name", "Unknown")
            page = metadata.get("page", "Unknown")

            # Calculate similarity score if available
            similarity = metadata.get("similarity_score", None)
            similarity_str = f" (Similarity: {similarity:.2f})" if similarity is not None else ""

            # Format the document
            context_parts.append(
                f"--- KNOWLEDGE BASE DOCUMENT {i+1}: {file_name}, Page {page}{similarity_str} ---\n"
                f"CONTENT:\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)

    def _create_classification_prompt(self,
                                      requirement_text: str,
                                      kb_context: str,
                                      format_instructions: str) -> List[Any]:
        """
        Create a classification prompt for the LLM.

        Args:
            requirement_text: The text of the requirement to classify
            kb_context: Formatted context from knowledge base documents
            format_instructions: Format instructions for the output

        Returns:
            Formatted prompt messages
        """
        system_message_content = """You are an expert in regulatory compliance and KYC (Know Your Customer) requirements. 
        Your task is to carefully analyze a regulatory requirement and determine if it's related to 
        Customer Due Diligence (CDD) or if it's a Program-Level requirement.
        
        Key characteristics of each type:
        
        1. Customer Due Diligence (CDD) requirements:
           - Focus on processes for identifying and verifying customers
           - Include customer identification, verification, and risk assessment
           - Deal with collecting and verifying customer information
           - Cover enhanced due diligence for high-risk customers
           - Address ongoing monitoring of customer accounts and transactions
           - Include beneficial ownership identification for legal entities
        
        2. Program-Level requirements:
           - Focus on the overall compliance program framework
           - Include governance, policies, procedures, and controls
           - Cover staff training and awareness
           - Address record-keeping, reporting, and documentation
           - Include risk assessment at the program or institutional level
           - Deal with independent testing and audit functions
           - Cover organizational structure and responsibilities
        
        You must think deeply about the requirement, analyze its content and context, and provide a well-reasoned 
        classification. Support your conclusion with detailed explanations of why the requirement belongs 
        to the chosen category.
        """

        system_message = SystemMessagePromptTemplate.from_template(system_message_content)

        human_message_template = """
        ## REGULATORY REQUIREMENT TO CLASSIFY:
        {requirement_text}
        
        ## RELEVANT KNOWLEDGE BASE DOCUMENTS:
        {kb_context}
        
        Based on the requirement text and the knowledge base documents, carefully analyze whether this requirement 
        is related to Customer Due Diligence (CDD) or is a Program-Level requirement.
        
        Think step by step:
        1. What is the main focus of this requirement?
        2. Does it deal primarily with customer identification and verification processes?
        3. Or does it focus more on program elements, governance, and overall controls?
        4. What specific elements in the requirement point to either CDD or Program-Level classification?
        5. How confident are you in this classification?
        
        Format your response according to the following JSON schema:
        {format_instructions}
        """

        human_message = HumanMessagePromptTemplate.from_template(human_message_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        return chat_prompt.format_prompt(
            requirement_text=requirement_text,
            kb_context=kb_context,
            format_instructions=format_instructions
        ).to_messages()

    def batch_classify(self,
                       requirements: List[Union[str, Document]],
                       get_kb_docs_func: callable,
                       batch_size: int = 5,
                       progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Classify multiple requirements in batches.

        Args:
            requirements: List of requirements (text or Documents)
            get_kb_docs_func: Function that returns knowledge base documents for a requirement
                              Should take a requirement and return a list of relevant KB documents
            batch_size: Number of requirements to process in each batch
            progress_callback: Optional callback function for progress updates (receives float 0-1)

        Returns:
            List of classification results
        """
        if not requirements:
            return []

        results = []
        total = len(requirements)

        for i in range(0, total, batch_size):
            batch = requirements[i:i+batch_size]
            batch_results = []

            for req in batch:
                # Get relevant knowledge base documents
                kb_docs = get_kb_docs_func(req)

                # Classify requirement
                result = self.classify_requirement(req, kb_docs)
                batch_results.append(result)

            results.extend(batch_results)

            # Update progress
            if progress_callback:
                progress = min(1.0, (i + len(batch)) / total)
                progress_callback(progress)

        return results
