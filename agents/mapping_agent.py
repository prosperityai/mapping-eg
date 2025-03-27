# File: agents/mapping_agent.py

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
logger = logging.getLogger('mapping_agent')

# Define Pydantic models for structured output
class PolicyReference(BaseModel):
    """Reference to a specific KYC policy."""
    master_sheet: str = Field(description="Master Sheet identifier")
    minor_sheet: str = Field(description="Minor Sheet identifier", default="")
    section: str = Field(description="Section identifier", default="")
    title: str = Field(description="Title of the policy", default="")
    relevance_score: float = Field(description="Score between 0 and 1 indicating relevance to the requirement")
    explanation: str = Field(description="Explanation of how this policy relates to the requirement")

    @validator('relevance_score')
    def score_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Relevance score must be between 0 and 1")
        return v

class MappingResult(BaseModel):
    """Mapping result for a regulatory requirement."""
    coverage: str = Field(description="Coverage level: 'Equivalent', 'Partial Uplift', or 'Full Uplift'")
    mapped_policies: List[PolicyReference] = Field(description="List of mapped policy references")
    coverage_explanation: str = Field(description="Explanation for the coverage determination")
    gap_analysis: str = Field(description="Analysis of any gaps between requirement and policies")
    confidence: float = Field(description="Confidence score between 0 and 1")

    @validator('coverage')
    def coverage_must_be_valid(cls, v):
        valid_coverages = ['Equivalent', 'Partial Uplift', 'Full Uplift']
        if v not in valid_coverages:
            raise ValueError(f"Coverage must be one of {valid_coverages}")
        return v

    @validator('confidence')
    def confidence_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

class MappingAgent:
    """
    Agent responsible for mapping classified regulatory requirements to KYC policies.

    This agent:
    1. Takes a classified regulatory requirement and relevant KYC policy documents
    2. Analyzes the requirement against policies to determine coverage level
    3. Identifies specific policies that address the requirement
    4. Analyzes gaps between the requirement and existing policies
    5. Provides detailed explanations and confidence scores
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4o"):
        """
        Initialize the mapping agent.

        Args:
            openai_api_key: API key for OpenAI (optional, will use environment variable if not provided)
            model_name: Name of the LLM model to use for mapping
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
        self.parser = PydanticOutputParser(pydantic_object=MappingResult)

        logger.info(f"Initialized MappingAgent with model {model_name}")

    def map_requirement(self,
                        requirement: Dict[str, Any],
                        kyc_policies: List[Document]) -> Dict[str, Any]:
        """
        Map a classified regulatory requirement to KYC policies.

        Args:
            requirement: Dict containing requirement text, metadata, and classification
            kyc_policies: List of KYC policy documents to map against

        Returns:
            Mapping result as a dictionary
        """
        # Extract requirement information
        requirement_text = requirement.get("requirement", {}).get("text", "")
        requirement_metadata = requirement.get("requirement", {}).get("metadata", {})
        requirement_type = requirement.get("classification", {}).get("type", "Unknown")
        requirement_explanation = requirement.get("classification", {}).get("explanation", "")

        logger.info(f"Mapping requirement of type '{requirement_type}': {requirement_text[:100]}...")

        # Create context from KYC policy documents
        kyc_context = self._prepare_kyc_context(kyc_policies)

        # Create the prompt for the LLM
        prompt = self._create_mapping_prompt(
            requirement_text,
            requirement_type,
            requirement_explanation,
            kyc_context,
            self.parser.get_format_instructions()
        )

        try:
            # Call the LLM for mapping
            start_time = time.time()
            response = self.llm.invoke(prompt)
            elapsed_time = time.time() - start_time

            logger.info(f"LLM responded in {elapsed_time:.2f} seconds")

            # Parse the response
            mapping_str = response.content
            mapping_result = self.parser.parse(mapping_str)

            # Convert to dictionary
            result = mapping_result.dict()

            # Add the requirement information to the result
            result["requirement"] = {
                "text": requirement_text,
                "metadata": requirement_metadata,
                "type": requirement_type,
                "explanation": requirement_explanation
            }

            logger.info(f"Mapping result: Coverage={result['coverage']}, Confidence={result['confidence']}")

            return result

        except Exception as e:
            error_msg = f"Error during mapping: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            # Return error result
            return {
                "error": str(e),
                "requirement": {
                    "text": requirement_text,
                    "metadata": requirement_metadata,
                    "type": requirement_type,
                    "explanation": requirement_explanation
                },
                "coverage": "Unknown",
                "mapped_policies": [],
                "coverage_explanation": f"Error during mapping: {str(e)}",
                "gap_analysis": "",
                "confidence": 0.0
            }

    def _prepare_kyc_context(self, kyc_docs: List[Document]) -> str:
        """
        Prepare context from KYC policy documents.

        Args:
            kyc_docs: List of KYC policy documents

        Returns:
            Formatted context string
        """
        if not kyc_docs:
            return "No KYC policy documents provided."

        context_parts = []

        for i, doc in enumerate(kyc_docs):
            metadata = doc.metadata

            # Format policy identifiers for B1 Mastersheet
            master_sheet = metadata.get("master_sheet", "Unknown")
            minor_sheet = metadata.get("minor_sheet", "Unknown")
            section = metadata.get("section", "")
            title = metadata.get("title", "Unknown")

            # Calculate similarity score if available
            similarity = metadata.get("similarity_score", None)
            similarity_str = f" (Similarity: {similarity:.2f})" if similarity is not None else ""

            # Format the document
            context_parts.append(
                f"--- KYC POLICY {i+1}: Master Sheet: {master_sheet}, Minor Sheet: {minor_sheet}, Section: {section}, Title: {title}{similarity_str} ---\n"
                f"CONTENT:\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)

    def _create_mapping_prompt(self,
                               requirement_text: str,
                               requirement_type: str,
                               requirement_explanation: str,
                               kyc_context: str,
                               format_instructions: str) -> List[Any]:
        """
        Create a mapping prompt for the LLM.

        Args:
            requirement_text: The text of the requirement to map
            requirement_type: The type of the requirement (CDD or Program-Level)
            requirement_explanation: Explanation of the requirement classification
            kyc_context: Formatted context from KYC policy documents
            format_instructions: Format instructions for the output

        Returns:
            Formatted prompt messages
        """
        system_message_content = """You are an expert in regulatory compliance and KYC (Know Your Customer) mapping. 
        Your task is to map a regulatory requirement to relevant KYC policies, analyzing how well the existing policies 
        cover the requirement.
        
        You will determine:
        1. Overall coverage level (Equivalent, Partial Uplift, Full Uplift)
        2. Specific KYC policies that map to the requirement
        3. Gaps between the requirement and existing policies
        
        Coverage levels are defined as:
        - "Equivalent": Existing policies fully address the requirement with no gaps
        - "Partial Uplift": Existing policies partially address the requirement, but some gaps exist
        - "Full Uplift": Existing policies don't adequately address the requirement, significant gaps exist
        
        For each mapping:
        1. Think deeply about how closely each policy matches the requirement's specific demands
        2. Consider both explicit matches and implicit coverage
        3. Analyze the comprehensiveness of the policy in addressing all aspects of the requirement
        4. Identify specific gaps or areas where policies could be enhanced
        5. Assess your confidence in the mapping, noting areas of uncertainty
        
        Be thorough and precise in your analysis. The quality of your mapping will determine how effectively 
        an organization can address regulatory obligations.
        """

        system_message = SystemMessagePromptTemplate.from_template(system_message_content)

        human_message_template = """
        ## REGULATORY REQUIREMENT TO MAP:
        {requirement_text}
        
        ## REQUIREMENT CLASSIFICATION:
        Type: {requirement_type}
        Classification Explanation: {requirement_explanation}
        
        ## RELEVANT KYC POLICIES:
        {kyc_context}
        
        Based on the regulatory requirement and the provided KYC policies, I need you to:

        1. Determine the overall coverage level: Is this requirement Equivalent (fully covered), Partial Uplift (partially covered with gaps), or Full Uplift (inadequately covered)?
        
        2. Identify specific KYC policies that address this requirement, with references to Master Sheet, Minor Sheet, etc.
        
        3. For each mapped policy, explain exactly how it relates to the requirement and assign a relevance score.
        
        4. Provide a gap analysis - what aspects of the requirement are not addressed by existing policies?
        
        5. Explain your coverage determination with detailed reasoning.
        
        Think step by step:
        - First, understand what the requirement is demanding in specific, concrete terms
        - Then, evaluate each policy for relevance to those specific demands
        - Assess how comprehensively the policies address all aspects of the requirement
        - Identify any specific gaps or areas where policies could be enhanced
        - Determine an overall coverage level based on your analysis
        
        Format your response according to the following JSON schema:
        {format_instructions}
        """

        human_message = HumanMessagePromptTemplate.from_template(human_message_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        return chat_prompt.format_prompt(
            requirement_text=requirement_text,
            requirement_type=requirement_type,
            requirement_explanation=requirement_explanation,
            kyc_context=kyc_context,
            format_instructions=format_instructions
        ).to_messages()

    def batch_map(self,
                  classified_requirements: List[Dict[str, Any]],
                  get_kyc_docs_func: callable,
                  batch_size: int = 3,
                  progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Map multiple classified requirements in batches.

        Args:
            classified_requirements: List of classified requirements
            get_kyc_docs_func: Function that returns KYC policy documents for a requirement
                               Should take a requirement and return a list of relevant policy documents
            batch_size: Number of requirements to process in each batch
            progress_callback: Optional callback function for progress updates (receives float 0-1)

        Returns:
            List of mapping results
        """
        if not classified_requirements:
            return []

        results = []
        total = len(classified_requirements)

        for i in range(0, total, batch_size):
            batch = classified_requirements[i:i+batch_size]
            batch_results = []

            for req in batch:
                # Get relevant KYC policy documents
                kyc_docs = get_kyc_docs_func(req)

                # Map requirement
                result = self.map_requirement(req, kyc_docs)
                batch_results.append(result)

            results.extend(batch_results)

            # Update progress
            if progress_callback:
                progress = min(1.0, (i + len(batch)) / total)
                progress_callback(progress)

        return results
