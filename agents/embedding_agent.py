from typing import List, Dict, Any, Optional, Union, Tuple
import os
import numpy as np
import pandas as pd
import tempfile
import logging
import traceback
import pickle
import time
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedding_agent')

class EmbeddingAgent:
    """
    Agent responsible for embedding documents and creating/updating vector stores.

    This agent handles:
    1. Embedding documents using OpenAI embeddings
    2. Creating and managing FAISS vector stores
    3. Persisting vector stores for later use
    4. Combining and updating existing vector stores
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 model_name: str = "text-embedding-ada-002"):
        """
        Initialize the embedding agent.

        Args:
            openai_api_key: API key for OpenAI (optional, will use environment variable if not provided)
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

        # Set up OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model=model_name
        )

        # Dictionary to store vector stores by name
        self.vector_stores = {}

        logger.info(f"Initialized EmbeddingAgent with model {model_name}")

    def create_vector_store(self,
                            documents: List[Document],
                            store_name: str,
                            persist_directory: Optional[str] = None) -> FAISS:
        """
        Create a new FAISS vector store from documents.

        Args:
            documents: List of documents to embed
            store_name: Name for this vector store
            persist_directory: Directory to persist the vector store (optional)

        Returns:
            FAISS vector store containing the embedded documents
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")

        logger.info(f"Creating vector store '{store_name}' with {len(documents)} documents")

        try:
            # Create FAISS index from documents
            start_time = time.time()
            vector_store = FAISS.from_documents(documents, self.embeddings)
            elapsed_time = time.time() - start_time

            logger.info(f"Vector store creation completed in {elapsed_time:.2f} seconds")

            # Store in our dictionary
            self.vector_stores[store_name] = vector_store

            # Persist if directory provided
            if persist_directory:
                self.persist_vector_store(store_name, persist_directory)

            return vector_store

        except Exception as e:
            error_msg = f"Error creating vector store: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

    def update_vector_store(self,
                            documents: List[Document],
                            store_name: str,
                            persist_directory: Optional[str] = None) -> FAISS:
        """
        Update an existing vector store with new documents.

        Args:
            documents: List of new documents to add
            store_name: Name of the vector store to update
            persist_directory: Directory to persist the updated vector store (optional)

        Returns:
            Updated FAISS vector store
        """
        if not documents:
            logger.warning("No documents provided to update vector store")
            return self.vector_stores.get(store_name)

        if store_name not in self.vector_stores:
            logger.warning(f"Vector store '{store_name}' does not exist, creating new one")
            return self.create_vector_store(documents, store_name, persist_directory)

        logger.info(f"Updating vector store '{store_name}' with {len(documents)} new documents")

        try:
            # Get existing vector store
            vector_store = self.vector_stores[store_name]

            # Add new documents
            start_time = time.time()
            vector_store.add_documents(documents)
            elapsed_time = time.time() - start_time

            logger.info(f"Vector store update completed in {elapsed_time:.2f} seconds")

            # Persist if directory provided
            if persist_directory:
                self.persist_vector_store(store_name, persist_directory)

            return vector_store

        except Exception as e:
            error_msg = f"Error updating vector store: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

    def persist_vector_store(self, store_name: str, persist_directory: str) -> None:
        """
        Persist a vector store to disk.

        Args:
            store_name: Name of the vector store to persist
            persist_directory: Directory to save the vector store
        """
        if store_name not in self.vector_stores:
            raise ValueError(f"Vector store '{store_name}' does not exist")

        vector_store = self.vector_stores[store_name]

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Save to the specified directory
        logger.info(f"Persisting vector store '{store_name}' to {persist_directory}")

        try:
            store_path = os.path.join(persist_directory, store_name)
            vector_store.save_local(store_path)
            logger.info(f"Vector store saved to {store_path}")

            # Also save metadata about the store
            metadata = {
                "store_name": store_name,
                "document_count": len(vector_store.docstore._dict),
                "embedding_model": self.model_name,
                "created_at": time.time()
            }

            metadata_path = os.path.join(persist_directory, f"{store_name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Vector store metadata saved to {metadata_path}")

        except Exception as e:
            error_msg = f"Error persisting vector store: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

    def load_vector_store(self, store_name: str, persist_directory: str) -> FAISS:
        """
        Load a vector store from disk.

        Args:
            store_name: Name of the vector store to load
            persist_directory: Directory where the vector store is saved

        Returns:
            Loaded FAISS vector store
        """
        store_path = os.path.join(persist_directory, store_name)

        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Vector store not found at {store_path}")

        logger.info(f"Loading vector store '{store_name}' from {store_path}")

        try:
            # Load using the embedding function
            vector_store = FAISS.load_local(store_path, self.embeddings)

            # Store in our dictionary
            self.vector_stores[store_name] = vector_store

            # Load and log metadata if available
            metadata_path = os.path.join(persist_directory, f"{store_name}_metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                logger.info(f"Loaded vector store with {metadata.get('document_count', 'unknown')} documents, "
                            f"created with {metadata.get('embedding_model', 'unknown')} embedding model")
            else:
                logger.info(f"Loaded vector store (metadata not available)")

            return vector_store

        except Exception as e:
            error_msg = f"Error loading vector store: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

    def get_vector_store(self, store_name: str) -> Optional[FAISS]:
        """
        Get a vector store by name.

        Args:
            store_name: Name of the vector store

        Returns:
            FAISS vector store if it exists, None otherwise
        """
        return self.vector_stores.get(store_name)

    def list_vector_stores(self) -> List[str]:
        """
        List all available vector stores.

        Returns:
            List of vector store names
        """
        return list(self.vector_stores.keys())

    def combine_vector_stores(self,
                              target_name: str,
                              source_names: List[str],
                              persist_directory: Optional[str] = None) -> FAISS:
        """
        Combine multiple vector stores into one.

        Args:
            target_name: Name for the combined vector store
            source_names: Names of the vector stores to combine
            persist_directory: Directory to persist the combined vector store (optional)

        Returns:
            Combined FAISS vector store
        """
        if not source_names:
            raise ValueError("No source vector stores provided")

        # Check that all source stores exist
        missing_stores = [name for name in source_names if name not in self.vector_stores]
        if missing_stores:
            raise ValueError(f"Vector stores not found: {', '.join(missing_stores)}")

        logger.info(f"Combining vector stores {source_names} into '{target_name}'")

        try:
            # Get all documents from all source stores
            all_documents = []
            for name in source_names:
                store = self.vector_stores[name]
                documents = self._extract_documents_from_store(store)
                all_documents.extend(documents)
                logger.info(f"Added {len(documents)} documents from '{name}'")

            # Create new vector store with all documents
            return self.create_vector_store(all_documents, target_name, persist_directory)

        except Exception as e:
            error_msg = f"Error combining vector stores: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

    def _extract_documents_from_store(self, vector_store: FAISS) -> List[Document]:
        """
        Extract all documents from a vector store.

        Args:
            vector_store: FAISS vector store

        Returns:
            List of documents from the store
        """
        documents = []
        for id, doc in vector_store.docstore._dict.items():
            documents.append(doc)

        return documents

    def get_stats(self, store_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about vector stores.

        Args:
            store_name: Name of specific vector store to get stats for (optional)

        Returns:
            Dictionary with statistics
        """
        if store_name:
            if store_name not in self.vector_stores:
                return {"error": f"Vector store '{store_name}' not found"}

            store = self.vector_stores[store_name]
            return {
                "name": store_name,
                "document_count": len(store.docstore._dict),
                "embedding_model": self.model_name
            }
        else:
            # Stats for all stores
            all_stats = {
                "total_stores": len(self.vector_stores),
                "store_names": list(self.vector_stores.keys()),
                "embedding_model": self.model_name,
                "stores": {}
            }

            for name, store in self.vector_stores.items():
                all_stats["stores"][name] = {
                    "document_count": len(store.docstore._dict)
                }

            return all_stats
