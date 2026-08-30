import logging
from typing import List, Any
from langchain_core.documents import Document

from app.config import QDRANT_URL, QDRANT_COLLECTION_NAME

logger = logging.getLogger("repomind.vectorstore")


def store_documents(
    documents: List[Document],
    embeddings: Any,
    url: str = QDRANT_URL,
    collection_name: str = QDRANT_COLLECTION_NAME
) -> Any:
    """
    Connects to Qdrant, recreates the target collection for idempotency,
    and stores document vectors with full metadata.
    
    Args:
        documents: List of chunked LangChain Document objects.
        embeddings: LangChain Embeddings instance.
        url: Qdrant server URL (default http://localhost:6333).
        collection_name: Qdrant collection name (default repomind).
        
    Returns:
        QdrantVectorStore instance.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
        from langchain_qdrant import QdrantVectorStore
    except ImportError:
        raise ImportError(
            "qdrant-client and langchain-qdrant are required. "
            "Please run: pip install qdrant-client langchain-qdrant"
        )

    logger.info(f"Connecting to Qdrant server at '{url}'...")
    
    try:
        client = QdrantClient(url=url)
        # Test server connectivity by getting collections
        _ = client.get_collections()
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant at {url}: {e}")
        raise ConnectionError(
            f"ERROR: Qdrant vector database is unavailable at '{url}'.\n"
            "Please ensure Qdrant is running locally using Docker:\n"
            "  docker run -p 6333:6333 qdrant/qdrant\n"
            f"Original error details: {e}"
        )

    # Idempotent collection setup: delete existing collection if present
    try:
        if client.collection_exists(collection_name=collection_name):
            logger.info(f"Collection '{collection_name}' already exists. Recreating collection for fresh indexing...")
            client.delete_collection(collection_name=collection_name)
    except Exception as e:
        logger.warning(f"Error checking/deleting collection '{collection_name}': {e}")

    logger.info(f"Generating embeddings and storing {len(documents)} vector(s) into Qdrant collection '{collection_name}'...")
    
    try:
        vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            url=url,
            collection_name=collection_name,
        )
        logger.info(f"Successfully stored {len(documents)} vectors in Qdrant collection '{collection_name}'.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to store vectors in Qdrant: {e}")
        raise RuntimeError(f"ERROR: Failed to store vectors in Qdrant collection '{collection_name}': {e}")
