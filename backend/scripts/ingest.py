#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

# Add backend directory to sys.path to enable smooth module imports
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    DEFAULT_REPO_PATH,
    validate_config
)
from app.rag.loader import load_repository
from app.rag.chunker import chunk_documents
from app.rag.embeddings import get_embeddings_model
from app.rag.vectorstore import store_documents

# Configure clean logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("repomind.ingest")


def main() -> None:
    """
    Main entry point for Phase 1 repository ingestion pipeline.
    """
    print("RepoMind ingestion started...")
    
    # 1. Resolve repository path
    repo_path = Path(DEFAULT_REPO_PATH)
    if not repo_path.is_absolute():
        repo_path = (backend_dir / DEFAULT_REPO_PATH).resolve()

    print(f"Repository: {repo_path}")

    # Validate repository directory existence
    if not repo_path.exists() or not repo_path.is_dir():
        print(f"ERROR: Repository directory '{DEFAULT_REPO_PATH}' does not exist. Please clone the repository first.")
        sys.exit(1)

    # 2. Validate API key configuration
    try:
        validate_config(require_api_key=True)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    # 3. Load repository files into LangChain Document objects
    try:
        documents = load_repository(repo_path=str(repo_path))
    except Exception as e:
        print(f"ERROR: Failed to load repository: {e}")
        sys.exit(1)

    print(f"Files loaded: {len(documents)}")

    if not documents:
        print("WARNING: No valid files found to ingest.")
        sys.exit(0)

    # 4. Chunk documents with RecursiveCharacterTextSplitter
    chunks = chunk_documents(documents=documents)
    print(f"Chunks created: {len(chunks)}")

    # 5. Initialize Embeddings model
    print("Generating embeddings...")
    try:
        embeddings = get_embeddings_model()
    except Exception as e:
        print(f"ERROR: Failed to initialize embeddings model: {e}")
        sys.exit(1)

    # 6. Store vectors in Qdrant Vector Database
    print("Storing vectors in Qdrant...")
    print(f"Collection: {QDRANT_COLLECTION_NAME}")
    try:
        _ = store_documents(
            documents=chunks,
            embeddings=embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION_NAME
        )
    except ConnectionError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Ingestion failed while storing vectors: {e}")
        sys.exit(1)

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
