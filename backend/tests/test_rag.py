import sys
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.rag.loader import detect_language, is_file_supported, load_repository
from app.rag.chunker import chunk_documents


def test_language_detection():
    """
    Test language detection based on extensions and special file names.
    """
    assert detect_language("OwnerController.java", ".java") == "java"
    assert detect_language("main.py", ".py") == "python"
    assert detect_language("app.ts", ".ts") == "typescript"
    assert detect_language("index.js", ".js") == "javascript"
    assert detect_language("pom.xml", ".xml") == "xml"
    assert detect_language("application.yml", ".yml") == "yaml"
    assert detect_language("README.md", ".md") == "markdown"
    assert detect_language("unknown.foo", ".foo") == "unknown"


def test_file_filtering():
    """
    Test file filtering logic for supported vs binary/ignored files.
    """
    assert is_file_supported(Path("OwnerController.java")) is True
    assert is_file_supported(Path("pom.xml")) is True
    assert is_file_supported(Path("README.md")) is True
    assert is_file_supported(Path("build.gradle")) is True
    
    # Binary / ignored extensions
    assert is_file_supported(Path("logo.png")) is False
    assert is_file_supported(Path("app.jar")) is False
    assert is_file_supported(Path("Owner.class")) is False
    assert is_file_supported(Path("archive.zip")) is False


def test_loader_metadata(tmp_path):
    """
    Test loading files from a directory and verifying relative path & metadata.
    """
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()
    
    src_dir = repo_dir / "src" / "main" / "java"
    src_dir.mkdir(parents=True)
    
    java_file = src_dir / "TestService.java"
    java_file.write_text("package com.example;\n\npublic class TestService {}", encoding="utf-8")

    ignored_dir = repo_dir / "target"
    ignored_dir.mkdir()
    (ignored_dir / "compiled.class").write_bytes(b"\x00\x01\x02\x03")

    docs = load_repository(repo_path=str(repo_dir), repo_name="test-repo")
    
    assert len(docs) == 1
    doc = docs[0]
    
    assert doc.metadata["repository"] == "test-repo"
    assert doc.metadata["file_name"] == "TestService.java"
    assert doc.metadata["extension"] == ".java"
    assert doc.metadata["language"] == "java"
    assert doc.metadata["file_path"] == "src/main/java/TestService.java"
    assert "public class TestService" in doc.page_content


def test_chunking_metadata_and_index():
    """
    Test that chunking preserves file metadata and assigns sequential chunk_index.
    """
    large_code = "public class BigClass {\n" + "    // line of code\n" * 100 + "}\n"
    
    initial_metadata = {
        "repository": "spring-petclinic",
        "file_path": "src/main/java/BigClass.java",
        "file_name": "BigClass.java",
        "extension": ".java",
        "language": "java",
    }
    
    doc = Document(page_content=large_code, metadata=initial_metadata)
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=50)
    
    assert len(chunks) > 1
    
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["repository"] == "spring-petclinic"
        assert chunk.metadata["file_path"] == "src/main/java/BigClass.java"
        assert chunk.metadata["file_name"] == "BigClass.java"
        assert chunk.metadata["language"] == "java"
        assert chunk.metadata["chunk_index"] == idx


def test_qdrant_in_memory_vectorstore():
    """
    Test storing and retrieving document vectors in Qdrant (in-memory).
    """
    embeddings = FakeEmbeddings(size=768)
    docs = [
        Document(
            page_content="public class OwnerController {}",
            metadata={
                "repository": "spring-petclinic",
                "file_path": "src/main/java/OwnerController.java",
                "file_name": "OwnerController.java",
                "extension": ".java",
                "language": "java",
                "chunk_index": 0
            }
        )
    ]
    
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name="repomind"
    )
    
    results = vector_store.similarity_search("OwnerController", k=1)
    assert len(results) == 1
    assert results[0].page_content == "public class OwnerController {}"
    assert results[0].metadata["file_name"] == "OwnerController.java"
    assert results[0].metadata["chunk_index"] == 0
