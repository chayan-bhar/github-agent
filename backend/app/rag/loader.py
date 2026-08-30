import os
import logging
from pathlib import Path
from typing import List, Optional, Set
from langchain_core.documents import Document

from app.config import MAX_FILE_SIZE_BYTES

logger = logging.getLogger("repomind.loader")

# Supported file extensions and explicit file names
SUPPORTED_EXTENSIONS: Set[str] = {
    ".java", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".xml", ".yml", ".yaml", ".json", ".properties", ".gradle"
}

SUPPORTED_FILENAMES: Set[str] = {
    "pom.xml", "readme.md", "dockerfile", ".gitignore", "build.gradle", "settings.gradle"
}

# Directories to ignore
IGNORED_DIRS: Set[str] = {
    ".git", "node_modules", "target", "build", "dist",
    ".idea", ".vscode", "__pycache__", ".venv", "venv"
}

# Binary and non-text extensions to ignore explicitly
IGNORED_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".jar",
    ".class", ".tar", ".gz", ".ico", ".svg", ".eot", ".ttf", ".woff",
    ".woff2", ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe"
}

# Language mapping
LANGUAGE_MAP: dict[str, str] = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".xml": "xml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".properties": "properties",
    ".gradle": "gradle",
}


def detect_language(file_name: str, extension: str) -> str:
    """
    Determines programming/markup language from file extension or file name.
    """
    ext_lower = extension.lower()
    if ext_lower in LANGUAGE_MAP:
        return LANGUAGE_MAP[ext_lower]
    
    # Specific file name checks
    name_lower = file_name.lower()
    if name_lower == "pom.xml":
        return "xml"
    if name_lower in ("build.gradle", "settings.gradle"):
        return "gradle"
    if name_lower == "readme.md":
        return "markdown"

    return "unknown"


def is_file_supported(file_path: Path) -> bool:
    """
    Checks if a file should be included in the ingestion pipeline.
    """
    file_name = file_path.name.lower()
    ext = file_path.suffix.lower()

    if ext in IGNORED_EXTENSIONS:
        return False

    if ext in SUPPORTED_EXTENSIONS or file_name in SUPPORTED_FILENAMES:
        return True

    return False


def load_repository(
    repo_path: str,
    max_file_size: int = MAX_FILE_SIZE_BYTES,
    repo_name: str = "spring-petclinic"
) -> List[Document]:
    """
    Recursively scans the target repository path and creates LangChain Document objects.
    
    Args:
        repo_path: Absolute or relative path to the target repository directory.
        max_file_size: Maximum allowed file size in bytes (default 1 MB).
        repo_name: Name of the repository for metadata tagging.
        
    Returns:
        List of LangChain Document objects with file contents and metadata.
    """
    root_path = Path(repo_path).resolve()

    if not root_path.exists():
        raise FileNotFoundError(
            f"ERROR: Repository directory '{repo_path}' does not exist. Please clone the repository first."
        )

    if not root_path.is_dir():
        raise ValueError(f"ERROR: Repository path '{repo_path}' is not a directory.")

    documents: List[Document] = []
    skipped_count = 0
    size_skipped_count = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter out ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]

        for filename in filenames:
            full_file_path = Path(dirpath) / filename
            
            # Check directory parts for hidden/ignored folders
            rel_parts = full_file_path.relative_to(root_path).parts
            if any(part in IGNORED_DIRS or part.startswith(".") for part in rel_parts[:-1]):
                continue

            if not is_file_supported(full_file_path):
                skipped_count += 1
                continue

            # Check file size limit
            try:
                file_size = full_file_path.stat().st_size
                if file_size > max_file_size:
                    logger.warning(
                        f"Skipping large file '{full_file_path}': size {file_size} bytes exceeds limit {max_file_size}"
                    )
                    size_skipped_count += 1
                    continue
            except OSError as e:
                logger.warning(f"Could not stat file '{full_file_path}': {e}")
                continue

            # Read file contents
            try:
                with open(full_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, PermissionError) as e:
                logger.warning(f"Skipping unreadable file '{full_file_path}': {e}")
                skipped_count += 1
                continue

            # Calculate relative path string (e.g. src/main/java/.../OwnerController.java)
            relative_file_path = full_file_path.relative_to(root_path).as_posix()
            ext = full_file_path.suffix.lower()
            lang = detect_language(filename, ext)

            metadata = {
                "repository": repo_name,
                "file_path": relative_file_path,
                "file_name": filename,
                "extension": ext,
                "language": lang,
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

    logger.info(
        f"Loaded {len(documents)} document(s) from repository '{repo_path}' "
        f"(skipped {skipped_count} unsupported/binary files, {size_skipped_count} over size limit)."
    )

    return documents
