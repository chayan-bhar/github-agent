import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root or backend directory if present
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
root_dir = backend_dir.parent

# Check for .env in current working dir, backend, or root
env_paths = [
    Path.cwd() / ".env",
    backend_dir / ".env",
    root_dir / ".env",
]

for env_path in env_paths:
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
        break
else:
    load_dotenv()  # Fallback to standard search

# Configuration Settings
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "repomind")

# Configurable repository path relative to backend root by default
DEFAULT_REPO_PATH: str = os.getenv("REPO_PATH", "../repo")

# Sensible configurable maximum file size (default 1 MB)
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", str(1 * 1024 * 1024)))

def validate_config(require_api_key: bool = True) -> None:
    """
    Validates required configuration settings.
    Raises ValueError if critical environment variables are missing.
    """
    if require_api_key and not GEMINI_API_KEY and not OPENAI_API_KEY:
        raise ValueError(
            "ERROR: GEMINI_API_KEY (or OPENAI_API_KEY) is not configured. "
            "Please create a .env file based on .env.example and set your API key."
        )
