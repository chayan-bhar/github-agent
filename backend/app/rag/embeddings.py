import logging
from typing import Optional, Any
from app.config import GEMINI_API_KEY, OPENAI_API_KEY

logger = logging.getLogger("repomind.embeddings")


def get_embeddings_model(
    provider: Optional[str] = None,
    api_key: Optional[str] = None
) -> Any:
    """
    Initializes and returns a LangChain Embeddings model instance.
    
    Supports 'gemini' (GoogleGenerativeAIEmbeddings) and 'openai' (OpenAIEmbeddings).
    If provider is omitted, defaults to 'gemini' if GEMINI_API_KEY is present,
    otherwise falls back to 'openai'.
    
    Args:
        provider: 'gemini' or 'openai'.
        api_key: Explicit API key string (optional).
        
    Returns:
        Embeddings model instance implementing LangChain Embeddings interface.
    """
    selected_provider = (provider or "").lower()

    if not selected_provider:
        if GEMINI_API_KEY or api_key:
            selected_provider = "gemini"
        elif OPENAI_API_KEY:
            selected_provider = "openai"
        else:
            selected_provider = "gemini"

    if selected_provider == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-google-genai package is required for Gemini embeddings. "
                "Please run: pip install langchain-google-genai"
            )
        
        effective_key = api_key or GEMINI_API_KEY
        if not effective_key:
            raise ValueError(
                "ERROR: GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in your .env file."
            )

        logger.info("Initializing Google Gemini Embeddings (models/text-embedding-004)...")
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=effective_key
        )

    elif selected_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai package is required for OpenAI embeddings. "
                "Please run: pip install langchain-openai"
            )
        
        effective_key = api_key or OPENAI_API_KEY
        if not effective_key:
            raise ValueError(
                "ERROR: OPENAI_API_KEY is missing. Please set OPENAI_API_KEY in your .env file."
            )

        logger.info("Initializing OpenAI Embeddings (text-embedding-3-small)...")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=effective_key
        )

    else:
        raise ValueError(f"Unsupported embedding provider: '{selected_provider}'. Use 'gemini' or 'openai'.")
