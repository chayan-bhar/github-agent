import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("repomind.chunker")


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits documents into smaller text chunks using RecursiveCharacterTextSplitter.
    Preserves all original file metadata and appends chunk_index.
    
    Args:
        documents: List of input Document objects loaded from repository.
        chunk_size: Target maximum chunk character length (default 1000).
        chunk_overlap: Number of overlapping characters between adjacent chunks (default 150).
        
    Returns:
        List of chunked Document objects with updated metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ";\n",
            "{\n",
            "}\n",
            " ",
            "",
        ]
    )

    chunked_documents: List[Document] = []

    for doc in documents:
        # Split individual document content
        raw_chunks = splitter.split_text(doc.page_content)
        
        # If text splitting yields no chunks (e.g. empty file), skip or create single empty document
        if not raw_chunks:
            continue

        for idx, chunk_text in enumerate(raw_chunks):
            # Create fresh metadata copy to prevent reference sharing across chunks
            chunk_metadata = dict(doc.metadata)
            chunk_metadata["chunk_index"] = idx

            chunk_doc = Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            )
            chunked_documents.append(chunk_doc)

    logger.info(
        f"Chunked {len(documents)} document(s) into {len(chunked_documents)} total chunk(s) "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})."
    )

    return chunked_documents
