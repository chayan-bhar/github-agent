# RepoMind — GitHub Repository RAG Agent

RepoMind is a portfolio-quality conversational AI agent that indexes GitHub code repositories into a vector database to answer technical questions about codebases with precise source attribution.

---

## Current Phase: Phase 1 (Ingestion & Vector Storage)

Phase 1 focuses exclusively on repository loading, file filtering, language-aware document chunking, embeddings generation, and vector storage in **Qdrant**.

### Architecture Pipeline

```text
Fixed GitHub Repository (spring-petclinic)
               │
               ▼
       Repository Loader (loader.py)
   (Filters binaries, maps extensions to languages)
               │
               ▼
      LangChain Documents
  (With metadata: file_path, language, etc.)
               │
               ▼
        Code/Text Chunker (chunker.py)
  (Recursive character splitter + chunk_index)
               │
               ▼
     Embeddings Generation (embeddings.py)
      (Google Gemini / OpenAI Embeddings)
               │
               ▼
     Qdrant Vector Store (vectorstore.py)
         (Collection: 'repomind')
```

---

## Prerequisites

- **Python**: `3.11+`
- **Docker**: For running Qdrant locally (Port `6333`)
- **API Key**: Google Gemini API key (`GEMINI_API_KEY`) or OpenAI API key (`OPENAI_API_KEY`)

---

## Setup Instructions

### 1. Clone Repository & Setup Environment

```bash
# Clone the RepoMind repository (or navigate to workspace)
cd repo-mind

# Change into backend directory
cd backend

# Create Python 3.11 virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` inside `backend/` (or project root):

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```env
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Vector Database Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=repomind
```

### 3. Clone Target Knowledge Base Repository

RepoMind currently indexes the **Spring PetClinic** repository as its fixed knowledge base:

```bash
cd ..
git clone https://github.com/spring-projects/spring-petclinic.git repo
```

*(Ensure `repo/` exists as a sibling directory to `backend/`)*

### 4. Start Local Qdrant Server

Run Qdrant using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Running Ingestion Pipeline

To process the repository and load vector embeddings into Qdrant:

```bash
cd backend
python scripts/ingest.py
```

### Example Ingestion Output

```text
RepoMind ingestion started...
Repository: /path/to/repomind/repo
Files loaded: 76
Chunks created: 264
Generating embeddings...
Storing vectors in Qdrant...
Collection: repomind
Ingestion completed successfully.
```

---

## Verification

### Automated Unit Tests

Run the test suite to verify loader filtering, metadata preservation, chunking, and Qdrant in-memory integration without requiring an external database or API key:

```bash
cd backend
pytest
```

### Verifying Qdrant Storage

Verify that the Qdrant collection `repomind` was created and populated:

```bash
curl http://localhost:6333/collections/repomind
```

Expected response:

```json
{
  "result": {
    "status": "green",
    "vectors_count": 264,
    "indexed_vectors_count": 264,
    "points_count": 264,
    "config": { ... }
  },
  "status": "ok"
}
```

---

## Project Structure

```text
repomind/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── loader.py
│   │       ├── chunker.py
│   │       ├── embeddings.py
│   │       └── vectorstore.py
│   ├── scripts/
│   │   └── ingest.py
│   ├── tests/
│   │   └── test_rag.py
│   ├── requirements.txt
│   └── .env.example
├── .env.example
├── .gitignore
├── repo/                  # Fixed target repository (spring-petclinic)
└── README.md
```
