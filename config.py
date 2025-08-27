
import os

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# VectorStore
FAISS_DIR = os.getenv("FAISS_DIR", ".faiss_index")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# LLM
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# App
APP_TITLE = "RAG – Code de la Route Sénégal"
DEFAULT_DOCSET = os.getenv("DOCSET", "default")
DATA_DIR = os.getenv("DATA_DIR", "data")
