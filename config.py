import os

from dotenv import load_dotenv

load_dotenv()

CORPUS_DIR = os.getenv("CORPUS_DIR", "python-3.14-docs-text/")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
