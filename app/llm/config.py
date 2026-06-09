import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────
CURRENT_MODEL = os.getenv("CURRENT_MODEL", "gemma2:2b")
LLM_CONFIG = {
    "model": CURRENT_MODEL,
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    # ── 속도 최적화 ──────────────────────────────────────────────────────────
    # num_ctx: 컨텍스트 윈도우 (토큰). 4096→2048 로 줄이면 KV-cache 절반 감소
    "num_ctx": int(os.getenv("LLM_NUM_CTX", "2048")),
    # num_predict: 최대 출력 토큰 수. 무제한→512 로 제한 (요약·답변에 충분)
    "num_predict": int(os.getenv("LLM_NUM_PREDICT", "512")),
}

# ── Embedding ─────────────────────────────────────
# ingest / retriever 양쪽이 동일 모델을 참조해야 벡터 공간이 일치함
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
    "normalize_embeddings": os.getenv("NORMALIZE_EMBEDDINGS", "True").lower() == "true",
}

# ── ChromaDB ─────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_pdf_db")
COLLECTION_NAME = "qa_collection"

# ── Retrieval ────────────────────────────────────
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "10"))  # 초기 후보 수 (20→10 속도 최적화)
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))  # 리랭킹 후 최종 수

# ── PostgreSQL ───────────────────────────────────
_db_url = os.getenv("DATABASE_URL", "")
if _db_url:
    _p = urlparse(_db_url)
    DB_CONFIG = {
        "host": _p.hostname or "localhost",
        "port": _p.port or 5432,
        "database": (_p.path or "/pdf_db").lstrip("/"),
        "user": _p.username or "postgres",
        "password": _p.password or os.getenv("DB_PASSWORD", "8342"),
    }
else:
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "pdf_db"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "8342"),
    }
