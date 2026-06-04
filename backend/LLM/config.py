"""
통합 설정 관리
===============
모든 모델 설정을 한 곳에서 관리합니다.

담당 영역:
- [LLM 팀] LLM_CONFIG: 대형 언어 모델 설정
- [임베딩 팀] EMBEDDING_CONFIG: 텍스트 임베딩 모델 설정
- [공용] DB_CONFIG, CHROMA_DB_PATH, TEXT_SPLITTER_CONFIG, API_CONFIG

모델 변경 시:
- LLM 모델 변경: .env 파일의 CURRENT_MODEL과 LLM_TEMPERATURE 수정
- 임베딩 모델 변경: .env 파일의 EMBEDDING_MODEL 수정

⚠️ 중요: 비밀정보는 .env 파일에서 관리하므로, config.py를 수정하지 마세요.
         .env 파일을 수정한 후 애플리케이션을 다시 시작하세요.
"""
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os

# .env 파일 로드 (비밀정보, 환경 변수 로드)
load_dotenv()

# ========== 데이터베이스 URL 처리 ==========
# SQLAlchemy 형식의 DATABASE_URL을 지원하면서도
# 기존의 개별 설정도 유지 (하위 호환성)
DATABASE_URL = os.getenv("DATABASE_URL", "")

# DATABASE_URL에서 개별 값 파싱
def parse_database_url(url: str) -> dict:
    """
    SQLAlchemy 형식의 DATABASE_URL을 파싱하여 psycopg2 연결 설정으로 변환
    
    예: postgresql://postgres:password@localhost:5432/pdf_db
    """
    if not url:
        return {}
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/") or "pdf_db",
            "user": parsed.username or "postgres",
            "password": "8342"
        }
    except Exception as e:
        print(f"[경고] DATABASE_URL 파싱 실패: {e}")
        return {}

# DATABASE_URL이 있으면 우선 사용, 없으면 개별 설정 사용
if DATABASE_URL:
    db_from_url = parse_database_url(DATABASE_URL)
    DB_CONFIG = {
        "host": db_from_url.get("host", os.getenv("DB_HOST", "localhost")),
        "port": db_from_url.get("port", int(os.getenv("DB_PORT", "5432"))),
        "database": db_from_url.get("database", os.getenv("DB_NAME", "pdf_db")),
        "user": db_from_url.get("user", os.getenv("DB_USER", "postgres")),
        "password": db_from_url.get("password", os.getenv("DB_PASSWORD", "8342"))
    }
else:
    # 개별 환경 변수 사용
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "pdf_db"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "8342")
    }

# ========== ChromaDB 설정 ==========
# 벡터 데이터베이스 저장 경로 (로컬 폴더)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_pdf_db")

# ========== 사용할 LLM 모델명 (최상단에서 선언) ==========
# .env 파일에서 CURRENT_MODEL을 읽음
# 사용 가능한 모델: "gemma3n:e2b", "qwen2.5:3b", "llama2" 등
CURRENT_MODEL = os.getenv("CURRENT_MODEL", "gemma3n:e2b")

# ========== LLM (대형 언어 모델) 설정 ==========
# Ollama를 통해 로컬에서 실행되는 모델 설정 (.env 파일에서 관리)
LLM_CONFIG = {
    "model_name": CURRENT_MODEL,                                      # .env의 CURRENT_MODEL
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),       # .env의 LLM_TEMPERATURE
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # .env의 OLLAMA_BASE_URL
}

# ========== 임베딩 모델 설정 ==========
# 텍스트를 벡터로 변환하는 모델 (한국어 최적화, .env 파일에서 관리)
# 빠른 임베딩을 위해 기본값을 sentence-transformers로 변경 (더 가벼움)
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
    "normalize_embeddings": os.getenv("NORMALIZE_EMBEDDINGS", "True").lower() == "true"
}

# ========== 텍스트 분할 설정 ==========
# PDF를 청크로 분할할 때의 파라미터
TEXT_SPLITTER_CONFIG = {
    "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50"))
}

# ========== API 설정 ==========
# FastAPI 서버 설정
# - CURRENT_MODEL 변수를 직접 참조하여 타이틀 생성
# - 모델을 변경하면 자동으로 Swagger UI에도 반영됨
API_CONFIG = {
    "title": f"로컬 AI({CURRENT_MODEL}) 기반 PDF 요약 및 RAG API",
    "description": "Ollama와 HuggingFace를 이용한 100% 로컬 프라이빗 파이프라인입니다.",
    "version": "2.0.0"
}

# ========== 편의 함수 ==========
def get_llm_config_summary():
    """현재 LLM 설정을 사용자가 읽기 좋게 반환"""
    return f"모델: {CURRENT_MODEL}, 온도: {LLM_CONFIG['temperature']}, URL: {LLM_CONFIG['base_url']}"

def get_embedding_config_summary():
    """현재 임베딩 모델 설정을 사용자가 읽기 좋게 반환"""
    return f"모델: {EMBEDDING_CONFIG['model_name']}, 디바이스: {EMBEDDING_CONFIG['device']}"

def print_config_security_notice():
    """보안 안내 메시지 출력"""
    print("=" * 60)
    print("🔒 설정 관리 안내")
    print("=" * 60)
    print("✓ 비밀정보는 .env 파일에서 관리됩니다")
    print("✓ .env 파일은 .gitignore에 추가되어 Git에 커밋되지 않습니다")
    print("✓ 개발 환경 설정은 .env.example을 참고하세요")
    print("=" * 60)

