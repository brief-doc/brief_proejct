"""
통합 설정 관리
===============
모든 모델 설정을 한 곳에서 관리합니다.

담당 영역:
- [LLM 팀] LLM_CONFIG: 대형 언어 모델 설정
- [임베딩 팀] EMBEDDING_CONFIG: 텍스트 임베딩 모델 설정
- [공용] DB_CONFIG, CHROMA_DB_PATH, TEXT_SPLITTER_CONFIG, API_CONFIG

모델 변경 시:
- LLM 모델 변경: model_name과 LLM_CONFIG만 수정
- 임베딩 모델 변경: EMBEDDING_CONFIG만 수정
"""

# ========== 사용할 LLM 모델명 (최상단에서 선언) ==========
# 모든 곳에서 이 변수를 참조하므로 여기만 변경하면 됨
# 사용 가능한 모델: "gemma3n:e2b", "qwen2.5:3b", "llama2" 등
model_name = "gemma3n:e2b"

# ========== LLM (대형 언어 모델) 설정 ==========
# Ollama를 통해 로컬에서 실행되는 모델 설정
LLM_CONFIG = {
    "model_name": model_name,              # 위에서 선언한 model_name 참조
    "temperature": 0.1,                    # 0.0~1.0 (낮을수록 정확성, 높을수록 창의성)
    "base_url": "http://localhost:11434"   # Ollama 서버 주소 (기본값)
}

# ========== 임베딩 모델 설정 ==========
# 텍스트를 벡터로 변환하는 모델 (한국어 최적화)
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-m3",  # HuggingFace 모델명
                                                   # 다른 모델로 바꾸려면: "monologg/kobert-base" 등
    "device": "cuda",                             # "cuda" (GPU) 또는 "cpu"
    "normalize_embeddings": True                  # 임베딩 정규화 (유사도 계산 개선)
}

# ========== PostgreSQL 설정 ==========
# 메타데이터 저장 용 데이터베이스
DB_CONFIG = {
    "host": "localhost",          # PostgreSQL 서버 주소
    "port": 5432,                 # PostgreSQL 포트
    "database": "pdf_db",         # 데이터베이스명
    "user": "postgres",           # 사용자명
    "password": "your_password"   # 비밀번호 (실제 값으로 변경 필수)
}

# ========== ChromaDB 설정 ==========
# 벡터 데이터베이스 저장 경로 (로컬 폴더)
CHROMA_DB_PATH = "./chroma_pdf_db"

# ========== 텍스트 분할 설정 ==========
# PDF를 청크로 분할할 때의 파라미터
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,      # 한 청크의 최대 길이 (문자 수)
    "chunk_overlap": 50     # 청크 간 겹침 길이 (문맥 연결성 유지)
}

# ========== API 설정 ==========
# FastAPI 서버 설정
# - model_name 변수를 직접 참조하여 타이틀 생성
# - 모델을 변경하면 자동으로 Swagger UI에도 반영됨
API_CONFIG = {
    "title": f"로컬 AI({model_name}) 기반 PDF 요약 및 RAG API",
    "description": "Ollama와 HuggingFace를 이용한 100% 로컬 프라이빗 파이프라인입니다.",
    "version": "2.0.0"
}

# ========== 편의 함수 ==========
def get_llm_config_summary():
    """현재 LLM 설정을 사용자가 읽기 좋게 반환"""
    return f"모델: {LLM_CONFIG['model_name']}, 온도: {LLM_CONFIG['temperature']}, URL: {LLM_CONFIG['base_url']}"

def get_embedding_config_summary():
    """현재 임베딩 모델 설정을 사용자가 읽기 좋게 반환"""
    return f"모델: {EMBEDDING_CONFIG['model_name']}, 디바이스: {EMBEDDING_CONFIG['device']}"
