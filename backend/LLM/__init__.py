"""
LLM 모듈 - FastAPI 서버 및 RAG 파이프라인
========================================

이 패키지는 다음을 포함합니다:
- config.py: 중앙 설정 관리 (모델, DB, API)
- llm_module.py: LLM 인스턴스 관리 (싱글톤 패턴)
- rag_pipeline_v2.py: RAG 파이프라인 (검색 + LLM 답변)
- vectordb_manager_v2.py: 벡터 DB 관리
- main.py: FastAPI 서버 (PDF 업로드, 요약, 배치)
"""

__version__ = "2.0.0"
