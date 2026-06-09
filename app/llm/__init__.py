"""
LLM 모듈 - FastAPI 서버 및 RAG 파이프라인
========================================

이 패키지는 다음을 포함합니다:
- config.py      : 중앙 설정 관리 (모델, DB, API)
- llm.py         : LLM 인스턴스 관리 (싱글톤 패턴)
- pipeline.py    : RAG 파이프라인 (검색 + LLM 답변, 행정 격식체 프롬프트)
- summarizer.py  : 카테고리별 문서 요약 (법령·조례 / 가이드라인·지침 / 공모·사업 /
                   감사 / 내부 규정 / 기타)
- ingest.py      : 벡터 DB 저장 (PDF·마크다운, category 메타 포함)
- retriever.py   : 하이브리드 검색 + 크로스인코더 리랭킹
"""

from .pipeline import NO_ANSWER_MSG, run_query  # RAG 공개 API
from .summarizer import CATEGORIES, summarize_document  # 카테고리 요약 공개 API

__all__ = ["run_query", "NO_ANSWER_MSG", "summarize_document", "CATEGORIES"]

__version__ = "2.1.0"
