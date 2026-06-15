"""
LLM 모듈 — LangChain 기반 RAG 파이프라인 (v3.0)
================================================

╔══════════════════════════════════════════════════════════╗
║              레고 블록 구조 (교체 단위)                  ║
╠══════════════════╦═══════════════════════════════════════╣
║ 파일             ║ 역할 / 교체 시 영향 범위              ║
╠══════════════════╬═══════════════════════════════════════╣
║ config.py        ║ 환경 변수·상수 (전체 공통)            ║
║ llm.py           ║ LLM 싱글톤 → pipeline·summarizer      ║
║ embeddings.py    ║ 임베딩 싱글톤 → vectorstore·ingest    ║
║ vectorstore.py   ║ ChromaDB → ingest·retriever           ║
║ chunker.py       ║ 청킹 전략 → ingest                    ║
║ prompts.py       ║ 프롬프트 → pipeline·summarizer        ║
║ retriever.py     ║ 하이브리드 검색+리랭킹 → pipeline     ║
║ ingest.py        ║ 문서 저장 (PDF·마크다운)              ║
║ pipeline.py      ║ RAG 파이프라인 (LCEL)                 ║
║ summarizer.py    ║ 문서 요약 (map_reduce chain)          ║
║ loader.py        ║ QA 데이터 로더                        ║
╚══════════════════╩═══════════════════════════════════════╝

빠른 교체 가이드:
    LLM 변경     → llm.py 의 get_llm() 수정
    임베딩 변경  → embeddings.py 의 get_embeddings() 수정
    청킹 변경    → chunker.py 의 pdf_splitter / markdown_splitter 교체
    프롬프트 변경 → prompts.py 의 RAG_PROMPT / MAP_PROMPT 교체
    검색 변경    → retriever.py 의 HybridRetriever._get_relevant_documents() 수정
    벡터 DB 변경 → vectorstore.py 의 get_vectorstore() 수정
"""

# sentence_transformers 를 langchain_chroma 보다 먼저 로드해야 segfault 방지
try:
    import sentence_transformers as _st  # noqa: F401
except ImportError:
    pass

from .pipeline import NO_ANSWER_MSG, build_rag_chain, invalidate_cache, run_query
from .prompts import CATEGORIES
from .summarizer import classify_document_category, summarize_document

__all__ = [
    # RAG
    "run_query",
    "build_rag_chain",
    "invalidate_cache",
    "NO_ANSWER_MSG",
    # 요약
    "summarize_document",
    "classify_document_category",
    "CATEGORIES",
]

__version__ = "3.0.0"
