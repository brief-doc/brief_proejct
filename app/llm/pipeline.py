"""RAG 파이프라인 — LangChain LCEL 기반

파이프라인 구조:
    question
      │
      ├─[retriever]──→ docs ──→ _format_docs() ──→ context ─┐
      │                                                       ├──→ RAG_PROMPT → LLM → answer
      └─[passthrough]─────────────────────────→ question ───┘

레고 교체:
    - LLM 교체      → llm.py 의 get_llm() 변경
    - 프롬프트 교체  → prompts.py 의 RAG_PROMPT 변경
    - 검색 방식 교체 → retriever.py 의 HybridRetriever 변경

공개 API:
    build_rag_chain(user_id)  → 스트리밍 등 직접 체인 사용 시
    run_query(question, ...)  → 캐시·references·히스토리 포함 완전 실행
    invalidate_cache(user_id) → 새 파일 업로드 후 캐시 무효화
"""

import hashlib
from datetime import datetime

import psycopg2
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from .config import DB_CONFIG
from .llm import get_llm
from .prompts import RAG_PROMPT
from .retriever import get_retriever

NO_ANSWER_MSG = "제공된 문서에서 해당 내용을 확인할 수 없습니다."


# ── 문서 포매터 ───────────────────────────────────────────────────────────────
def _format_docs(docs: list[Document]) -> str:
    """Document 리스트 → [문서 N] 형식의 컨텍스트 문자열"""
    parts = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        category = meta.get("category", "")
        doc_name = meta.get("doc_name") or meta.get("file_name") or meta.get("case_no", "?")
        page = meta.get("page_num", "")

        header = f"[문서 {i + 1}]"
        if category:
            header += f" [{category}]"
        header += f" {doc_name}"
        if page:
            header += f" (p.{page})"

        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


# ── LCEL 체인 빌더 ────────────────────────────────────────────────────────────
def build_rag_chain(user_id: int | None = None) -> Runnable:
    """사용자별 RAG LCEL 체인을 반환합니다.

    스트리밍 / 비동기 호출 등 직접 체인을 사용하고 싶을 때 활용하세요.

    사용 예:
        chain = build_rag_chain(user_id=1)
        answer = chain.invoke("보조금 신청 마감일은?")

        # 스트리밍
        for chunk in chain.stream("질문"):
            print(chunk, end="", flush=True)
    """
    retriever = get_retriever(user_id)

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(_format_docs),
            question=RunnablePassthrough(),
        )
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return chain


# ── 캐시 ─────────────────────────────────────────────────────────────────────
_cache_data: dict[str, dict] = {}
_cache_index: dict[int, set[str]] = {}


def invalidate_cache(user_id: int | None = None) -> int:
    """캐시 무효화 — 새 파일 업로드 후 호출하세요.

    Args:
        user_id: 특정 사용자 캐시만 삭제. None 이면 전체 삭제.

    Returns:
        삭제된 캐시 항목 수
    """
    global _cache_data, _cache_index
    if user_id is None:
        count = len(_cache_data)
        _cache_data, _cache_index = {}, {}
        return count
    keys = _cache_index.pop(user_id, set())
    for k in keys:
        _cache_data.pop(k, None)
    return len(keys)


# ── 히스토리 로깅 ─────────────────────────────────────────────────────────────
def _log_history(user_id: int, query: str, answer: str, metas: list[dict]) -> None:
    try:
        with psycopg2.connect(**DB_CONFIG, options="-c client_encoding=UTF8") as conn:
            with conn.cursor() as cur:
                doc_ids = ",".join(str(m.get("case_no", "")) for m in metas)
                cur.execute(
                    "INSERT INTO history (user_id,query_text,answer_text,doc_ids,created_at) VALUES (%s,%s,%s,%s,%s)",
                    (user_id, query, answer, doc_ids, datetime.now()),
                )
    except Exception as e:
        print(f"[history] 저장 실패: {e}")


# ── 메인 실행 함수 ────────────────────────────────────────────────────────────
def run_query(
    question: str,
    user_id: int | None = None,
    cat_id: int | None = None,
) -> dict:
    """RAG 쿼리를 실행합니다 (캐시 · references · 히스토리 포함).

    Returns:
        {
            "status":     "success" | "error",
            "answer":     str,
            "references": [{"doc_name", "category", "page", "snippet"}, ...]
        }
    """
    key = hashlib.md5(f"{question}_{user_id}_{cat_id}".encode()).hexdigest()
    if key in _cache_data:
        return _cache_data[key]

    # 1. 문서 검색
    retriever = get_retriever(user_id)
    docs = retriever.invoke(question)

    if not docs:
        return {
            "status": "error",
            "message": "관련 문서를 찾을 수 없습니다.",
            "answer": None,
            "references": [],
        }

    # 2. LLM 호출 (검색 결과를 직접 포맷 — retriever 중복 호출 방지)
    context = _format_docs(docs)
    print(f"[pipeline] 컨텍스트 {len(docs)}개 문서, {len(context)}자")
    print(f"[pipeline] 컨텍스트 앞 500자:\n{context[:500]}")

    answer_text = (RAG_PROMPT | get_llm() | StrOutputParser()).invoke({"context": context, "question": question})
    print(f"[pipeline] LLM 응답: {answer_text[:300]}")

    # 3. references 구성
    references = [
        {
            "doc_name": doc.metadata.get("doc_name") or doc.metadata.get("file_name", "?"),
            "category": doc.metadata.get("category", ""),
            "page": doc.metadata.get("page_num", ""),
            "snippet": doc.page_content[:200] + "...",
        }
        for doc in docs
    ]

    result = {"status": "success", "answer": answer_text, "references": references}

    # 4. 캐시 저장
    _cache_data[key] = result
    if user_id is not None:
        _cache_index.setdefault(user_id, set()).add(key)
        _log_history(user_id, question, answer_text, [d.metadata for d in docs])

    return result


# ── 하위 호환 ─────────────────────────────────────────────────────────────────
class ImprovedRAGPipeline:
    def query_with_context(
        self,
        query: str,
        user_id: int | None = None,
        cat_id: int | None = None,
        top_k: int = 5,
    ) -> dict:
        return run_query(query, user_id, cat_id)
