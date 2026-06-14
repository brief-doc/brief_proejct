from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.llm.llm import get_llm
from app.llm.pipeline import _format_docs, run_query
from app.llm.prompts import RAG_PROMPT
from app.llm.retriever import get_retriever
from app.services import rag_service

router = APIRouter(prefix="/rag", tags=["rag"])

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag")


# ── 응답 스키마 ──────────────────────────────────────────────────────────────


class RagQueryRequest(BaseModel):
    question: str
    cat_id: int | None = None


class ReferenceOut(BaseModel):
    doc_name: str
    category: str
    page: str
    snippet: str


class RagQueryResponse(BaseModel):
    query_id: int
    question: str
    answer: str
    references: list[ReferenceOut]


class RagHistoryItem(BaseModel):
    query_id: int
    query_text: str
    answer_text: str | None
    source_count: int
    created_at: str


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────


def _build_refs(docs) -> list[dict]:
    return [
        {
            "doc_id": doc.metadata.get("doc_id"),
            "doc_name": doc.metadata.get("doc_name") or doc.metadata.get("file_name", "?"),
            "category": doc.metadata.get("category", ""),
            "page": str(doc.metadata.get("page_num", "")),
            "snippet": doc.page_content[:200] + "...",
        }
        for doc in docs
    ]


# ── 엔드포인트 ────────────────────────────────────────────────────────────────


@router.post("/query", response_model=RagQueryResponse)
async def query_rag(
    body: RagQueryRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """RAG 질의 (동기). 완전한 답변과 참고 문서를 반환합니다."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: run_query(question, current_user.user_id, body.cat_id),
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message", "관련 문서를 찾을 수 없습니다."))

    references = result["references"]
    saved = rag_service.log_query(
        db=db,
        user_id=current_user.user_id,
        question=question,
        answer=result["answer"],
        references=references,
    )

    return RagQueryResponse(
        query_id=saved.query_id,
        question=question,
        answer=result["answer"],
        references=[
            ReferenceOut(
                doc_name=r["doc_name"],
                category=r["category"],
                page=str(r.get("page", "")),
                snippet=r["snippet"],
            )
            for r in references
        ],
    )


@router.get("/query/stream")
async def query_stream(
    question: str = Query(..., min_length=1),
    cat_id: int | None = Query(None),  # noqa: ARG001 (reserved for future retriever filter)
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """RAG 질의 (SSE 스트리밍). LLM 토큰을 실시간으로 전송합니다.

    이벤트 형식:
      {"type": "token",   "content": "..."}   — LLM 토큰
      {"type": "sources", "references": [...]} — 참고 문서 목록
      {"type": "done"}                         — 종료 신호
      {"type": "error",   "content": "..."}    — 오류
    """
    q = question.strip()
    user_id = current_user.user_id
    loop = asyncio.get_running_loop()

    async def generate():
        # 1. 하이브리드 검색 (blocking → executor)
        retriever = get_retriever(user_id)
        try:
            docs = await loop.run_in_executor(_executor, lambda: retriever.invoke(q))
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"
            return

        if not docs:
            yield f"data: {json.dumps({'type': 'error', 'content': '관련 문서를 찾을 수 없습니다.'}, ensure_ascii=False)}\n\n"
            return

        context = _format_docs(docs)
        stream_chain = RAG_PROMPT | get_llm() | StrOutputParser()

        # 2. 토큰 스트리밍: sync generator → async Queue 브릿지
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _stream_worker():
            try:
                for chunk in stream_chain.stream({"context": context, "question": q}):
                    asyncio.run_coroutine_threadsafe(token_queue.put(chunk), loop)
            except Exception as err:
                asyncio.run_coroutine_threadsafe(token_queue.put(f"\n[오류: {err}]"), loop)
            finally:
                asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)

        loop.run_in_executor(_executor, _stream_worker)

        answer_parts: list[str] = []
        while True:
            chunk = await token_queue.get()
            if chunk is None:
                break
            answer_parts.append(chunk)
            yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

        full_answer = "".join(answer_parts)

        # 3. 참고 문서 전송
        references = _build_refs(docs)
        yield f"data: {json.dumps({'type': 'sources', 'references': references}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        # 4. DB 저장 (실패해도 스트림에 영향 없음)
        try:
            rag_service.log_query(
                db=db,
                user_id=user_id,
                question=q,
                answer=full_answer,
                references=references,
            )
        except Exception:
            pass

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/history", response_model=list[RagHistoryItem])
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """현재 사용자의 RAG 질의 이력을 반환합니다."""
    items = rag_service.get_query_history(db=db, user_id=current_user.user_id, skip=skip, limit=limit)
    return [
        RagHistoryItem(
            query_id=item.query_id,
            query_text=item.query_text,
            answer_text=item.answer_text,
            source_count=item.source_count or 0,
            created_at=item.created_at.isoformat() if item.created_at else "",
        )
        for item in items
    ]
