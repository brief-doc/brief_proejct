"""
RAG 파이프라인 — 하이브리드 검색 + 프롬프트 규율

프롬프트 규율:
  - [참고문서]에 있는 내용만 사용 (문서 외 지식 금지)
  - [문서 N] 형식 인용 강제 + 조항·페이지 병기
  - 문서 외 답변 시 고정 문구 "제공된 문서에서 해당 내용을 확인할 수 없습니다." 반환
  - 공공기관 행정 격식체(합쇼체) 강제
  - few-shot 예시 2개 내장 (행정 문서 도메인)
"""

import hashlib
from datetime import datetime

import psycopg2

from .config import DB_CONFIG
from .llm import get_llm
from .retriever import get_retriever

# ── 시스템 규칙 ───────────────────────────────────────────────────────────────
_SYSTEM_RULES = """당신은 공공기관 행정 문서 전문 AI입니다. 반드시 아래 규칙을 따르십시오.
규칙 1. [참고문서]에 있는 내용만 사용하여 답변하십시오.
규칙 2. 문서에 없는 내용은 절대 추론하거나 외부 지식으로 보완하지 마십시오.
규칙 3. 모든 주장에 [문서 N] 형식으로 출처를 인용하고, 가능하면 조항·페이지를 병기하십시오.
         예시 인용 형식: [문서 1] "신청 기간은 3월 31일까지" (제3조 제1항)
규칙 4. 문서에서 답을 찾을 수 없으면 반드시 다음 문구만 답하십시오.
         → "제공된 문서에서 해당 내용을 확인할 수 없습니다."
규칙 5. 답변은 공공기관 행정 격식체(합쇼체: ~하십시오, ~입니다)로 작성하십시오.
규칙 6. 수치·날짜·고유명사는 원문 그대로 표기하십시오."""

# ── Few-shot 예시 (행정 문서 도메인) ─────────────────────────────────────────
_FEW_SHOT = """
[예시 1 - 문서에 답이 있는 경우]
질문: 보조금 신청 마감일은 언제입니까?
답변: [문서 1]에 따르면 보조금 신청 마감일은 2025년 3월 31일입니다.
("신청 기간: 2025. 3. 1. ~ 3. 31.", 제3조 제1항)

[예시 2 - 문서에 답이 없는 경우]
질문: 타 지자체의 동일 사업 예산 규모는 얼마입니까?
답변: 제공된 문서에서 해당 내용을 확인할 수 없습니다.
""".strip()

# ── 고정 답변 문구 (프론트엔드 파싱 기준) ────────────────────────────────────
NO_ANSWER_MSG = "제공된 문서에서 해당 내용을 확인할 수 없습니다."


def _build_prompt(docs: list[str], metas: list[dict], query: str) -> str:
    """RAG 최종 프롬프트 생성 — 카테고리 메타데이터가 있으면 문서 헤더에 표시"""
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        # 메타에서 표시 가능한 식별자 추출
        category = meta.get("category", "")
        doc_name = meta.get("doc_name") or meta.get("file_name") or meta.get("case_no", "?")
        page = meta.get("page_num", "")
        header = f"[문서 {i + 1}]"
        if category:
            header += f" [{category}]"
        header += f" {doc_name}"
        if page:
            header += f" (p.{page})"
        context_parts.append(f"{header}\n{doc}")

    context = "\n\n".join(context_parts)

    return f"""{_SYSTEM_RULES}

{_FEW_SHOT}

[참고문서]
{context}

[질문]
{query}

[답변] 반드시 [문서 N] 형식으로 출처를 인용하며 합쇼체로 답변하십시오:"""


def _log_history(user_id: int, query: str, answer: str, refs: list[dict]):
    try:
        with psycopg2.connect(**DB_CONFIG, options="-c client_encoding=UTF8") as conn:
            with conn.cursor() as cur:
                doc_ids = ",".join(str(r.get("case_no", "")) for r in refs)
                cur.execute(
                    "INSERT INTO history (user_id,query_text,answer_text,doc_ids,created_at)"
                    " VALUES (%s,%s,%s,%s,%s)",
                    (user_id, query, answer, doc_ids, datetime.now()),
                )
    except Exception as e:
        print(f"[history] 저장 실패: {e}")


# 캐시 구조:
#   _cache_data[cache_key]  = result
#   _cache_index[user_id]   = {cache_key, cache_key, ...}  ← user_id 역인덱스
_cache_data: dict[str, dict] = {}
_cache_index: dict[int, set[str]] = {}


def invalidate_cache(user_id: int | None = None) -> int:
    """
    새 파일 업로드 후 호출 — 해당 사용자의 캐시만 정확히 삭제합니다.

    user_id 지정 → 그 사용자의 캐시만 삭제 (다른 사용자 캐시 유지)
    user_id=None → 전체 캐시 삭제
    반환값: 삭제된 캐시 항목 수
    """
    global _cache_data, _cache_index

    if user_id is None:
        count = len(_cache_data)
        _cache_data = {}
        _cache_index = {}
        return count

    keys_to_delete = _cache_index.pop(user_id, set())
    for k in keys_to_delete:
        _cache_data.pop(k, None)
    return len(keys_to_delete)


def run_query(question: str, user_id: int | None = None, cat_id: int | None = None) -> dict:
    key = hashlib.md5(f"{question}_{user_id}_{cat_id}".encode()).hexdigest()

    # 캐시 히트
    if key in _cache_data:
        return _cache_data[key]

    docs, metas = get_retriever().retrieve(question, user_id)
    if not docs:
        return {
            "status": "error",
            "message": "관련 문서를 찾을 수 없습니다.",
            "answer": None,
            "references": [],
        }

    answer = get_llm().invoke(_build_prompt(docs, metas, question))
    answer_text = answer.content if hasattr(answer, "content") else str(answer)

    # references: 파일명·카테고리·페이지 포함
    refs = [
        {
            "doc_name": metas[i].get("doc_name") or metas[i].get("file_name", "?"),
            "category": metas[i].get("category", ""),
            "page": metas[i].get("page_num", ""),
            "snippet": docs[i][:200] + "...",
        }
        for i in range(len(docs))
    ]

    result = {"status": "success", "answer": answer_text, "references": refs}

    # 캐시 저장 + user_id 역인덱스 등록
    _cache_data[key] = result
    if user_id is not None:
        _cache_index.setdefault(user_id, set()).add(key)

    return result


# 하위 호환
class ImprovedRAGPipeline:
    def query_with_context(self, query: str, user_id=None, cat_id=None, top_k=5) -> dict:
        return run_query(query, user_id, cat_id)
