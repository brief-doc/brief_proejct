"""카테고리별 문서 요약 — LangChain LCEL 기반

속도 최적화 전략 (3가지):
    1. 병렬 Map  : ThreadPoolExecutor 로 청크별 요약을 동시 실행
    2. 전용 LLM  : num_predict=300 으로 줄인 요약 전용 LLM 사용
    3. 큰 청크   : CHUNK_SIZE=4000 으로 청크 수 자체를 줄임

짧은 문서 (≤ CHUNK_SIZE 글자):  단일 호출 (stuff)
긴 문서   (>  CHUNK_SIZE 글자):  Map-Reduce (병렬 처리)

레고 교체:
    - LLM 교체         → llm.py 의 get_summary_llm() 변경
    - Map 프롬프트 교체 → prompts.py 의 MAP_PROMPT 변경
    - 카테고리 템플릿   → prompts.py 의 SUMMARY_TEMPLATES 변경
    - 병렬 수 변경      → config.py 의 SUMMARY_MAP_WORKERS 변경
    - 청킹 설정 변경    → CHUNK_SIZE / CHUNK_OVERLAP 상수 변경

공개 API:
    classify_document_category(doc_text)     → str   (자동 카테고리 분류)
    summarize_document(doc_text, category) → dict
    build_summary_prompt(doc_text, category) → str   (하위 호환)
    CATEGORIES                               (하위 호환)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.output_parsers import StrOutputParser

from .chunker import SimpleCharacterSplitter
from .config import SUMMARY_MAP_WORKERS
from .llm import get_summary_llm
from .prompts import CLASSIFY_PROMPT, MAP_PROMPT, SUMMARY_TEMPLATES, get_reduce_prompt  # noqa: F401

# ── 설정 ──────────────────────────────────────────────────────────────────────
# CHUNK_SIZE 를 크게 잡을수록 청크 수가 줄어 Map 호출 횟수 감소 → 빨라짐
CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", "4000"))
CHUNK_OVERLAP = int(os.getenv("SUMMARY_CHUNK_OVERLAP", "200"))

_splitter = SimpleCharacterSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

_parser = StrOutputParser()

# Map 단계 병렬 처리용 Executor (프로세스 재사용으로 오버헤드 최소화)
_map_executor = ThreadPoolExecutor(max_workers=SUMMARY_MAP_WORKERS)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────
def _invoke_map(chunk_text: str) -> str:
    """Map 단계: 단일 청크를 요약합니다 (요약 전용 LLM 사용)."""
    return (MAP_PROMPT | get_summary_llm() | _parser).invoke({"text": chunk_text})


def _invoke_reduce(combined: str, category: str) -> str:
    """Reduce 단계: 부분 요약들을 합산하여 최종 요약을 생성합니다."""
    return (get_reduce_prompt(category) | get_summary_llm() | _parser).invoke({"text": combined})


def _parallel_map(chunks: list[str]) -> list[str]:
    """청크 목록을 ThreadPoolExecutor 로 병렬 요약합니다.

    순서 보장: future → index 매핑으로 원래 청크 순서를 유지합니다.
    """
    results: list[str] = [""] * len(chunks)
    futures = {_map_executor.submit(_invoke_map, chunk): i for i, chunk in enumerate(chunks)}
    for future in as_completed(futures):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            results[idx] = f"[Map 오류: {e}]"
    return results


# ── 카테고리 자동 분류 ────────────────────────────────────────────────────────
def classify_document_category(doc_text: str) -> str:
    """문서 내용을 LLM으로 분석하여 카테고리를 자동 분류합니다.

    Returns:
        "민사법" | "행정법" | "형사법" | "지식재산권"
    """
    valid = list(SUMMARY_TEMPLATES.keys())
    sample = doc_text[:2000].strip()
    try:
        result = (CLASSIFY_PROMPT | get_summary_llm() | _parser).invoke({"text": sample})
        classified = result.strip()
        if classified in valid:
            return classified
        for cat in valid:
            if cat in classified:
                return cat
    except Exception as e:
        print(f"[classify] 분류 실패: {e}")
    return valid[0]


# ── 메인 요약 함수 ────────────────────────────────────────────────────────────
def summarize_document(doc_text: str, category: str) -> dict:
    """문서를 카테고리별 형식에 맞게 요약합니다.

    Args:
        doc_text:  요약할 원문 텍스트
        category:  CATEGORIES 중 하나 (없으면 '기타' 사용)

    Returns:
        {
            "status":      "success" | "error",
            "category":    str,
            "summary":     str,
            "chunks_used": int,
            "message":     str   # 오류 시만 포함
        }
    """
    if not doc_text or not doc_text.strip():
        return {
            "status": "error",
            "category": category,
            "summary": "",
            "chunks_used": 0,
            "message": "원문이 비어 있습니다.",
        }

    resolved = category if category in SUMMARY_TEMPLATES else list(SUMMARY_TEMPLATES.keys())[0]

    try:
        chunks = _splitter.split_text(doc_text)
        chunks_used = len(chunks)

        if chunks_used == 1:
            # ── 짧은 문서: 직접 요약 (stuff) ─────────────────────────────
            summary = _invoke_reduce(doc_text, resolved)
        else:
            # ── 긴 문서: 병렬 Map-Reduce ──────────────────────────────────
            # Map: 청크별 부분 요약을 병렬 실행
            partial = _parallel_map(chunks)
            combined = "\n\n".join(f"[부분 요약 {i + 1}]\n{s}" for i, s in enumerate(partial))

            # 부분 요약도 너무 길면 한 번 더 병렬 Map
            if len(combined) > CHUNK_SIZE:
                second_chunks = _splitter.split_text(combined)
                partial = _parallel_map(second_chunks)
                combined = "\n\n".join(f"[부분 요약 {i + 1}]\n{s}" for i, s in enumerate(partial))

            # Reduce: 최종 요약 (단일 호출)
            summary = _invoke_reduce(combined, resolved)

        return {
            "status": "success",
            "category": resolved,
            "summary": summary,
            "chunks_used": chunks_used,
        }

    except Exception as e:
        return {
            "status": "error",
            "category": resolved,
            "summary": "",
            "chunks_used": 0,
            "message": f"요약 생성 실패: {e}",
        }


# ── 하위 호환 ─────────────────────────────────────────────────────────────────
def build_summary_prompt(doc_text: str, category: str) -> str:
    """프롬프트 문자열 반환 (하위 호환 — 새 코드에서는 prompts.get_reduce_prompt() 사용)"""
    return get_reduce_prompt(category).format(text=doc_text)
