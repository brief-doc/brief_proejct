"""
카테고리별 문서 요약 모듈

지원 카테고리:
  법령·조례 / 가이드라인·지침 / 공모·사업 / 감사 / 내부 규정 / 기타

공통 규칙:
  - 원문에 명시된 내용만 사용 (추측·외부 지식 금지)
  - 공공기관 행정 격식체(합쇼체) 사용
  - 원문 인용 시 따옴표(" ") + 조항·페이지 병기
  - 수치·날짜·고유명사는 원문 그대로 표기
  - 해당 항목 없으면 "해당 내용 없음" 표기

긴 문서 처리 전략 (Map-Reduce):
  - 문서를 CHUNK_SIZE 단위로 분할
  - 각 청크를 개별 요약 (Map)
  - 부분 요약들을 합쳐 최종 요약 생성 (Reduce)
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm import get_llm

# Map 단계 병렬 워커 수 (Ollama는 단일 프로세스라 3이 적당)
_MAP_WORKERS = int(os.getenv("MAP_WORKERS", "3"))

# gemma2:2b 컨텍스트 4096 토큰 기준
# 시스템 프롬프트·템플릿 ~500자, 출력 여유 ~500자 → 입력 최대 약 3000자
CHUNK_SIZE = 2500  # 청크당 최대 글자 수 (여유 있게 설정)
CHUNK_OVERLAP = 200  # 청크 간 겹침 (문맥 연속성 유지)

# ── 공통 시스템 지시 ──────────────────────────────────────────────────────────
_SYSTEM_RULES = """당신은 공공기관 행정 문서 요약 전문가입니다. 반드시 아래 규칙을 따르십시오.
규칙 1. 원문에 명시된 내용만 사용하고, 추측이나 외부 지식을 추가하지 마십시오.
규칙 2. 공공기관 행정 격식체(합쇼체: ~하십시오, ~입니다)를 사용하십시오.
규칙 3. 원문 인용 시 따옴표(" ")로 감싸고 조항·페이지를 병기하십시오.
         예시: "위반 시 과태료 500만원" (제12조 제2항)
규칙 4. 수치·날짜·고유명사는 원문 그대로 표기하십시오.
규칙 5. 해당 항목이 원문에 없으면 반드시 "해당 내용 없음"으로 표기하십시오."""

# ── 카테고리별 추출 항목 템플릿 ───────────────────────────────────────────────
_TEMPLATES: dict[str, str] = {
    "법령·조례": """\
아래 항목을 번호 순서대로 반드시 추출하여 작성하십시오.
1. 법령명 및 제정·개정 일자
2. 목적 및 적용 범위
3. 주요 의무·금지 사항 (조항 번호 포함)
4. 위반 시 제재·벌칙
5. 시행 기관 및 담당 부서""",
    "가이드라인·지침": """\
아래 항목을 번호 순서대로 반드시 추출하여 작성하십시오.
1. 지침명 및 발행 기관·발행 일자
2. 적용 대상
3. 핵심 절차 (단계별로 작성)
4. 준수 기준 및 예외 사항
5. 문의처 및 담당 부서""",
    "공모·사업": """\
아래 항목을 번호 순서대로 반드시 추출하여 작성하십시오.
1. 사업명 및 주관 기관
2. 지원 대상 및 자격 요건
3. 지원 규모·금액
4. 신청 기간 및 신청 방법
5. 선정 기준 및 주요 일정""",
    "감사": """\
아래 항목을 번호 순서대로 반드시 추출하여 작성하십시오.
1. 감사 기관 및 피감사 기관
2. 감사 기간 및 감사 범위
3. 주요 지적 사항 (항목별로 작성)
4. 조치 요구 사항 및 이행 기한
5. 처분 결과 및 후속 조치""",
    "내부 규정": """\
아래 항목을 번호 순서대로 반드시 추출하여 작성하십시오.
1. 규정명 및 제정·개정 일자
2. 적용 범위
3. 주요 내용 (조항별로 핵심만 작성)
4. 담당 부서 및 책임자
5. 시행일""",
    "기타": """\
아래 항목을 번호 순서대로 반드시 추출하여 작성하십시오.
1. 문서명 및 발행 기관
2. 핵심 내용 요약 (3~5문장)
3. 주요 일정 또는 기한
4. 관련 기관·부서
5. 특이 사항""",
}

# 지원 카테고리 목록 (외부에서 참조용)
CATEGORIES = list(_TEMPLATES.keys())


def build_summary_prompt(doc_text: str, category: str) -> str:
    """
    카테고리에 맞는 요약 프롬프트를 생성합니다.

    Args:
        doc_text: 요약할 원문 텍스트
        category: 문서 카테고리 (CATEGORIES 중 하나, 없으면 '기타' 사용)

    Returns:
        LLM에 전달할 완성된 프롬프트 문자열
    """
    template = _TEMPLATES.get(category, _TEMPLATES["기타"])
    return f"""{_SYSTEM_RULES}

[문서 카테고리] {category}

[추출 항목]
{template}

[원문]
{doc_text}

[요약 결과]"""


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """텍스트를 chunk_size 단위로 분할합니다 (overlap으로 문맥 유지)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _map_summarize(chunk: str, chunk_idx: int, total: int, category: str) -> str:
    """단일 청크를 요약합니다 (Map 단계)."""
    prompt = f"""{_SYSTEM_RULES}

[문서 카테고리] {category}
[청크 정보] 전체 {total}개 중 {chunk_idx + 1}번째 부분

아래 원문 내용의 핵심 정보를 빠짐없이 추출하여 요약하십시오.
수치·날짜·고유명사·조항 번호는 원문 그대로 표기하십시오.

[원문]
{chunk}

[부분 요약]"""
    try:
        result = get_llm().invoke(prompt)
        return result.content.strip() if hasattr(result, "content") else str(result).strip()
    except Exception as e:
        return f"[청크 {chunk_idx + 1} 요약 실패: {e}]"


def _reduce_summarize(partial_summaries: list[str], category: str) -> str:
    """부분 요약들을 합쳐 최종 카테고리별 요약을 생성합니다 (Reduce 단계)."""
    template = _TEMPLATES.get(category, _TEMPLATES["기타"])
    combined = "\n\n".join(f"[부분 요약 {i + 1}]\n{s}" for i, s in enumerate(partial_summaries))
    prompt = f"""{_SYSTEM_RULES}

[문서 카테고리] {category}

아래는 하나의 문서를 여러 부분으로 나눠 요약한 결과들입니다.
이를 종합하여 최종 요약을 작성하십시오.

[추출 항목]
{template}

[부분 요약 모음]
{combined}

[최종 요약 결과]"""
    try:
        result = get_llm().invoke(prompt)
        return result.content.strip() if hasattr(result, "content") else str(result).strip()
    except Exception:
        # Reduce 실패 시 부분 요약 병합으로 대체
        return "\n\n".join(partial_summaries)


def summarize_document(doc_text: str, category: str) -> dict:
    """
    문서를 카테고리별 형식에 맞게 요약합니다.
    문서 길이에 관계없이 전체 내용을 요약합니다.

    짧은 문서 (≤ CHUNK_SIZE):  직접 요약
    긴 문서   (>  CHUNK_SIZE):  Map-Reduce 방식
      ① 청크 분할 → 각 청크 요약 (Map)
      ② 부분 요약 합산 → 최종 요약 (Reduce)

    Returns:
        {
            "status": "success" | "error",
            "category": str,
            "summary": str,
            "chunks_used": int,   # 처리된 청크 수
            "message": str        # 오류 시
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

    resolved_category = category if category in _TEMPLATES else "기타"

    try:
        # ── 짧은 문서: 직접 요약 ─────────────────────────────────
        if len(doc_text) <= CHUNK_SIZE:
            prompt = build_summary_prompt(doc_text, resolved_category)
            result = get_llm().invoke(prompt)
            summary = result.content.strip() if hasattr(result, "content") else str(result).strip()
            return {
                "status": "success",
                "category": resolved_category,
                "summary": summary,
                "chunks_used": 1,
            }

        # ── 긴 문서: Map-Reduce ───────────────────────────────────
        chunks = _split_text(doc_text)
        total = len(chunks)

        # Map: 각 청크 병렬 요약 (ThreadPoolExecutor)
        # Ollama는 단일 프로세스이므로 _MAP_WORKERS 스레드가 동시 요청 → 큐잉됨
        # 직렬 대비 전체 대기 시간이 유사하거나 빠름 (스레드 오버헤드 최소)
        partial_summaries: list[str | None] = [None] * total
        with ThreadPoolExecutor(max_workers=_MAP_WORKERS) as pool:
            future_to_idx = {
                pool.submit(_map_summarize, chunk, i, total, resolved_category): i
                for i, chunk in enumerate(chunks)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                partial_summaries[idx] = fut.result()

        # Reduce: 부분 요약 → 최종 요약
        # 부분 요약 합계도 너무 길면 한 번 더 청크 분할 (병렬)
        combined_text = "\n\n".join(partial_summaries)  # type: ignore[arg-type]
        if len(combined_text) > CHUNK_SIZE:
            second_chunks = _split_text(combined_text)
            sec_total = len(second_chunks)
            second_summaries: list[str | None] = [None] * sec_total
            with ThreadPoolExecutor(max_workers=_MAP_WORKERS) as pool:
                future_to_idx2 = {
                    pool.submit(_map_summarize, c, i, sec_total, resolved_category): i
                    for i, c in enumerate(second_chunks)
                }
                for fut in as_completed(future_to_idx2):
                    idx = future_to_idx2[fut]
                    second_summaries[idx] = fut.result()
            partial_summaries = second_summaries  # type: ignore[assignment]

        final_summary = _reduce_summarize(partial_summaries, resolved_category)

        return {
            "status": "success",
            "category": resolved_category,
            "summary": final_summary,
            "chunks_used": total,
        }

    except Exception as e:
        return {
            "status": "error",
            "category": resolved_category,
            "summary": "",
            "chunks_used": 0,
            "message": f"요약 생성 실패: {e}",
        }
