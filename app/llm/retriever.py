"""하이브리드 검색 + 크로스인코더 리랭킹 — LangChain BaseRetriever 구현

LangChain LCEL 파이프라인에 바로 끼워 쓸 수 있습니다:
    chain = get_retriever() | format_docs | prompt | llm | StrOutputParser()

레고 교체:
    - 벡터 스토어 교체 → vectorstore.py 의 get_vectorstore() 변경
    - 리랭킹 모델 교체 → config.py 의 RERANKER_MODEL 변경
    - BM25 비율 조정  → combined = α * bm25 + (1-α) * ce_scores 수정
    - BM25 비활성화   → rank_bm25 미설치 시 자동 fallback

선택적 패키지:
    rank_bm25             → pip install rank-bm25
    sentence_transformers → pip install sentence-transformers
"""

from typing import Any

import numpy as np

# ── 선택적 패키지 (langchain_chroma 보다 먼저 임포트해야 segfault 방지) ──────
try:
    from rank_bm25 import BM25Okapi

    _BM25_OK = True
except ImportError:
    _BM25_OK = False
    print("[retriever] rank_bm25 없음 → 벡터 검색만 사용 (pip install rank-bm25 권장)")

# ── CrossEncoder: chromadb + sentence_transformers 동시 로드 시 segfault 발생 ──
# 두 라이브러리가 같은 프로세스에 공존하면 httpx HTTP 호출 시 메모리 충돌.
# 리랭킹이 필요하면 ENABLE_RERANKER=true 환경변수로 명시적으로 활성화하세요.
import os as _os

_RERANKER_ENABLED = _os.getenv("ENABLE_RERANKER", "false").lower() == "true"

if _RERANKER_ENABLED:
    try:
        from sentence_transformers import CrossEncoder

        _CE_OK = True
    except ImportError:
        _CE_OK = False
        print("[retriever] sentence_transformers 없음 → 리랭킹 생략")
else:
    _CE_OK = False
    print("[retriever] CrossEncoder 비활성 (ENABLE_RERANKER=true 로 활성화 가능)")

from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from .config import RERANKER_MODEL, TOP_K_FINAL, TOP_K_RETRIEVE  # noqa: E402
from .vectorstore import get_vectorstore  # noqa: E402

# CrossEncoder 싱글톤
_reranker: "CrossEncoder | None" = None


def _get_reranker() -> "CrossEncoder | None":
    global _reranker
    if _reranker is None and _CE_OK:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _normalize(arr: np.ndarray) -> np.ndarray:
    span = arr.max() - arr.min()
    return (arr - arr.min()) / (span + 1e-9)


# ── LangChain BaseRetriever 구현 ──────────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    """하이브리드(Vector + BM25) + CrossEncoder 리랭킹 Retriever

    LangChain LCEL 인터페이스를 구현하므로 파이프라인에 직접 연결 가능합니다.

    Attributes:
        user_id: 사용자별 문서 필터 (None 이면 전체 문서 검색)
    """

    user_id: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        vs = get_vectorstore()
        total = vs._collection.count()
        print(f"[retriever] DB 총 청크 수: {total}, user_id={self.user_id}, 쿼리: {query[:60]!r}")

        # user_id 필터: 정수·문자열 양쪽 시도 후 전체 폴백
        filters_to_try: list[dict[str, Any] | None] = []
        if self.user_id is not None:
            filters_to_try.append({"user_id": int(self.user_id)})
            filters_to_try.append({"user_id": str(self.user_id)})
        filters_to_try.append(None)

        docs: list[Document] = []
        for f in filters_to_try:
            try:
                kwargs: dict[str, Any] = {"k": TOP_K_RETRIEVE}
                if f is not None:
                    kwargs["filter"] = f
                results = vs.similarity_search(query, **kwargs)
                print(f"[retriever] filter={f} → {len(results)}개")
                if results:
                    docs = results
                    if f is None and self.user_id is not None:
                        print("[retriever] user_id 필터 결과 없음 → 전체 검색 fallback")
                    break
            except Exception as e:
                print(f"[retriever] 쿼리 오류(filter={f}): {e}")
                continue

        if not docs:
            print("[retriever] 검색 결과 없음")
            return []

        doc_names = [d.metadata.get("doc_name") or d.metadata.get("file_name", "?") for d in docs]
        print(f"[retriever] 검색된 문서: {doc_names}")

        texts = [d.page_content for d in docs]

        # ── BM25 점수 계산 ────────────────────────────────────────────────────
        if _BM25_OK:
            try:
                tokenized = [t.split() or ["_empty_"] for t in texts]
                bm25_scores = _normalize(
                    np.array(
                        BM25Okapi(tokenized).get_scores(query.split() or ["_"]),
                        dtype=float,
                    )
                )
            except Exception:
                bm25_scores = np.ones(len(docs))
        else:
            bm25_scores = np.ones(len(docs))

        # ── CrossEncoder 리랭킹 ───────────────────────────────────────────────
        reranker = _get_reranker()
        if reranker:
            ce_scores = _normalize(
                np.array(
                    reranker.predict([[query, t] for t in texts]),
                    dtype=float,
                )
            )
            combined = 0.3 * bm25_scores + 0.7 * ce_scores
        else:
            combined = bm25_scores

        top_idx = np.argsort(combined)[::-1][:TOP_K_FINAL]
        return [docs[i] for i in top_idx]


# ── 팩토리 함수 ───────────────────────────────────────────────────────────────
_default_retriever: HybridRetriever | None = None


def get_retriever(user_id: int | None = None) -> HybridRetriever:
    """HybridRetriever 반환.

    user_id 없으면 싱글톤을 반환하고,
    user_id 있으면 필터가 적용된 새 인스턴스를 반환합니다.
    """
    global _default_retriever
    if user_id is not None:
        return HybridRetriever(user_id=user_id)
    if _default_retriever is None:
        _default_retriever = HybridRetriever()
        mode = "hybrid" if _BM25_OK else "vector-only"
        rerank = "+rerank" if _CE_OK else ""
        print(f"[retriever] {mode}{rerank} 초기화 완료")
    return _default_retriever
