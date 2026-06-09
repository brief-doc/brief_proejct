"""
하이브리드 검색 + 크로스인코더 리랭킹

rank_bm25 미설치 시 → 벡터 검색만 사용 (자동 fallback)
sentence_transformers 미설치 시 → 리랭킹 생략 (자동 fallback)

⚠️ 중요: ingest.py 와 동일한 EMBEDDING_CONFIG 를 사용해야
          저장/검색 벡터 공간이 일치합니다.
"""

import chromadb
import numpy as np

from .config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_CONFIG,
    RERANKER_MODEL,
    TOP_K_FINAL,
    TOP_K_RETRIEVE,
)

# chromadb HuggingFaceEmbeddingFunction 은 원격 HF API 를 호출하므로 사용 안 함
# → langchain_huggingface 로컬 모델을 chromadb EF 인터페이스로 래핑해서 사용
try:
    from langchain_huggingface import HuggingFaceEmbeddings as _LCEmbeddings

    class _STEmbeddingFunction:
        """로컬 BGE-M3 모델을 chromadb EF 인터페이스로 래핑 (API 키 불필요)"""

        def __init__(self):
            self._model = _LCEmbeddings(
                model_name=EMBEDDING_CONFIG["model_name"],
                model_kwargs={"device": EMBEDDING_CONFIG.get("device", "cpu")},
                encode_kwargs={
                    "normalize_embeddings": EMBEDDING_CONFIG.get("normalize_embeddings", True)
                },
            )
            print(f"[retriever] 로컬 임베딩 모델 로드: {EMBEDDING_CONFIG['model_name']}")

        # chromadb 최신 버전이 요구하는 메서드
        def name(self) -> str:
            return f"local-{EMBEDDING_CONFIG['model_name'].replace('/', '-')}"

        def __call__(self, input: list[str]) -> list:
            return self._model.embed_documents(input)

    _ST_OK = True
except ImportError:
    _ST_OK = False
    print("[retriever] ❌ langchain_huggingface 없음 → pip install langchain-huggingface")

# 선택적 패키지
try:
    from rank_bm25 import BM25Okapi

    _BM25_OK = True
except ImportError:
    _BM25_OK = False
    print("[retriever] rank_bm25 없음 → 벡터 검색만 사용 (pip install rank-bm25 권장)")

try:
    from sentence_transformers import CrossEncoder

    _CE_OK = True
except ImportError:
    _CE_OK = False
    print("[retriever] sentence_transformers 없음 → 리랭킹 생략")

_retriever = None


def get_retriever() -> "HybridRetriever":
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


class HybridRetriever:
    def __init__(self):
        # 로컬 BGE-M3 임베딩 함수 (API 키 불필요)
        if not _ST_OK:
            raise RuntimeError(
                "langchain_huggingface 없음. pip install langchain-huggingface 실행하세요."
            )
        ef = _STEmbeddingFunction()

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # 컬렉션 가져오기 (없으면 생성)
        try:
            self.col = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            self.col = client.create_collection(name=COLLECTION_NAME)
            print(f"[retriever] 컬렉션 '{COLLECTION_NAME}' 새로 생성")

        self._ef = ef

        # 검색 방식 자동 결정: BGE-M3 테스트 쿼리로 차원 호환 여부 확인
        self._use_texts = False  # 기본: BGE-M3 직접 임베딩
        if self.col.count() > 0:
            try:
                test_emb = ef(["차원 테스트"])
                self.col.query(query_embeddings=test_emb, n_results=1)
                print(f"[retriever] BGE-M3 임베딩 검색 사용 (dim={len(test_emb[0])})")
            except Exception as dim_err:
                if "dimension" in str(dim_err).lower():
                    self._use_texts = True
                    print("[retriever] 차원 불일치 감지 -> chromadb 기본 임베딩으로 검색")
                    print("[retriever] 권장: step4_rebuild_chromadb.py 로 BGE-M3 재구축")
                else:
                    raise

        self.reranker = CrossEncoder(RERANKER_MODEL) if _CE_OK else None
        mode = "hybrid" if _BM25_OK else "vector-only"
        rerank = "+rerank" if self.reranker else ""
        search = "query_texts" if self._use_texts else "BGE-M3"
        print(f"[retriever] {mode} {rerank} | {search} | docs={self.col.count()}")

    def _vector_search(self, query: str, n: int, user_id):
        n_results = min(n, self.col.count() or 1)

        # user_id 필터: 정수·문자열 양쪽 시도 (저장 시 타입이 다를 수 있음)
        # 우선순위: user_id 일치 → 전체 문서 폴백
        filters_to_try: list = []
        if user_id is not None:
            filters_to_try.append({"user_id": {"$eq": int(user_id)}})  # 정수
            filters_to_try.append({"user_id": {"$eq": str(user_id)}})  # 문자열
        filters_to_try.append(None)  # 최종 폴백: 필터 없이 전체 검색

        def _run_query(where):
            base_kwargs = {"n_results": n_results}
            if where:
                base_kwargs["where"] = where
            if self._use_texts:
                return self.col.query(query_texts=[query], **base_kwargs)
            else:
                return self.col.query(query_embeddings=self._ef([query]), **base_kwargs)

        for where in filters_to_try:
            try:
                res = _run_query(where)
                docs = res["documents"][0]
                if docs:  # 결과가 있으면 즉시 반환
                    if where is None and user_id is not None:
                        print(f"[retriever] user_id={user_id} 필터 결과 없음 → 전체 문서 검색")
                    return docs, res["metadatas"][0]
            except Exception as e:
                print(f"[retriever] 쿼리 오류(where={where}): {e}")
                continue

        return [], []

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        span = arr.max() - arr.min()
        return (arr - arr.min()) / (span + 1e-9)

    def retrieve(self, query: str, user_id=None) -> tuple[list[str], list[dict]]:
        docs, metas = self._vector_search(query, TOP_K_RETRIEVE, user_id)
        if not docs:
            return [], []

        # BM25 점수 (설치된 경우)
        if _BM25_OK:
            try:
                tokenized = [d.split() for d in docs]
                # 빈 토큰 문서 방지 (ZeroDivisionError 방지)
                tokenized = [t if t else ["_empty_"] for t in tokenized]
                bm25 = self._normalize(
                    np.array(BM25Okapi(tokenized).get_scores(query.split() or ["_"]), dtype=float)
                )
            except Exception:
                bm25 = np.ones(len(docs))
        else:
            bm25 = np.ones(len(docs))  # 동일 가중치 fallback

        # CrossEncoder 리랭킹 (설치된 경우)
        if self.reranker:
            ce = self._normalize(
                np.array(self.reranker.predict([[query, d] for d in docs]), dtype=float)
            )
            combined = 0.3 * bm25 + 0.7 * ce
        else:
            combined = bm25  # BM25만 또는 동일 가중치

        top_idx = np.argsort(combined)[::-1][:TOP_K_FINAL]
        return [docs[i] for i in top_idx], [metas[i] for i in top_idx]
