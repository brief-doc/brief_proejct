"""ChromaDB VectorStore 싱글톤

레고 교체:
    get_vectorstore() 의 반환 타입만 맞추면 다른 벡터 DB로 교체 가능합니다.

    예) FAISS 로 교체:
        from langchain_community.vectorstores import FAISS
        ...

    reset_vectorstore() — 새 문서 ingest 후 강제 재연결이 필요할 때 호출
"""

from langchain_chroma import Chroma

from .config import CHROMA_DB_PATH, COLLECTION_NAME
from .embeddings import get_embeddings

_instance: Chroma | None = None


def get_vectorstore() -> Chroma:
    """ChromaDB VectorStore 싱글톤 반환"""
    global _instance
    if _instance is None:
        import os

        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _instance = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
        count = _instance._collection.count()
        print(f"[vectorstore] 초기화 완료 — 컬렉션: {COLLECTION_NAME}, 저장 청크 수: {count}")
    return _instance


def reset_vectorstore() -> None:
    """싱글톤 초기화 — 인제스트 후 반드시 호출해야 검색에 새 문서가 반영됨"""
    global _instance
    _instance = None
    print("[vectorstore] 싱글톤 리셋 — 다음 검색 시 DB 재로드")


def get_document_count() -> int:
    """컬렉션에 저장된 총 청크 수 반환"""
    return get_vectorstore()._collection.count()
