"""진단 도구 — ChromaDB 상태 / DB 설정 / 라우트 확인"""

import sys

import chromadb

from .config import CHROMA_DB_PATH, DB_CONFIG


def check_chromadb():
    """ChromaDB 저장 현황 출력"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        for col in client.list_collections():
            data = col.get()
            total = len(data["ids"])
            print(f"컬렉션: {col.name} — {total}개 문서")
            if total:
                print(f"  샘플 메타: {data['metadatas'][0] if data['metadatas'] else 'N/A'}")
                preview = (data["documents"][0] if data["documents"] else "")[:100]
                print(f"  샘플 내용: {preview}...")
    except Exception as e:
        print(f"ChromaDB 오류: {e}")


def check_db_config():
    """DB_CONFIG 인코딩 진단"""
    for key, val in DB_CONFIG.items():
        print(f"[{key}] {val!r}  ({type(val).__name__})")
        if isinstance(val, str):
            try:
                val.encode("ascii")
                print("  → ASCII OK")
            except UnicodeEncodeError as e:
                print(f"  → 비ASCII 문자 발견 (위치 {e.start}: {val[e.start]!r})")


def check_routes():
    """등록된 FastAPI 라우트 출력"""
    try:
        from app.main import app

        for route in app.routes:
            methods = getattr(route, "methods", {"-"})
            print(f"{','.join(methods):8} {route.path}")
    except Exception as e:
        print(f"라우트 조회 실패: {e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("chromadb", "all"):
        print("=== ChromaDB ===")
        check_chromadb()
    if cmd in ("db", "all"):
        print("\n=== DB Config ===")
        check_db_config()
    if cmd in ("routes", "all"):
        print("\n=== Routes ===")
        check_routes()
