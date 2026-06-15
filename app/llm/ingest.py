"""문서 저장 통합 모듈 — chunker + vectorstore 레고 조합

레고 교체:
    - 청킹 전략 변경 → chunker.py 의 markdown_splitter 교체
    - 벡터 스토어 변경 → vectorstore.py 의 get_vectorstore() 교체
    - 임베딩 변경 → embeddings.py 의 get_embeddings() 교체

흐름:
    마크다운: split_by_headers / markdown_splitter → metadata 주입 → vectorstore.add_documents()
    PDF·DOCX·HWP 등: app.ocr.extractor.process_document() → 마크다운 → 위 흐름과 동일
"""

from .chunker import markdown_splitter, split_by_headers
from .vectorstore import get_vectorstore, reset_vectorstore


# ── 마크다운 저장 ─────────────────────────────────────────────────────────────
def ingest_markdown(
    markdown_content: str,
    doc_id: int,
    user_id: int,
    doc_name: str = "마크다운 문서",
    chunking_method: str = "sections",
    category: str = "민사법",
    enable_summary: bool = False,  # 하위 호환 파라미터 (미사용)
) -> dict:
    """마크다운 문서를 청킹하여 ChromaDB에 저장합니다.

    Args:
        chunking_method: "sections" (헤더 기준) | "size" (글자 수 기준)
        category:        민사법 / 행정법 / 형사법 / 지식재산권
    """
    if chunking_method == "sections":
        raw_chunks = split_by_headers(markdown_content)
    else:
        raw_chunks = markdown_splitter.create_documents([markdown_content])

    chunks = [c for c in raw_chunks if len(c.page_content.strip()) > 10]
    if not chunks:
        return {"status": "error", "detail": "청킹 결과 없음"}

    for i, chunk in enumerate(chunks):
        section = chunk.metadata.get("section") or chunk.metadata.get("h1") or chunk.metadata.get("h2") or chunk.metadata.get("h3") or "본문"
        chunk.metadata.update(
            {
                "doc_id": str(doc_id),
                "user_id": str(user_id),
                "doc_name": doc_name,
                "section": section,
                "category": category,
                "chunk_id": f"{doc_id}_md_{i}",
            }
        )

    try:
        get_vectorstore().add_documents(chunks)
        reset_vectorstore()  # 다음 검색 시 새 문서가 반영되도록 싱글톤 리셋
        print(f"[ingest_md] 저장 완료: {doc_name}, {len(chunks)}청크, user_id={user_id}")
    except Exception as e:
        print(f"[ingest_md] 벡터 저장 실패: {e}")
        return {"status": "error", "detail": str(e)}

    return {
        "status": "success",
        "total_chunks": len(chunks),
        "category": category,
    }


# ── 하위 호환 ─────────────────────────────────────────────────────────────────
def save_markdown_to_vector_db(
    markdown_content: str,
    doc_id: int,
    user_id: int,
    doc_name: str = "마크다운 문서",
    chunking_method: str = "sections",
    enable_summary: bool = True,  # 하위 호환 파라미터 (현재 미사용)
) -> dict:
    return ingest_markdown(markdown_content, doc_id, user_id, doc_name, chunking_method)
