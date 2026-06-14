"""마크다운 처리 — ingest.py + chunker.py 로 위임 (하위 호환 유지)

새 코드에서는 이 파일 대신 아래를 직접 사용하세요:
    from .ingest import ingest_markdown
    from .chunker import split_by_headers, markdown_splitter
"""

from .chunker import markdown_splitter, split_by_headers
from .ingest import ingest_markdown


class MarkdownProcessor:
    """하위 호환용 래퍼 클래스 — 내부적으로 ingest.py / chunker.py 를 사용합니다."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, **kwargs: object) -> None:
        # kwargs 는 이전 버전 호환용 (ollama_url, llm_model 등) — 무시
        pass

    def chunk_by_sections(self, content: str) -> list[dict]:
        """섹션별 청킹 (마크다운 헤더 기준)"""
        docs = split_by_headers(content)
        return [
            {
                "id": i,
                "section": doc.metadata.get("section", "본문"),
                "content": doc.page_content,
                "char_count": len(doc.page_content),
            }
            for i, doc in enumerate(docs)
            if len(doc.page_content.strip()) > 10
        ]

    def chunk_by_size(self, content: str) -> list[dict]:
        """크기별 청킹 (RecursiveCharacterTextSplitter)"""
        docs = markdown_splitter.create_documents([content])
        return [
            {"id": i, "content": doc.page_content, "char_count": len(doc.page_content)}
            for i, doc in enumerate(docs)
            if len(doc.page_content.strip()) > 10
        ]

    def process_markdown(
        self,
        markdown_content: str,
        doc_id: int,
        user_id: int,
        doc_name: str,
        chunking_method: str = "sections",
        enable_summary: bool = True,
        save_to_db: bool = True,
    ) -> dict:
        """마크다운 문서 처리 (청킹 → 저장)"""
        if not save_to_db:
            chunks = self.chunk_by_sections(markdown_content) if chunking_method == "sections" else self.chunk_by_size(markdown_content)
            return {
                "status": "success",
                "total_chunks": len(chunks),
                "chunks": chunks,
                "detail": "마크다운 처리 완료 (DB 저장 안 함)",
            }

        return ingest_markdown(
            markdown_content,
            doc_id,
            user_id,
            doc_name,
            chunking_method,
        )


def process_markdown_text(
    markdown_text: str,
    doc_id: int,
    user_id: int,
    doc_name: str,
    chunking_method: str = "sections",
    enable_summary: bool = True,
) -> dict:
    """편의 함수 — 하위 호환"""
    return ingest_markdown(markdown_text, doc_id, user_id, doc_name, chunking_method)
