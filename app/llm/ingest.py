"""문서 저장 통합 모듈 — chunker + vectorstore 레고 조합

레고 교체:
    - 청킹 전략 변경 → chunker.py 의 pdf_splitter / markdown_splitter 교체
    - 벡터 스토어 변경 → vectorstore.py 의 get_vectorstore() 교체
    - 임베딩 변경 → embeddings.py 의 get_embeddings() 교체

흐름:
    PDF    : PyPDFLoader → pdf_splitter → metadata 주입 → vectorstore.add_documents()
    마크다운: split_by_headers / markdown_splitter → metadata 주입 → vectorstore.add_documents()
"""

import glob
import os

from langchain_core.documents import Document

from .chunker import markdown_splitter, pdf_splitter, split_by_headers
from .vectorstore import get_vectorstore, reset_vectorstore


def _load_pdf(file_path: str) -> list[Document]:
    """PDF를 페이지별 Document 로 로드합니다.

    추출 전략 (순서대로 시도):
        1. pdfplumber  — 텍스트 레이어 추출 (정확도 높음)
        2. pypdf       — 폴백
        3. pytesseract — 스캔 이미지형 PDF OCR 폴백

    langchain_community 는 일부 환경에서 segfault 를 유발하므로 직접 구현합니다.
    """
    docs: list[Document] = []

    # ── 1단계: pdfplumber ─────────────────────────────────────────────────────
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path, "page": i + 1},
                    )
                )
        # 의미 있는 텍스트가 있으면 바로 반환
        if any(len(d.page_content) > 20 for d in docs):
            return docs
    except Exception as e:
        print(f"[ingest] pdfplumber 실패: {e}")

    # ── 2단계: pypdf 폴백 ─────────────────────────────────────────────────────
    docs = []
    try:
        import pypdf

        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path, "page": i + 1},
                    )
                )
        if any(len(d.page_content) > 20 for d in docs):
            return docs
    except Exception as e:
        print(f"[ingest] pypdf 폴백 실패: {e}")

    # ── 3단계: pytesseract OCR 폴백 (스캔 이미지형 PDF) ──────────────────────
    print("[ingest] 텍스트 레이어 없음 → pytesseract OCR 시도")
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(file_path, dpi=200)
        docs = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang="kor+eng").strip()
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "page": i + 1, "ocr": True},
                )
            )
        return docs
    except Exception as e:
        print(f"[ingest] OCR 폴백 실패: {e}")

    # 모두 실패해도 빈 페이지라도 반환 (에러 처리는 호출부에서)
    return docs if docs else [Document(page_content="", metadata={"source": file_path})]


# ── PDF 저장 ──────────────────────────────────────────────────────────────────
def ingest_pdf(
    file_path: str,
    doc_id: int,
    user_id: int,
    doc_name: str | None = None,
    category: str = "기타",
) -> dict:
    """PDF를 청킹하여 ChromaDB에 저장합니다.

    Args:
        file_path: PDF 파일 경로
        doc_id:    문서 고유 ID
        user_id:   업로드한 사용자 ID
        doc_name:  표시 이름 (미지정 시 파일명 사용)
        category:  법령·조례 / 가이드라인·지침 / 공모·사업 / 감사 / 내부 규정 / 기타
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return {"status": "error", "detail": "파일이 없거나 비어 있습니다"}

    try:
        pages = _load_pdf(file_path)
    except Exception as e:
        return {"status": "error", "detail": f"PDF 로드 실패: {e}"}

    name: str = doc_name or os.path.basename(file_path)
    chunks: list[Document] = []

    for pi, page in enumerate(pages):
        for ci, chunk in enumerate(pdf_splitter.split_documents([page])):
            chunk.metadata.update(
                {
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "file_name": name,
                    "page_num": pi + 1,
                    "chunk_id": f"{doc_id}_p{pi + 1}_c{ci}",
                    "category": category,
                }
            )
            chunks.append(chunk)

    if not chunks:
        return {"status": "error", "detail": "청킹 결과가 없습니다"}

    try:
        get_vectorstore().add_documents(chunks)
        reset_vectorstore()  # 다음 검색 시 새 문서가 반영되도록 싱글톤 리셋
        print(f"[ingest_pdf] 저장 완료: {name}, {len(chunks)}청크")
    except Exception as e:
        print(f"[ingest_pdf] 벡터 저장 실패: {e}")
        return {"status": "error", "detail": f"벡터 저장 실패: {e}"}

    return {
        "status": "success",
        "file_name": name,
        "category": category,
        "total_chunks": len(chunks),
        "total_pages": len(pages),
    }


# ── 마크다운 저장 ─────────────────────────────────────────────────────────────
def ingest_markdown(
    markdown_content: str,
    doc_id: int,
    user_id: int,
    doc_name: str = "마크다운 문서",
    chunking_method: str = "sections",
    category: str = "기타",
    enable_summary: bool = False,  # 하위 호환 파라미터 (미사용)
) -> dict:
    """마크다운 문서를 청킹하여 ChromaDB에 저장합니다.

    Args:
        chunking_method: "sections" (헤더 기준) | "size" (글자 수 기준)
        category:        법령·조례 / 가이드라인·지침 / 공모·사업 / 감사 / 내부 규정 / 기타
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


# ── 배치 PDF 저장 ─────────────────────────────────────────────────────────────
def batch_ingest_pdfs(
    pdf_folder: str = "./pdf_files",
    start_doc_id: int = 0,
    user_id: int = 1,
) -> dict:
    """폴더 내 모든 PDF를 일괄 저장합니다."""
    files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not files:
        return {"status": "error", "detail": f"PDF 없음: {pdf_folder}"}

    results, total = [], 0
    for idx, path in enumerate(files):
        r = ingest_pdf(path, start_doc_id + idx, user_id)
        total += r.get("total_chunks", 0)
        results.append(r)

    return {"status": "success", "total_chunks": total, "results": results}


# ── 하위 호환 (기존 코드 수정 없이 사용 가능) ─────────────────────────────────
def save_pdf_to_vector_db(file_path: str, doc_id: int, user_id: int) -> dict:
    return ingest_pdf(file_path, doc_id, user_id)


def save_markdown_to_vector_db(
    markdown_content: str,
    doc_id: int,
    user_id: int,
    doc_name: str = "마크다운 문서",
    chunking_method: str = "sections",
    enable_summary: bool = True,  # 하위 호환 파라미터 (현재 미사용)
) -> dict:
    return ingest_markdown(markdown_content, doc_id, user_id, doc_name, chunking_method)


def batch_save_to_vector_db(
    files_data: list,
    start_doc_id: int,
    user_id: int,
) -> dict:
    results, total = [], 0
    for idx, f in enumerate(files_data):
        r = ingest_pdf(
            f.get("file_path", ""),
            start_doc_id + idx,
            user_id,
            f.get("file_name"),
        )
        total += r.get("total_chunks", 0)
        results.append(r)
    return {"status": "success", "total_chunks": total, "results": results}
