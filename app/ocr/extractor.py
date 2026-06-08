"""
문서 추출 진입점
확장자 확인 후 적절한 추출기로 라우팅

지원 포맷:
    .pdf  .docx  → extractor_pdf_docx.py  (Docling + pypdfium2 + EasyOCR)
    .doc         → extractor_doc.py        (LlamaParse)
    .hwp  .hwpx  → extractor_hwp.py        (rhwp-python + EasyOCR)

사용 예시:
    from extractor import process_document

    md = process_document("가이드라인.pdf")
    md = process_document("법령.docx")
    md = process_document("문서.doc")
    md = process_document("의안.hwp")
    md = process_document("의안.hwpx")
"""

import sys
from pathlib import Path

from app.ocr.utils import SUPPORTED_EXTENSIONS


def process_document(file_path: str, pages: list = None) -> str:
    """
    확장자에 맞는 추출기를 호출하고 마크다운 문자열 반환.

    Args:
        file_path : 문서 경로
        pages     : PDF 전용 페이지 필터 (1-indexed 리스트). None이면 전체.
                    예) [3, 4] → 3, 4페이지만

    Returns:
        마크다운 형식의 추출 텍스트.
        지원하지 않는 포맷이면 빈 문자열("") 반환.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    print(f"\n처리 시작: {path.name}  [{ext.upper()}]")

    # ── PDF / DOCX ────────────────────────────────────────────────────────────
    if ext in (".pdf", ".docx"):
        from app.OCR.extractor_pdf_docx import extract

        if ext == ".pdf":
            print("  엔진: Docling + pypdfium2 + EasyOCR")
        else:
            print("  엔진: Docling")
        return extract(file_path, pages=pages)

    # ── DOC ───────────────────────────────────────────────────────────────────
    elif ext == ".doc":
        from app.OCR.extractor_doc import extract

        print("  엔진: LlamaParse")
        return extract(file_path)

    # ── HWP / HWPX ───────────────────────────────────────────────────────────
    elif ext in (".hwp", ".hwpx"):
        from app.OCR.extractor_hwp import extract

        print("  엔진: rhwp-python + EasyOCR")
        return extract(file_path)

    # ── 미지원 포맷 ───────────────────────────────────────────────────────────
    else:
        print(f"  지원하지 않는 포맷: {ext}")
        print(f"  지원 포맷: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return ""


# ── CLI 실행 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test2.pdf"
    pages = None
    if len(sys.argv) > 2:
        pages = [int(p) for p in sys.argv[2].split(",")]

    markdown = process_document(path, pages=pages)

    if markdown:
        print(f"\n--- 추출 결과 미리보기 ({len(markdown):,} chars) ---")
        print(markdown[:500])
        print("..." if len(markdown) > 500 else "")

        out = Path(path).with_suffix(".md").name
        Path(out).write_text(markdown, encoding="utf-8")
        print(f"\n저장: {out}")
