"""
PDF / DOCX 추출기
 - PDF  : Docling (레이아웃) + pypdfium2 (텍스트) + paddleocr (이미지/차트)
 - DOCX : Docling (레이아웃 + 텍스트 + 표)

"""

import gc

import cv2
import fitz
import numpy as np
import pypdfium2 as pdfium

from app.ocr.paddleocr_engine import create_reader, preprocess_image, readtext_filtered
from app.ocr.utils import BATCH_SIZE, OCR_ZOOM, SAVE_CROPS, clean_text


# ── pypdfium2: bbox 영역 텍스트 추출 ─────────────────────────────────────────
def extract_text_with_pypdfium2(pdf_pdfium, item) -> str:
    try:
        if not item.prov:
            return ""
        prov = item.prov[0]
        bbox = prov.bbox
        page = pdf_pdfium[prov.page_no - 1]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded(
            left=bbox.l,
            bottom=bbox.b,
            right=bbox.r,
            top=bbox.t,
        )
        textpage.close()
        return text.strip()
    except Exception:
        return ""


# ── 이미지 크롭 ───────────────────────────────────────────────────────────────
def crop_image(pdf_doc, item, padding=15):
    try:
        prov = item.prov[0]
        page_no = prov.page_no
        bbox = prov.bbox
        page = pdf_doc.load_page(page_no - 1)
        page_height = page.rect.height

        rect = fitz.Rect(
            bbox.l - padding,
            page_height - bbox.t - padding,
            bbox.r + padding,
            page_height - bbox.b + padding,
        ).normalize()
        rect = rect & page.rect

        if rect.width < 10 or rect.height < 10:
            return None, None

        pix = page.get_pixmap(matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM), clip=rect)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        crop_filename = f"crop_page{page_no}_{bbox.l:.0f}_{bbox.b:.0f}.png"

        if SAVE_CROPS:
            pix.save(crop_filename)
        pix = None

        return img_bgr, crop_filename
    except Exception as e:
        print(f"     크롭 실패: {e}")
        return None, None


# ── PaddleOCR 실행 ──────────────────────────────────────────────────────────────
def run_ocr(ocr, pdf_doc, item, label: str) -> str | None:
    page_no = item.prov[0].page_no
    img_bgr, crop_filename = crop_image(pdf_doc, item)

    if img_bgr is None:
        print(f"  Page {page_no} [{label}] bbox 너무 작음 → skip")
        return None

    try:
        img_bgr = preprocess_image(img_bgr)
        lines = readtext_filtered(ocr, img_bgr)
        img_bgr = None
        gc.collect()

        extracted = clean_text("\n".join(lines))

        if extracted.strip():
            formatted = extracted.strip().replace("\n", "\n> ")
            print(f"  Page {page_no} [{label}] OCR 성공!")
            return f"\n\n> {formatted}\n\n"
        else:
            msg = f"{crop_filename}" if SAVE_CROPS else "SAVE_CROPS=True 로 설정하면 확인 가능"
            print(f"  Page {page_no} [{label}] OCR 결과 없음 ({msg})")
            return None

    except Exception as e:
        print(f"  Page {page_no} [{label}] OCR 오류: {e}")
        return None


# ── Docling PDF 배치 변환 ─────────────────────────────────────────────────────
def convert_pdf_batch(pdf_path: str, batch_pages: list):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
    result = converter.convert(
        pdf_path,
        page_range=(min(batch_pages), max(batch_pages)),
    )
    return result.document


# ── PDF 처리 ──────────────────────────────────────────────────────────────────
def process_pdf(pdf_path: str, page_filter: set = None) -> list[str]:
    ocr = create_reader()
    pdf_doc = fitz.open(pdf_path)
    pdf_doc_pdfium = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf_doc_pdfium)
    blocks = []

    if page_filter:
        batches = [sorted(page_filter)]
    else:
        all_pages = list(range(1, total_pages + 1))
        batches = [all_pages[i : i + BATCH_SIZE] for i in range(0, len(all_pages), BATCH_SIZE)]

    for batch_idx, batch_pages in enumerate(batches):
        print(f"  배치 {batch_idx + 1}/{len(batches)}  페이지 {batch_pages}")
        try:
            doc = convert_pdf_batch(pdf_path, batch_pages)
        except Exception as e:
            print(f"  변환 실패 → skip: {e}")
            gc.collect()
            continue

        all_items = list(doc.iterate_items())
        all_items.sort(
            key=lambda x: (
                x[0].prov[0].page_no if x[0].prov else float("inf"),
                -x[0].prov[0].bbox.t if x[0].prov else 0,
                x[0].prov[0].bbox.l if x[0].prov else 0,
            )
        )

        for item, level in all_items:
            label = item.label.name

            if page_filter:
                if not item.prov:
                    continue
                if item.prov[0].page_no not in page_filter:
                    continue

            if label in [
                "TEXT",
                "CAPTION",
                "PARAGRAPH",
                "TITLE",
                "DOCUMENT_INDEX",
                "SECTION_HEADER",
                "LIST_ITEM",
            ]:
                text = extract_text_with_pypdfium2(pdf_doc_pdfium, item)
                if not text:
                    text = getattr(item, "text", None) or ""
                if text:
                    if label in ("SECTION_HEADER", "TITLE"):
                        blocks.append(f"### {clean_text(text)}\n\n")
                    elif label == "LIST_ITEM":
                        blocks.append(f"{clean_text(text)}\n")
                    else:
                        blocks.append(f"{clean_text(text)}\n\n")

            elif label == "TABLE":
                if hasattr(item, "export_to_markdown"):
                    blocks.append(f"\n{item.export_to_markdown(doc=doc)}\n\n")

            elif label in ("PICTURE", "CHART"):
                block = run_ocr(ocr, pdf_doc, item, label)
                if block:
                    blocks.append(block)

        del doc
        gc.collect()

    pdf_doc.close()
    pdf_doc_pdfium.close()
    return blocks


# ── DOCX 처리 ─────────────────────────────────────────────────────────────────
def process_docx(file_path: str) -> list[str]:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError("pip install docling")

    print("  Docling으로 DOCX 파싱 중...")
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document
        blocks = []

        for item, level in doc.iterate_items():
            label = item.label.name
            text = getattr(item, "text", None) or ""

            if label in ("TEXT", "CAPTION", "PARAGRAPH", "DOCUMENT_INDEX"):
                if text:
                    blocks.append(f"{clean_text(text)}\n\n")
            elif label in ("SECTION_HEADER", "TITLE"):
                if text:
                    blocks.append(f"### {clean_text(text)}\n\n")
            elif label == "LIST_ITEM":
                if text:
                    blocks.append(f"{clean_text(text)}\n")
            elif label == "TABLE":
                if hasattr(item, "export_to_markdown"):
                    blocks.append(f"\n{item.export_to_markdown(doc=doc)}\n\n")

        print(f"  Docling DOCX 완료 ({len(blocks)}개 블록)")
        return blocks
    except Exception as e:
        print(f"  Docling DOCX 오류: {e}")
        return [f"[Docling DOCX 파싱 실패: {e}]\n\n"]


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────
def extract(file_path: str, pages: list = None) -> str:
    """
    PDF 또는 DOCX 파일에서 마크다운 텍스트 추출.

    Args:
        file_path : PDF 또는 DOCX 파일 경로
        pages     : PDF 전용 페이지 필터 (1-indexed). None이면 전체.

    Returns:
        마크다운 문자열
    """
    from pathlib import Path

    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        page_filter = set(pages) if pages else None
        return "".join(process_pdf(file_path, page_filter))
    elif ext == ".docx":
        return "".join(process_docx(file_path))
    else:
        raise ValueError(f"지원하지 않는 포맷: {ext} (pdf, docx만 가능)")
