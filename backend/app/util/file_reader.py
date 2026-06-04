"""
문서 하이브리드 추출 파이프라인
 - 지원 포맷: PDF, DOCX, DOC, HWP, HWPX
 - PDF/DOCX  : Docling (레이아웃) + pypdfium2 (텍스트) + EasyOCR (이미지/차트)
 - DOC       : LlamaParse (클라우드 API)
 - HWP/HWPX  : rhwp-python (로컬, 한컴오피스 불필요)

설치:
    pip install pymupdf pypdfium2 docling easyocr llama-parse rhwp-python
    pip install "numpy==1.26.4" "opencv-python==4.6.0.66"

사양 최적화: RAM 16GB / VRAM 3GB / 저장소 224GB
"""

import gc
import os
import fitz
import numpy as np
import cv2
import warnings
from pathlib import Path

import pypdfium2 as pdfium
import easyocr

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── 설정 ─────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 5      # Docling 배치 크기 (RAM 부족 시 2~3으로 줄이기)
OCR_ZOOM        = 2      # 크롭 이미지 확대 배율
SAVE_CROPS      = False  # 크롭 이미지 저장 여부 (디버깅 시 True로)
LLAMA_API_KEY   = "llx-YOUR_API_KEY_HERE"   # ← LlamaParse API 키 입력


# ── 텍스트 후처리 ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        cleaned.append(line) if line else cleaned.append("")
    return "\n".join(cleaned)


# ══════════════════════════════════════════════════════════════════════════════
# LlamaParse: DOC
# ══════════════════════════════════════════════════════════════════════════════
def extract_with_llamaparse(file_path: str) -> list[str]:
    """
    LlamaParse 클라우드 API로 문서 파싱.
    DOC / HWP / HWPX 전용 (Docling 미지원 포맷).
    마크다운으로 반환 → 그대로 저장.
    """
    try:
        from llama_parse import LlamaParse
    except ImportError:
        raise ImportError("pip install llama-parse")

    print(f" LlamaParse API 호출 중...")

    # result_type 대신 최신 API: output_metadata 또는 파라미터 없이 기본값 사용
    try:
        parser = LlamaParse(
            api_key  = LLAMA_API_KEY,
            language = "ko",
            verbose  = False,
        )
    except TypeError:
        # 구버전 호환
        parser = LlamaParse(api_key=LLAMA_API_KEY)

    try:
        documents = parser.load_data(file_path)
        blocks = []
        for doc in documents:
            # 반환 구조 확인: text 또는 get_content()
            text = ""
            if hasattr(doc, "text") and doc.text:
                text = doc.text
            elif hasattr(doc, "get_content"):
                text = doc.get_content()
            elif hasattr(doc, "page_content"):
                text = doc.page_content

            if text.strip():
                blocks.append(clean_text(text) + "\n\n")

        if blocks:
            print(f"  LlamaParse 완료 ({len(blocks)}개 블록)")
        else:
            print(f"  LlamaParse 결과 없음 — 반환 구조 확인:")
            if documents:
                doc = documents[0]
                print(f"     type: {type(doc)}")
                print(f"     attrs: {[a for a in dir(doc) if not a.startswith('_')][:15]}")
        return blocks
    except Exception as e:
        print(f"  LlamaParse 오류: {e}")
        import traceback; traceback.print_exc()
        return [f"[LlamaParse 파싱 실패: {e}]\n\n"]


# ══════════════════════════════════════════════════════════════════════════════
# rhwp-python: HWP / HWPX
# ══════════════════════════════════════════════════════════════════════════════
def extract_hwp_with_rhwp(file_path: str, ocr_reader=None) -> list[str]:
    try:
        import rhwp
    except ImportError:
        raise ImportError("pip install rhwp-python")

    print("  rhwp-python으로 파싱 중...")
    try:
        doc = rhwp.parse(file_path)
        ir  = doc.to_ir()
        blocks = []

        def cell_text(cell) -> str:
            """TableCell에서 텍스트 추출. cell.blocks 안의 ParagraphBlock.text 사용."""
            parts = []
            for b in (getattr(cell, "blocks", None) or []):
                t = (getattr(b, "text", "") or "").strip()
                if t:
                    parts.append(t)
            return " ".join(parts)

        def process_block(block):
            kind = block.__class__.__name__

            # ── 단락 ─────────────────────────────────────────────────────────
            if kind == "ParagraphBlock":
                text = (getattr(block, "text", "") or "").strip()
                if not text:
                    return
                level = getattr(block, "outline_level", 0) or 0
                if level == 1:
                    blocks.append("## " + clean_text(text) + "\n\n")
                elif level >= 2:
                    blocks.append("### " + clean_text(text) + "\n\n")
                else:
                    blocks.append(clean_text(text) + "\n\n")

            # ── 리스트 항목 ───────────────────────────────────────────────────
            elif kind == "ListItemBlock":
                text = (getattr(block, "text", "") or "").strip()
                if text:
                    blocks.append(clean_text(text) + "\n")

            # ── 표 ───────────────────────────────────────────────────────────
            elif kind == "TableBlock":
                tbl_cells = getattr(block, "cells", []) or []
                if not tbl_cells:
                    return

                # row/col 인덱스 기준으로 그리드 구성
                row_map = {}
                for tc in tbl_cells:
                    tr = getattr(tc, "row", 0)
                    tc_col = getattr(tc, "col", 0)
                    if tr not in row_map:
                        row_map[tr] = {}
                    row_map[tr][tc_col] = cell_text(tc)

                md_rows = []
                sorted_rows = sorted(row_map.keys())
                for row_idx, tr_idx in enumerate(sorted_rows):
                    sorted_cols = sorted(row_map[tr_idx].keys())
                    row_cells   = [row_map[tr_idx][tc_idx] for tc_idx in sorted_cols]
                    md_rows.append("| " + " | ".join(row_cells) + " |")
                    if row_idx == 0:
                        md_rows.append("| " + " | ".join(["---"] * len(row_cells)) + " |")
                if md_rows:
                    blocks.append("\n" + "\n".join(md_rows) + "\n\n")

            # ── 이미지 ────────────────────────────────────────────────────────
            elif kind == "PictureBlock":
                # rhwp로 이미지 바이트 추출 → EasyOCR
                try:
                    img_bytes = doc.bytes_for_image(block)
                    if img_bytes:
                        import numpy as np
                        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        if img_bgr is not None:
                            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.3, beta=10)
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            results = ocr_reader.readtext(img_rgb)
                            lines = [
                                t.strip() for (_, t, conf) in results
                                if t.strip() and conf >= 0.4
                            ]
                            if lines:
                                text = clean_text("\n".join(lines))
                                formatted = text.replace("\n", "\n> ")
                                blocks.append(
                                    "\n> **[이미지 내 텍스트]**\n"
                                    "> " + formatted + "\n\n"
                                )
                except Exception as img_e:
                    print("  이미지 OCR 실패: " + str(img_e))

        for block in ir.body:
            process_block(block)

        print("  rhwp-python 완료 (" + str(len(blocks)) + "개 블록)")
        return blocks

    except Exception as e:
        print("  rhwp-python 오류: " + str(e))
        import traceback; traceback.print_exc()
        return ["[rhwp-python 파싱 실패: " + str(e) + "]\n\n"]


# ══════════════════════════════════════════════════════════════════════════════
# Docling: DOCX
# ══════════════════════════════════════════════════════════════════════════════
def extract_docx_with_docling(file_path: str) -> list[str]:
    """
    Docling으로 DOCX 파싱.
    표 구조, 헤더, 레이아웃까지 정확하게 추출.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError("pip install docling")

    print(f"  Docling으로 DOCX 파싱 중...")
    try:
        converter = DocumentConverter()
        result    = converter.convert(file_path)
        doc       = result.document
        blocks    = []

        for item, level in doc.iterate_items():
            label = item.label.name
            text  = getattr(item, 'text', None) or ""

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
                if hasattr(item, 'export_to_markdown'):
                    blocks.append(f"\n{item.export_to_markdown(doc=doc)}\n\n")

        print(f"  Docling DOCX 완료 ({len(blocks)}개 블록)")
        return blocks
    except Exception as e:
        print(f"  Docling DOCX 오류: {e}")
        return [f"[Docling DOCX 파싱 실패: {e}]\n\n"]


# ══════════════════════════════════════════════════════════════════════════════
# PDF 추출: Docling (레이아웃) + pypdfium2 (텍스트) + EasyOCR (이미지/차트)
# ══════════════════════════════════════════════════════════════════════════════
def extract_text_with_pypdfium2(pdf_pdfium, item) -> str:
    try:
        if not item.prov:
            return ""
        prov     = item.prov[0]
        bbox     = prov.bbox
        page     = pdf_pdfium[prov.page_no - 1]
        textpage = page.get_textpage()
        text     = textpage.get_text_bounded(
            left=bbox.l, bottom=bbox.b, right=bbox.r, top=bbox.t,
        )
        textpage.close()
        return text.strip()
    except Exception:
        return ""


def crop_image(pdf_doc, item, padding=15):
    try:
        prov        = item.prov[0]
        page_no     = prov.page_no
        bbox        = prov.bbox
        page        = pdf_doc.load_page(page_no - 1)
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

        img_bgr       = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr       = cv2.convertScaleAbs(img_bgr, alpha=1.3, beta=10)
        crop_filename = f"crop_page{page_no}_{bbox.l:.0f}_{bbox.b:.0f}.png"

        if SAVE_CROPS:
            pix.save(crop_filename)
        pix = None

        return img_bgr, crop_filename
    except Exception as e:
        print(f"     크롭 실패: {e}")
        return None, None


def run_ocr(ocr, pdf_doc, item, label: str) -> str | None:
    page_no = item.prov[0].page_no
    img_bgr, crop_filename = crop_image(pdf_doc, item)

    if img_bgr is None:
        print(f"  Page {page_no} [{label}] bbox 너무 작음 → skip")
        return None

    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = ocr.readtext(img_rgb)
        img_bgr = img_rgb = None
        gc.collect()

        lines = [
            text.strip()
            for (_, text, conf) in results
            if text.strip() and conf >= 0.4
        ]
        extracted = clean_text("\n".join(lines))

        if extracted.strip():
            # icon      = "📊" if label == "CHART" else "🖼️"
            type_name = "차트/그래프" if label == "CHART" else "이미지"
            formatted = extracted.strip().replace("\n", "\n> ")
            print(f"  Page {page_no} [{label}] OCR 성공!")
            return (
                f"\n> **[{type_name} 내 텍스트] (Page {page_no})**\n"
                f"> {formatted}\n\n"
            )
        else:
            msg = f"{crop_filename}" if SAVE_CROPS else "SAVE_CROPS=True 로 설정하면 확인 가능"
            print(f"  Page {page_no} [{label}] OCR 결과 없음 ({msg})")
            return None

    except Exception as e:
        print(f"  Page {page_no} [{label}] OCR 오류: {e}")
        return None


def convert_pdf_batch(pdf_path: str, batch_pages: list):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options        = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(
        pdf_path,
        page_range=(min(batch_pages), max(batch_pages)),
    )
    return result.document


def process_pdf(pdf_path: str, ocr, page_filter: set) -> list[str]:
    pdf_doc        = fitz.open(pdf_path)
    pdf_doc_pdfium = pdfium.PdfDocument(pdf_path)
    total_pages    = len(pdf_doc_pdfium)
    blocks         = []

    if page_filter:
        batches = [sorted(page_filter)]
    else:
        all_pages = list(range(1, total_pages + 1))
        batches   = [all_pages[i:i+BATCH_SIZE]
                     for i in range(0, len(all_pages), BATCH_SIZE)]

    for batch_idx, batch_pages in enumerate(batches):
        print(f"  배치 {batch_idx+1}/{len(batches)}  페이지 {batch_pages}")
        try:
            doc = convert_pdf_batch(pdf_path, batch_pages)
        except Exception as e:
            print(f"  변환 실패 → skip: {e}")
            gc.collect()
            continue

        for item, level in doc.iterate_items():
            label = item.label.name

            if page_filter:
                if not item.prov:
                    continue
                if item.prov[0].page_no not in page_filter:
                    continue

            if label in ["TEXT", "CAPTION", "PARAGRAPH", "TITLE",
                         "DOCUMENT_INDEX", "SECTION_HEADER", "LIST_ITEM"]:
                text = extract_text_with_pypdfium2(pdf_doc_pdfium, item)
                if not text:
                    text = getattr(item, 'text', None) or ""
                if text:
                    if label in ("SECTION_HEADER", "TITLE"):
                        blocks.append(f"### {clean_text(text)}\n\n")
                    elif label == "LIST_ITEM":
                        blocks.append(f"{clean_text(text)}\n")
                    else:
                        blocks.append(f"{clean_text(text)}\n\n")

            elif label == "TABLE":
                if hasattr(item, 'export_to_markdown'):
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


# ══════════════════════════════════════════════════════════════════════════════
# 메인 처리 함수
# ══════════════════════════════════════════════════════════════════════════════

# 지원 확장자 목록
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".hwp", ".hwpx"}


def extract_pdf(file_path: str, pages: list = None) -> str:
    """PDF 문서에서 마크다운 텍스트 추출."""
    page_filter = set(pages) if pages else None
    ocr         = easyocr.Reader(['ko', 'en'], gpu=False)
    blocks      = process_pdf(file_path, ocr, page_filter)
    return "".join(blocks)


def extract_docx(file_path: str) -> str:
    """DOCX 문서에서 마크다운 텍스트 추출."""
    blocks = extract_docx_with_docling(file_path)
    return "".join(blocks)


def extract_doc(file_path: str) -> str:
    """DOC 문서에서 마크다운 텍스트 추출 (LlamaParse)."""
    blocks = extract_with_llamaparse(file_path)
    return "".join(blocks)


def extract_hwp(file_path: str) -> str:
    """HWP / HWPX 문서에서 마크다운 텍스트 추출 (rhwp-python)."""
    ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
    blocks     = extract_hwp_with_rhwp(file_path, ocr_reader=ocr_reader)
    return "".join(blocks)


def process_document(file_path: str, pages: list = None) -> str:
    """
    확장자에 맞는 추출 함수를 호출하고 마크다운 문자열을 반환.

    팀원 연동용 인터페이스 — 파일 저장 없이 str 반환.

    Args:
        file_path : 문서 경로 (.pdf / .docx / .doc / .hwp / .hwpx)
        pages     : PDF 전용 페이지 필터 (1-indexed 리스트). None이면 전체.

    Returns:
        마크다운 형식의 추출 텍스트 문자열.
        지원하지 않는 포맷이면 빈 문자열("") 반환.

    Example:
        >>> from docling_ocr_fixed import process_document
        >>> md = process_document("가명정보_가이드라인.pdf")
        >>> md = process_document("의안원문.hwp")
        >>> md = process_document("법령.docx")
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    print(f"\n처리 시작: {path.name}  [{ext.upper()}]")

    if ext == ".pdf":
        print(f"  엔진: Docling + pypdfium2 + EasyOCR  "
              f"(BATCH={BATCH_SIZE} / ZOOM={OCR_ZOOM}x)")
        return extract_pdf(file_path, pages)

    elif ext == ".docx":
        print(f"  엔진: Docling")
        return extract_docx(file_path)

    elif ext == ".doc":
        print(f"  엔진: LlamaParse")
        return extract_doc(file_path)

    elif ext in (".hwp", ".hwpx"):
        print(f"  엔진: rhwp-python")
        return extract_hwp(file_path)

    else:
        print(f"  지원하지 않는 포맷: {ext}")
        print(f"  지원 포맷: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return ""


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test2.pdf"
    pages = None
    if len(sys.argv) > 2:
        pages = [int(p) for p in sys.argv[2].split(",")]

    markdown = process_document(path, pages=pages)

    if markdown:
        # 콘솔 미리보기 (앞 500자)
        print(f"\n--- 추출 결과 미리보기 ({len(markdown):,} chars) ---")
        print(markdown)
        # print("..." if len(markdown) > 500 else "")

        # 파일로도 저장 (확인용)
        # out = Path(path).with_suffix(".md").name
        # Path(out).write_text(markdown, encoding="utf-8")
        # print(f"\n저장: {out}")