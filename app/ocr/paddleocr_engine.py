# 버전 : paddlepaddle==3.0.0
import sys
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR, PPStructureV3

# ── 0. 크롭 이미지 OCR 엔진 (extractor_pdf_docx, extractor_hwp 용) ──────────


def preprocess_image(img_bgr):
    return cv2.convertScaleAbs(img_bgr, alpha=1.3, beta=10)


def create_reader(langs: list = None, gpu: bool = False) -> PaddleOCR:
    return PaddleOCR(use_textline_orientation=True, lang="korean")


def readtext_filtered(reader: PaddleOCR, img_bgr, conf_threshold: float = 0.4) -> list:
    lines = []
    for res in reader.predict(img_bgr):
        for text, score in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
            if score >= conf_threshold and text.strip():
                lines.append(text.strip())
    return lines


# ── 1. 이미지 전처리 ────────────────────────────────────────────────────────


def preprocess(img_path: str) -> np.ndarray:
    """
    PPStructureV3용 전처리.
    헤더(어두운 배경+흰 글씨)와 본문(밝은 배경+검정 글씨)을
    로컬 배경 밝기 기반으로 선택 반전하여 모두 검정 글씨+흰 배경으로 통일.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1단계: 표 선 추출 (원본 기준으로 보존)
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 8))
    lines_mask = cv2.bitwise_or(
        cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel_h),
        cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel_v),
    )

    # 2단계: 유채색·어두운 배경 감지 → 해당 영역만 그레이스케일 반전
    # S > 40: 파란/청록 등 유채색 배경 (그레이스케일 밝기만으론 구분 불가)
    # V < 110: 검정/매우 어두운 배경 (표 헤더 등)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    bg_candidate = cv2.bitwise_or(
        cv2.threshold(s_ch, 40, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(v_ch, 110, 255, cv2.THRESH_BINARY_INV)[1],
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 30), max(h // 40, 5)))
    bg_mask = cv2.morphologyEx(bg_candidate, cv2.MORPH_OPEN, open_kernel)
    bg_mask = cv2.dilate(bg_mask, np.ones((3, 3), np.uint8), iterations=1)
    normalized = np.where(bg_mask > 0, 255 - gray, gray).astype(np.uint8)

    # 3단계: CLAHE (대비 균일화)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 4단계: 원본 표 선 재합성
    enhanced[lines_mask > 0] = 0

    # 5단계: 2배 업스케일
    upscaled = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 6단계: 언샤프 마스킹 (획 경계 선명화)
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)

    # 7단계: 3채널 변환 (PPStructureV3 입력 형식)
    return cv2.cvtColor(np.clip(sharpened, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


# ── 2. HTML 테이블 → 마크다운 변환 ──────────────────────────────────────────


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list = []
        self._row: list = []
        self._cell_parts: list = []
        self._in_cell = False

    def handle_starttag(self, tag, _attrs):
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell_parts = []
            self._in_cell = True
        elif tag == "br" and self._in_cell:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th"):
            self._row.append(unescape(" ".join(self._cell_parts)).strip())
            self._in_cell = False
        elif tag == "tr":
            if self._row:
                self.rows.append(self._row[:])

    def handle_data(self, data):
        if self._in_cell and data.strip():
            self._cell_parts.append(data.strip())


def html_table_to_markdown(html: str) -> str:
    parser = _TableParser()
    parser.feed(html)

    if not parser.rows:
        return ""

    col_count = max(len(row) for row in parser.rows)
    md_rows = []
    for i, row in enumerate(parser.rows):
        cells = row + [""] * (col_count - len(row))
        md_rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * col_count) + " |")

    return "\n".join(md_rows)


# ── 3. 블록 현황 출력 (디버그용) ────────────────────────────────────────────


def print_block_summary(parsing_res_list: list):
    from collections import Counter

    counter = Counter(b.label for b in parsing_res_list)
    print("  [감지된 블록 현황]")
    for label, count in counter.most_common():
        print(f"    {label}: {count}개")


# ── 4. 블록 → 마크다운 변환 ────────────────────────────────────────────────

# 텍스트가 없거나 OCR 불가능한 블록 (건너뜀)
SKIP_LABELS = {"figure", "figure_caption", "formula", "chart", "seal", "unknown"}


def blocks_to_markdown(parsing_res_list: list) -> str:
    blocks = sorted(parsing_res_list, key=lambda b: b.index if b.index is not None else 9999)

    lines = []
    for block in blocks:
        content = str(block.content).strip()
        label = block.label

        print(f"  [DEBUG] label={label!r:20} content_type={type(block.content).__name__:10} content={repr(content[:80])}")

        if not content or label in SKIP_LABELS:
            continue

        if label == "table":
            lines.append(html_table_to_markdown(content))
        elif label == "doc_title":
            lines.append(f"# {content}")
        elif "title" in label:
            lines.append(f"## {content}")
        else:
            # text, paragraph, list, footnote 등 나머지 모든 텍스트 블록
            lines.append(content)

        lines.append("")

    return "\n".join(lines).strip()


# ── 5. 기본 OCR 폴백 (인포그래픽/지도 등 비문서 이미지용) ──────────────────


def fallback_ocr(img_path: str) -> str:
    """PPStructureV3로 추출 실패 시 기본 PaddleOCR로 텍스트만 추출."""
    print("  → 기본 PaddleOCR 폴백 실행 중...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="korean")
    results = ocr.predict(img_path)
    lines = []
    for res in results:
        for text, score in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
            if score >= 0.4 and text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


# ── 6. 공개 인터페이스 ──────────────────────────────────────────────────────


def extract(file_path: str) -> str:
    img_path = str(file_path)
    preprocessed = preprocess(img_path)

    pipeline = PPStructureV3(
        lang="korean",
        use_table_recognition=True,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )
    results = pipeline.predict(preprocessed)

    all_pages_md = []
    for res in results:
        parsing_res_list = res.get("parsing_res_list", [])
        if not parsing_res_list:
            continue
        has_table = any(b.label == "table" for b in parsing_res_list)
        if has_table:
            page_md = blocks_to_markdown(parsing_res_list)
        else:
            page_md = res.get("markdown_texts", "")
        if page_md:
            all_pages_md.append(page_md)

    if not all_pages_md:
        return fallback_ocr(img_path)

    return "\n\n---\n\n".join(all_pages_md)


# ── 7. 메인 ─────────────────────────────────────────────────────────────────


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "test2.pdf")

    print(f"[1/3] 이미지 전처리: {img_path}")
    preprocessed = preprocess(img_path)

    print("[2/3] PPStructureV3 실행 중...")
    pipeline = PPStructureV3(
        lang="korean",
        use_table_recognition=True,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )
    results = pipeline.predict(preprocessed)

    print("[3/3] 결과 처리 중...\n")

    all_pages_md = []

    for page_idx, res in enumerate(results):
        parsing_res_list = res.get("parsing_res_list", [])
        if not parsing_res_list:
            continue

        print(f"── 페이지 {page_idx + 1} ──")
        print_block_summary(parsing_res_list)

        has_table = any(b.label == "table" for b in parsing_res_list)

        if has_table:
            print("  → 표 블록 감지: 블록 단위 변환 사용")
            page_md = blocks_to_markdown(parsing_res_list)
        else:
            print("  → 표 없음: full_md 방식 사용")
            page_md = res.get("markdown_texts", "")

        print()
        if page_md:
            all_pages_md.append(page_md)

    if not all_pages_md:
        print("PPStructureV3 결과 없음 → 기본 OCR 폴백")
        final_md = fallback_ocr(img_path)
        if not final_md:
            print("결과를 추출하지 못했습니다.")
            return
    else:
        final_md = "\n\n---\n\n".join(all_pages_md)

    print("=" * 60)
    print(final_md)
    print("=" * 60)


if __name__ == "__main__":
    main()
