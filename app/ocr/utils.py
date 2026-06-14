"""
공통 유틸리티
- clean_text: 텍스트 후처리
- 공통 설정값
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── 공통 설정 ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 5  # Docling 배치 크기 (RAM 부족 시 2~3으로 줄이기)
OCR_ZOOM = 2  # 크롭 이미지 확대 배율
SAVE_CROPS = False  # 크롭 이미지 저장 여부 (디버깅 시 True로)
LLAMA_API_KEY = "llx-YOUR_API_KEY_HERE"  # ← LlamaParse API 키 입력(doc 작업할시)

# 지원 확장자
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".hwp", ".hwpx", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def clean_text(text: str) -> str:
    """앞뒤 공백 제거 및 빈 줄 정리."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        cleaned.append(line) if line else cleaned.append("")
    return "\n".join(cleaned)
