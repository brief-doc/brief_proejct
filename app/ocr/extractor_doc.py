"""
DOC 추출기
 - DOC : LlamaParse (클라우드 API)

"""

from app.OCR.utils import LLAMA_API_KEY, clean_text


def process_doc(file_path: str) -> list[str]:
    try:
        from llama_parse import LlamaParse
    except ImportError:
        raise ImportError("pip install llama-parse")

    print("  LlamaParse API 호출 중...")

    try:
        parser = LlamaParse(
            api_key=LLAMA_API_KEY,
            language="ko",
            verbose=False,
        )
    except TypeError:
        parser = LlamaParse(api_key=LLAMA_API_KEY)

    try:
        documents = parser.load_data(file_path)
        blocks = []

        for doc in documents:
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
            print("  LlamaParse 결과 없음")
            if documents:
                doc = documents[0]
                print(f"     type: {type(doc)}")
                print(
                    f"     attrs: {[a for a in dir(doc) if not a.startswith('_')][:15]}"
                )
        return blocks

    except Exception as e:
        print(f"  LlamaParse 오류: {e}")
        import traceback

        traceback.print_exc()
        return [f"[LlamaParse 파싱 실패: {e}]\n\n"]


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────
def extract(file_path: str, **kwargs) -> str:
    """
    DOC 파일에서 마크다운 텍스트 추출 (LlamaParse).

    Args:
        file_path : DOC 파일 경로

    Returns:
        마크다운 문자열
    """
    from pathlib import Path

    ext = Path(file_path).suffix.lower()

    if ext != ".doc":
        raise ValueError(f"지원하지 않는 포맷: {ext} (doc만 가능)")

    return "".join(process_doc(file_path))
