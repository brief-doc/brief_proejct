"""qa_data JSON → ChromaDB qa_collection 로드 (임베딩 없이 직접 저장)"""

import json
import sys
from pathlib import Path

import chromadb

from .config import CHROMA_DB_PATH


def _extract(data: dict) -> str:
    parts = []
    info = data.get("info", {})
    if v := info.get("caseNm"):
        parts.append(f"사건명: {v}")
    if v := info.get("caseTitle"):
        parts.append(f"판결: {v}")
    for item in data.get("jdgmnInfo", []):
        if q := item.get("question"):
            parts.append(f"Q: {q}")
        if a := item.get("answer"):
            parts.append(f"A: {a}")
    for item in data.get("Summary", []):
        if s := item.get("summ_pass"):
            parts.append(f"요약: {s}")
    kws = [k.get("keyword") for k in data.get("keyword_tagg", []) if k.get("keyword")]
    if kws:
        parts.append(f"키워드: {', '.join(kws)}")
    return "\n".join(parts)


def load_qa_data(limit: int = None, batch_size: int = 100) -> dict:
    """qa_data 폴더의 JSON 파일을 ChromaDB에 로드"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_or_create_collection("qa_collection")
    existing = set(col.get()["ids"])

    qa_dir = Path(__file__).parent.parent / "qa_data"
    if not qa_dir.exists():
        return {"status": "error", "detail": f"qa_data 디렉토리 없음: {qa_dir}"}

    files = sorted(qa_dir.glob("*.json"))[:limit]
    ids, texts, metas, new, skip = [], [], [], 0, 0

    for idx, f in enumerate(files):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            case_no = data.get("info", {}).get("caseNo") or f.stem
            if case_no in existing:
                skip += 1
                continue
            ids.append(case_no)
            texts.append(_extract(data))
            metas.append(
                {
                    "source": f.name,
                    "case_no": case_no,
                    "user_id": 1,
                    "doc_type": "qa_data",
                }
            )
            new += 1
            if len(ids) >= batch_size:
                col.upsert(ids=ids, documents=texts, metadatas=metas)
                ids, texts, metas = [], [], []
                print(f"  저장: {idx + 1}/{len(files)} | 신규 {new} | 스킵 {skip}")
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")

    if ids:
        col.upsert(ids=ids, documents=texts, metadatas=metas)

    total = col.count()
    print(f"완료 — 신규 {new}개 | 스킵 {skip}개 | DB 총 {total}개")
    return {"status": "success", "new": new, "skipped": skip, "total": total}


if __name__ == "__main__":
    result = load_qa_data()
    sys.exit(0 if result["status"] == "success" else 1)
