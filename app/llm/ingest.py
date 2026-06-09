"""문서 저장 통합 모듈 — PDF / 마크다운 청킹 → ChromaDB 저장"""

import glob
import os
from datetime import datetime

import chromadb
import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHROMA_DB_PATH, EMBEDDING_CONFIG, LLM_CONFIG

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
_MD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", "。", "，", ""]
)
_OLLAMA_URL = LLM_CONFIG.get("base_url", "http://localhost:11434")
_MODEL = LLM_CONFIG.get("model", "gemma3n:e2b")


# ── 임베딩 모델 싱글톤 (최초 1회만 로드) ──────────────────────────────────────
# _embeddings() 를 매번 호출할 때마다 새 모델을 생성하면 BGE-M3(570MB) 를
# 호출마다 다시 로드하게 되어 수십 초가 낭비됩니다.
# 모듈 레벨 싱글톤으로 관리하여 서버 수명 동안 한 번만 로드합니다.
_EMBEDDINGS_INSTANCE: HuggingFaceEmbeddings | None = None


def _embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        print(f"[ingest] 임베딩 모델 최초 로드: {EMBEDDING_CONFIG['model_name']}")
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model_name"],
            model_kwargs={"device": EMBEDDING_CONFIG.get("device", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[ingest] 임베딩 모델 로드 완료 (이후 재사용)")
    return _EMBEDDINGS_INSTANCE


from .config import COLLECTION_NAME  # noqa: E402  retriever 와 동일 컬렉션 사용


def _chroma_vs(collection: str | None = None):
    """retriever.py 와 동일한 COLLECTION_NAME 을 기본값으로 사용"""
    col = collection or COLLECTION_NAME
    return Chroma(
        persist_directory=CHROMA_DB_PATH, embedding_function=_embeddings(), collection_name=col
    )


def _qa_collection():
    """COLLECTION_NAME 컬렉션 반환 — 없으면 자동 생성"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME)


# ── PDF 저장 ──────────────────────────────────────────────────────────────────
def ingest_pdf(
    file_path: str, doc_id: int, user_id: int, doc_name: str = None, category: str = "기타"
) -> dict:
    """
    PDF를 청킹하여 ChromaDB에 저장합니다.

    Args:
        file_path: PDF 파일 경로
        doc_id:    문서 고유 ID
        user_id:   업로드한 사용자 ID
        doc_name:  문서 표시 이름 (미지정 시 파일명 사용)
        category:  문서 카테고리 (법령·조례 / 가이드라인·지침 / 공모·사업 /
                   감사 / 내부 규정 / 기타)
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return {"status": "error", "detail": "파일이 없거나 비어 있습니다"}
    try:
        documents = PyPDFLoader(file_path).load()
    except Exception as e:
        return {"status": "error", "detail": f"PDF 로드 실패: {e}"}

    name = doc_name or os.path.basename(file_path)
    chunks = []
    for pi, doc in enumerate(documents):
        for ci, chunk in enumerate(_SPLITTER.split_documents([doc])):
            chunk.metadata.update(
                {
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "file_name": name,
                    "page_num": pi + 1,
                    "chunk_id": f"{doc_id}_p{pi + 1}_c{ci}",
                    "category": category,  # ← 카테고리 메타 추가
                }
            )
            chunks.append(chunk)
    try:
        # retriever 와 동일 컬렉션(COLLECTION_NAME)에 저장
        Chroma.from_documents(
            chunks,
            _embeddings(),
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
        )
    except Exception as e:
        return {"status": "error", "detail": f"벡터 저장 실패: {e}"}
    return {
        "status": "success",
        "file_name": name,
        "category": category,
        "total_chunks": len(chunks),
        "total_pages": len(documents),
    }


# ── PDF 배치 저장 ─────────────────────────────────────────────────────────────
def batch_ingest_pdfs(
    pdf_folder: str = "./pdf_files", start_doc_id: int = 0, user_id: int = 1
) -> dict:
    files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not files:
        return {"status": "error", "detail": f"PDF 없음: {pdf_folder}"}

    vs = _chroma_vs()
    results, total = [], 0
    for idx, path in enumerate(files):
        name = os.path.basename(path)
        doc_id = start_doc_id + idx
        try:
            docs = PyPDFLoader(path).load()
            chunks = _SPLITTER.split_documents(docs)
            for i, c in enumerate(chunks):
                c.metadata.update(
                    {
                        "doc_id": doc_id,
                        "user_id": user_id,
                        "batch_processed_at": datetime.now().isoformat(),
                    }
                )
            vs.add_documents(chunks)
            total += len(chunks)
            results.append({"file": name, "chunks": len(chunks), "status": "success"})
        except Exception as e:
            results.append({"file": name, "chunks": 0, "status": "error", "detail": str(e)})
    return {"status": "success", "total_chunks": total, "results": results}


# ── 마크다운 청킹 ─────────────────────────────────────────────────────────────
def _chunk_by_sections(content: str) -> list[dict]:
    chunks, cur, section = [], [], "본문"
    for line in content.split("\n"):
        if line.startswith("#"):
            if cur:
                text = "\n".join(cur).strip()
                if len(text) > 10:
                    chunks.append({"section": section, "content": text})
            section = line.lstrip("#").strip()
            cur = [line]
        else:
            cur.append(line)
    if cur:
        text = "\n".join(cur).strip()
        if len(text) > 10:
            chunks.append({"section": section, "content": text})
    return chunks


def _summarize(text: str) -> str:
    try:
        r = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={
                "model": _MODEL,
                "stream": False,
                "prompt": f"200자 이내 요약:\n{text[:800]}\n요약:",
                "temperature": 0.3,
            },
            timeout=60,
        )
        return r.json().get("response", "").strip()[:200] if r.status_code == 200 else "[요약 실패]"
    except Exception:
        return "[요약 실패]"


def ingest_markdown(
    markdown_content: str,
    doc_id: int,
    user_id: int,
    doc_name: str = "마크다운 문서",
    chunking_method: str = "sections",
    enable_summary: bool = True,
    category: str = "기타",
) -> dict:
    """
    마크다운 문서를 청킹하여 ChromaDB qa_collection에 저장합니다.

    Args:
        category: 문서 카테고리 (법령·조례 / 가이드라인·지침 / 공모·사업 /
                  감사 / 내부 규정 / 기타)
    """
    chunks = (
        _chunk_by_sections(markdown_content)
        if chunking_method == "sections"
        else [
            {"section": "N/A", "content": c.strip()}
            for c in _MD_SPLITTER.split_text(markdown_content)
            if len(c.strip()) > 10
        ]
    )
    if not chunks:
        return {"status": "error", "detail": "청킹 결과 없음"}
    if enable_summary:
        for c in chunks:
            c["summary"] = _summarize(c["content"])
    try:
        col = _qa_collection()
        col.add(
            ids=[f"{doc_id}_md_{i}" for i in range(len(chunks))],
            documents=[c["content"] for c in chunks],
            metadatas=[
                {
                    "doc_id": str(doc_id),
                    "user_id": str(user_id),
                    "doc_name": doc_name,
                    "section": c.get("section", "N/A"),
                    "category": category,
                }
                for c in chunks
            ],
        )
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    return {"status": "success", "total_chunks": len(chunks), "category": category}


# ── 하위 호환 ─────────────────────────────────────────────────────────────────
def save_pdf_to_vector_db(file_path: str, doc_id: int, user_id: int) -> dict:
    return ingest_pdf(file_path, doc_id, user_id)


def save_markdown_to_vector_db(
    markdown_content: str,
    doc_id: int,
    user_id: int,
    doc_name: str = "마크다운 문서",
    chunking_method: str = "sections",
    enable_summary: bool = True,
) -> dict:
    return ingest_markdown(
        markdown_content, doc_id, user_id, doc_name, chunking_method, enable_summary
    )


def batch_save_to_vector_db(files_data: list, start_doc_id: int, user_id: int) -> dict:
    results, total = [], 0
    for idx, f in enumerate(files_data):
        path = f.get("file_path", "")
        r = ingest_pdf(path, start_doc_id + idx, user_id, f.get("file_name"))
        total += r.get("total_chunks", 0)
        results.append({**r, "file_name": f.get("file_name", os.path.basename(path))})
    return {"status": "success", "total_chunks": total, "results": results}
