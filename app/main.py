import asyncio
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

# Python 3.12+ 호환성: multiprocess RLock 에러 해결
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # 이미 설정된 경우 무시
    pass

import redis
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes.auth import router as auth_router
from app.api.routes.document import router as doc_router
from app.db.database import engine

from app.llm.config import CURRENT_MODEL, LLM_CONFIG
from app.llm.pipeline import invalidate_cache, run_query

# ── 앱 설정 ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=f"로컬 AI({CURRENT_MODEL}) 기반 RAG API",
    description="Ollama + ChromaDB 기반 법률 문서 질의응답",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 인프라 연결 ────────────────────────────────────────────────────────────
try:
    engine = create_engine(
        os.environ.get("DATABASE_URL", "postgresql://postgres:8342@localhost:5432/pdf_db")
    )
except Exception as e:
    print(f"[DB] 연결 실패: {e}")
    engine = None

try:
    redis_client = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
except Exception as e:
    print(f"[Redis] 연결 실패: {e}")
    redis_client = None


executor = ThreadPoolExecutor(max_workers=3)

# include API routers
app.include_router(auth_router)
app.include_router(doc_router)  # /docs 대신 /documents로 변경


# ── 기본 라우트 ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "FastAPI + Postgres + Redis + LLM RAG 정상 작동 중!"}


@app.get("/health")
def health():
    status = {}
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            status["db"] = "ok"
        except Exception as e:
            status["db"] = f"error: {e}"
    if redis_client:
        try:
            redis_client.ping()
            status["redis"] = "ok"
        except Exception as e:
            status["redis"] = f"error: {e}"
    return status


@app.get("/counter")
def counter():
    return {"visits": redis_client.incr("visits")}


# ── PDF 업로드 & 요약 ───────────────────────────────────────────────────────
@app.post("/upload-and-summarize/")
async def upload_and_summarize(
    file: UploadFile = File(...),
    doc_id: int = Form(...),
    user_id: int = Form(...),
    # 법령·조례 / 가이드라인·지침 / 공모·사업 / 감사 / 내부 규정 / 기타
    category: str = Form(default="기타"),
):
    """
    PDF를 업로드하면 카테고리별 형식으로 요약합니다.
    category 파라미터: 법령·조례 / 가이드라인·지침 / 공모·사업 / 감사 / 내부 규정 / 기타
    """
    import os as _os
    import uuid

    from langchain_community.document_loaders import PyPDFLoader

    from app.llm.ingest import ingest_pdf
    from app.llm.summarizer import summarize_document

    pdf_dir = _os.path.join(_os.getcwd(), "pdf_files")
    _os.makedirs(pdf_dir, exist_ok=True)
    tmp = _os.path.join(pdf_dir, f"tmp_{uuid.uuid4().hex}_{file.filename}")

    content = await file.read()
    if not content:
        return {"status": "error", "detail": "빈 파일입니다."}

    with open(tmp, "wb") as f:
        f.write(content)

    try:
        # 1. PDF 텍스트 추출
        docs = PyPDFLoader(tmp).load()
        if not docs:
            return {"status": "error", "detail": "PDF에서 텍스트 추출 실패"}

        # 전체 텍스트 사용 (긴 문서는 summarizer 내부에서 Map-Reduce 처리)
        doc_text = "\n\n".join(d.page_content for d in docs)

        # 2. 카테고리별 요약 (전체 텍스트)
        loop = asyncio.get_event_loop()
        summary_result = await loop.run_in_executor(
            executor,
            lambda: summarize_document(doc_text, category),
        )

        # 3. ChromaDB에 벡터 저장 (백그라운드)
        ingest_result = await loop.run_in_executor(
            executor,
            lambda: ingest_pdf(tmp, doc_id, user_id, doc_name=file.filename, category=category),
        )

        # 4. 새 문서 추가됐으므로 RAG 캐시 무효화
        invalidate_cache(user_id)

        return {
            "status": "success",
            "doc_id": doc_id,
            "file_name": file.filename,
            "category": category,
            "summary": summary_result.get("summary", ""),
            "chunks_saved": ingest_result.get("total_chunks", 0),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        # tmp_ 접두어 파일만 삭제, 원본(tmp_ 없는 파일)은 보관
        if "tmp" in tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


# ── 마크다운 요약 & 저장 ────────────────────────────────────────────────────
@app.post("/summarize-markdown/")
async def summarize_markdown(
    markdown_text: str = Form(...),
    doc_id: int = Form(...),
    user_id: int = Form(...),
    doc_name: str = Form(default="문서"),
    category: str = Form(default="기타"),  # 카테고리 추가
    save_to_db: bool = Form(default=True),
):
    """OCR 결과 마크다운을 받아 카테고리별 형식으로 요약 후 ChromaDB에 저장"""
    from app.llm.ingest import ingest_markdown  # 올바른 경로 (app.LLM → app.llm)
    from app.llm.summarizer import summarize_document

    if not markdown_text.strip():
        return {"status": "error", "detail": "마크다운 내용이 비어있습니다."}

    loop = asyncio.get_event_loop()

    # 1. 카테고리별 요약 (전체 텍스트, Map-Reduce 처리)
    try:
        summary_result = await loop.run_in_executor(
            executor,
            lambda: summarize_document(markdown_text, category),
        )
        summary = summary_result.get("summary", "")
    except Exception as e:
        return {"status": "error", "detail": f"요약 실패: {e}"}

    # 2. ChromaDB 저장 (선택)
    chunks_saved = 0
    if save_to_db:
        try:
            result = await loop.run_in_executor(
                executor,
                lambda: ingest_markdown(
                    markdown_text,
                    doc_id,
                    user_id,
                    doc_name,
                    chunking_method="sections",
                    enable_summary=False,
                    category=category,
                ),
            )
            chunks_saved = result.get("total_chunks", 0)
        except Exception as e:
            print(f"[ingest] 저장 실패: {e}")

    # 새 문서 추가됐으므로 RAG 캐시 무효화
    if chunks_saved > 0:
        invalidate_cache(user_id)

    return {
        "status": "success",
        "doc_id": doc_id,
        "doc_name": doc_name,
        "category": category,
        "summary": summary,
        "chunks_saved": chunks_saved,
    }


# ── MD 파일 업로드 & 요약 & 벡터 저장 ─────────────────────────────────────────
@app.post("/upload-and-summarize-md/")
async def upload_and_summarize_md(
    file: UploadFile = File(...),
    doc_id: int = Form(...),
    user_id: int = Form(...),
    category: str = Form(default="기타"),
    save_to_db: bool = Form(default=True),
):
    """
    마크다운(.md) 파일을 업로드하면 전체 내용을 카테고리별 형식으로 요약하고
    ChromaDB에 저장합니다.

    - 짧은 문서: LLM 직접 요약
    - 긴 문서  : Map-Reduce (청크별 요약 → 최종 통합 요약)
    """
    from app.llm.ingest import ingest_markdown
    from app.llm.summarizer import summarize_document

    # 확장자 확인
    filename = file.filename or ""
    if not filename.lower().endswith(".md"):
        return {"status": "error", "detail": ".md 파일만 업로드 가능합니다."}

    content_bytes = await file.read()
    if not content_bytes:
        return {"status": "error", "detail": "빈 파일입니다."}

    # UTF-8 디코딩 (BOM 처리 포함)
    try:
        markdown_text = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        markdown_text = content_bytes.decode("cp949", errors="replace")

    if not markdown_text.strip():
        return {"status": "error", "detail": "파일 내용이 비어 있습니다."}

    loop = asyncio.get_event_loop()

    # 1. 전체 텍스트 요약 (Map-Reduce 자동 적용)
    try:
        summary_result = await loop.run_in_executor(
            executor,
            lambda: summarize_document(markdown_text, category),
        )
    except Exception as e:
        return {"status": "error", "detail": f"요약 실패: {e}"}

    # 2. ChromaDB 저장 (선택)
    chunks_saved = 0
    if save_to_db:
        try:
            ingest_result = await loop.run_in_executor(
                executor,
                lambda: ingest_markdown(
                    markdown_text,
                    doc_id,
                    user_id,
                    doc_name=filename,
                    chunking_method="sections",
                    enable_summary=False,
                    category=category,
                ),
            )
            chunks_saved = ingest_result.get("total_chunks", 0)

            # 새 문서 추가 후 캐시 무효화
            invalidate_cache(user_id)
        except Exception as e:
            print(f"[ingest_md] 저장 실패: {e}")

    return {
        "status": "success",
        "doc_id": doc_id,
        "file_name": filename,
        "category": category,
        "char_count": len(markdown_text),
        "chunks_used": summary_result.get("chunks_used", 1),
        "summary": summary_result.get("summary", ""),
        "chunks_saved": chunks_saved,
    }


# ── RAG 질의응답 ────────────────────────────────────────────────────────────
@app.post("/query/")
async def query_documents(
    question: str = Form(...),
    user_id: int = Form(default=1),
    cat_id: int = Form(default=0),
):
    if not question.strip():
        return {"status": "error", "detail": "질문을 입력해주세요."}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: run_query(question.strip(), user_id or None, cat_id or None),
    )

    if result.get("status") == "error":
        return {"status": "error", "question": question, "detail": result.get("message")}

    return {
        "status": "success",
        "question": question,
        "answer": result["answer"],
        "references": result["references"],
    }


# ── LLM 헬스체크 ────────────────────────────────────────────────────────────
@app.get("/health-llm/")
def health_llm():
    import requests

    url = LLM_CONFIG.get("base_url", "http://localhost:11434")
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return {"status": "ok", "models": models, "current_model": CURRENT_MODEL}
    except Exception as e:
        return {"status": "error", "message": str(e)}
