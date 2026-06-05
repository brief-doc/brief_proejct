import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile

# LLM 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# LangChain 모듈
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 로컬 모듈 import
config = None
llm_module = None
rag_pipeline = None
ingest = None

try:
    from .config import (
        API_CONFIG,
        CHROMA_DB_PATH,
        CURRENT_MODEL,
        EMBEDDING_CONFIG,
        LLM_CONFIG,
    )

    config = "loaded"
except ImportError as e:
    print(f"[ERROR] config 모듈 import 실패: {e}")
    API_CONFIG = {
        "title": "LLM RAG API",
        "description": "민사법 질의응답 시스템",
        "version": "1.0",
    }
    CHROMA_DB_PATH = "./chroma_pdf_db"
    EMBEDDING_CONFIG = {"model_name": "BAAI/bge-m3", "device": "cpu"}
    CURRENT_MODEL = "gemma3n:e2b"
    LLM_CONFIG = {
        "model_name": CURRENT_MODEL,
        "temperature": 0.7,
        "base_url": "http://localhost:11434",
    }

try:
    from .llm_module import check_ollama_server, get_llm_manager

    llm_module = "loaded"
except ImportError as e:
    print(f"[ERROR] llm_module 모듈 import 실패: {e}")
    get_llm_manager = None
    check_ollama_server = None

try:
    from rag_pipeline_v2 import ImprovedRAGPipeline

    rag_pipeline = "loaded"
except ImportError as e:
    print(f"[ERROR] rag_pipeline_v2 모듈 import 실패: {e}")
    print(f"[DEBUG] 이유: {e.__class__.__name__}: {str(e)}")
    ImprovedRAGPipeline = None

try:
    from app.LLM.ingest import ingest_markdown, ingest_pdf  # 벡터화 모듈

    ingest = "loaded"
except ImportError as e:
    print(f"[ERROR] ingest 모듈 import 실패: {e}")
    ingest_pdf = None
    ingest_markdown = None

app = FastAPI(
    title=API_CONFIG.get("title", "LLM RAG API"),
    description=API_CONFIG.get("description", "민사법 질의응답 시스템"),
    version=API_CONFIG.get("version", "1.0"),
)

# 비동기 처리를 위한 ThreadPoolExecutor (최대 3개 동시 처리)
executor = ThreadPoolExecutor(max_workers=3)
# Note: Model config from CURRENT_MODEL in config.py (v1.1)

# ===================== Helper Functions =====================


def is_valid_pdf(file_path: str) -> tuple:
    """PDF 파일 유효성 검증 - PyPDFLoader로 실제 읽기 가능 여부 확인"""
    if not file_path.lower().endswith(".pdf"):
        return False, "파일 확장자가 .pdf가 아닙니다"

    try:
        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            return False, "파일을 찾을 수 없습니다"

        # 파일 크기 확인 (0보다 커야 함)
        if os.path.getsize(file_path) == 0:
            return False, "파일이 비어있습니다"

        # 실제 PyPDFLoader로 로드 가능 여부 확인
        loader = PyPDFLoader(file_path)
        test_docs = loader.load()

        if not test_docs:
            return False, "PDF에서 텍스트를 추출할 수 없습니다"

        return True, "OK"
    except Exception as e:
        return False, f"PDF 로드 실패: {str(e)[:100]}"


def save_to_vector_db(file_path: str, doc_id: int, user_id: int):
    """벡터화 (ingest.py의 ingest_pdf 호출)"""
    try:
        if ingest_pdf is None:
            return {"status": "error", "detail": "ingest 모듈을 찾을 수 없습니다"}

        # ingest.py의 ingest_pdf 함수 호출
        file_name = os.path.basename(file_path)
        result = ingest_pdf(
            file_path=file_path, doc_id=doc_id, user_id=user_id, doc_name=file_name
        )

        return result
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ===================== PDF Upload & Summarize =====================


@app.post("/upload-and-summarize/")
async def upload_and_summarize(
    file: UploadFile = File(...),
    doc_id: int = Form(...),
    user_id: int = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """PDF 파일을 업로드하고 요약"""
    try:
        # 임시 파일 저장 (절대 경로 사용)
        import uuid

        pdf_dir = os.path.join(os.getcwd(), "pdf_files")
        os.makedirs(pdf_dir, exist_ok=True)

        # 고유한 파일명 생성
        unique_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
        temp_path = os.path.join(pdf_dir, unique_filename)

        # 파일 내용 읽기
        file_content = await file.read()

        # 파일이 비어있는지 확인
        if not file_content:
            return {"status": "error", "detail": "업로드된 파일이 비어있습니다."}

        # 파일 저장
        with open(temp_path, "wb") as buffer:
            buffer.write(file_content)

        # PDF 유효성 검증
        is_valid, msg = is_valid_pdf(temp_path)
        if not is_valid:
            try:
                os.remove(temp_path)
            except:
                pass
            return {"status": "error", "detail": f"유효한 PDF 파일이 아닙니다: {msg}"}

        # PDF 로드 및 요약
        from langchain_ollama import ChatOllama

        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            if not documents:
                return {
                    "status": "error",
                    "detail": "PDF에서 텍스트를 추출할 수 없습니다.",
                }
        except Exception as load_error:
            return {
                "status": "error",
                "detail": f"PDF 로드 오류: {str(load_error)[:100]}",
            }

        llm = ChatOllama(
            model=LLM_CONFIG.get("model_name", CURRENT_MODEL),
            temperature=LLM_CONFIG.get("temperature", 0.7),
            base_url=LLM_CONFIG.get("base_url", "http://localhost:11434"),
        )

        # 간단한 요약 로직 (load_summarize_chain 대체)
        try:
            # 처음 5개 페이지의 내용을 합침 (길이 제한을 위해)
            texts = [doc.page_content for doc in documents[:5]]
            combined_text = "\n\n".join(texts)[:2000]  # 2000자로 제한

            # LLM에 요약 요청
            summary_prompt = f"""다음 PDF 내용을 한국어로 간결하게 요약해주세요:

{combined_text}

요약:"""

            summary_response = llm.invoke(summary_prompt)
            summary_text = (
                summary_response.content
                if hasattr(summary_response, "content")
                else str(summary_response)
            )
            summary = {"output_text": summary_text}
        except Exception as summary_error:
            summary = {"output_text": f"요약 생성 중 오류: {str(summary_error)[:100]}"}

        # 백그라운드에서 벡터 DB 저장
        background_tasks.add_task(save_to_vector_db, temp_path, doc_id, user_id)

        return {
            "status": "success",
            "model_used": {
                CURRENT_MODEL
            },  # config.py에서 설정한 현재 모델명 (동적 반영)
            "doc_id": doc_id,
            "summary": summary.get("output_text", "요약을 생성할 수 없습니다."),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ===================== Markdown Upload & Process =====================


@app.post("/upload-markdown/")
async def upload_markdown(
    markdown_text: str = Form(...),
    doc_id: int = Form(...),
    user_id: int = Form(...),
    doc_name: str = Form(default="마크다운 문서"),
    chunking_method: str = Form(default="sections"),
    enable_summary: bool = Form(default=True),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """마크다운 텍스트를 청킹하고 요약하여 저장"""
    try:
        # 입력 검증
        if not markdown_text or not markdown_text.strip():
            return {"status": "error", "detail": "마크다운 텍스트가 비어있습니다"}

        if len(markdown_text) > 1000000:  # 1MB 제한
            return {"status": "error", "detail": "텍스트가 너무 깁니다 (최대 1MB)"}

        # ingest_markdown 모듈 로드 확인
        if ingest_markdown is None:
            return {
                "status": "error",
                "detail": "마크다운 처리 모듈을 찾을 수 없습니다",
            }

        # 마크다운 처리 (백그라운드에서 실행하지만 결과는 기다림)
        result = ingest_markdown(
            markdown_content=markdown_text,
            doc_id=doc_id,
            user_id=user_id,
            doc_name=doc_name,
            chunking_method=chunking_method,
            enable_summary=enable_summary,
        )

        return {
            "status": result.get("status", "success"),
            "doc_id": doc_id,
            "doc_name": doc_name,
            "total_chunks": result.get("total_chunks", 0),
            "detail": result.get("detail", "마크다운이 처리되었습니다"),
            "chunking_method": chunking_method,
            "summary_enabled": enable_summary,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ===================== Batch Upload =====================


@app.post("/batch-upload/")
async def batch_upload(
    files: list = File(...), start_doc_id: int = Form(...), user_id: int = Form(...)
):
    """여러 PDF 파일을 배치로 업로드"""
    try:
        import uuid

        pdf_dir = os.path.join(os.getcwd(), "pdf_files")
        os.makedirs(pdf_dir, exist_ok=True)

        results = []
        total_chunks = 0

        for idx, file in enumerate(files):
            doc_id = start_doc_id + idx

            # 파일 저장 (고유 파일명)
            unique_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(pdf_dir, unique_filename)

            file_content = await file.read()
            if not file_content:
                continue

            with open(file_path, "wb") as buffer:
                buffer.write(file_content)

            is_valid, msg = is_valid_pdf(file_path)
            if not is_valid:
                try:
                    os.remove(file_path)
                except:
                    pass
                continue

            # PDF 로드 및 청크 분할
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(documents)

            # 메타데이터 추가
            for i, chunk in enumerate(chunks):
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["user_id"] = user_id
                chunk.metadata["chunk_index"] = i

            total_chunks += len(chunks)
            results.append(
                {"file_name": file.filename, "doc_id": doc_id, "chunks": len(chunks)}
            )

        return {"status": "success", "total_chunks": total_chunks, "results": results}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/summarize-markdown/")
async def summarize_markdown(
    markdown_text: str = Form(...), chunking_method: str = Form(default="sections")
):
    """마크다운 텍스트를 청킹하고 각 청크를 요약

    Note: ChromaDB에 저장하지 않고 요약만 반환
    """
    try:
        if not markdown_text or not markdown_text.strip():
            return {"status": "error", "detail": "마크다운 텍스트가 비어있습니다"}

        try:
            from .markdown_processor import MarkdownProcessor
        except ImportError:
            return {
                "status": "error",
                "detail": "마크다운 처리 모듈을 찾을 수 없습니다",
            }

        # 마크다운 처리기 초기화
        processor = MarkdownProcessor()

        # 청킹 수행
        if chunking_method == "sections":
            chunks = processor.chunk_by_sections(markdown_text)
        else:
            chunks = processor.chunk_by_size(markdown_text)

        # 각 청크 요약 생성
        results = []
        for i, chunk in enumerate(chunks, 1):
            summary = processor.summarize_chunk(chunk.get("content", ""))
            results.append(
                {
                    "chunk_id": i,
                    "section": chunk.get("section", chunk.get("id", "N/A")),
                    "original": chunk.get("content", "")[:200] + "...",
                    "summary": summary,
                    "char_count": chunk.get("char_count", 0),
                }
            )

        return {
            "status": "success",
            "total_chunks": len(results),
            "chunking_method": chunking_method,
            "chunks": results,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ===================== Query (RAG) =====================


@app.post("/query/")
async def query_documents(
    question: str = Form(...),
    user_id: int = Form(default=1),
    cat_id: int = Form(default=0),
):
    """RAG 기반 질의응답 (비동기 처리)"""
    try:
        if not question or not question.strip():
            return {"status": "error", "detail": "질문을 입력해주세요."}

        # RAG 파이프라인 모듈 로드 확인
        if ImprovedRAGPipeline is None:
            return {
                "status": "error",
                "detail": "RAG 파이프라인 모듈을 로드할 수 없습니다. 서버 로그를 확인하세요.",
                "module_status": {
                    "config": config,
                    "llm_module": llm_module,
                    "rag_pipeline": rag_pipeline,
                    "ingest": ingest,
                },
            }

        # RAG 파이프라인 초기화
        pipeline = ImprovedRAGPipeline()

        # 비동기로 RAG 처리 실행 (ThreadPoolExecutor 사용)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: pipeline.query_with_context(
                query=question.strip(), user_id=user_id, cat_id=cat_id, top_k=3
            ),
        )

        # 쿼리 히스토리 로깅 (비동기 백그라운드 처리)
        loop.run_in_executor(
            executor,
            lambda: pipeline.log_query_history(
                user_id=user_id,
                query=question,
                answer=result.get("answer", ""),
                references=result.get("references", []),
            ),
        )

        # RAG 파이프라인 결과 확인 (에러 여부 확인)
        if result.get("status") == "error":
            return {
                "status": "error",
                "question": question,
                "detail": result.get(
                    "message", result.get("detail", "알 수 없는 에러")
                ),
                "filters": {"user_id": user_id, "cat_id": cat_id},
            }

        return {
            "status": "success",
            "question": question,
            "answer": result.get("answer", "답변을 생성할 수 없습니다."),
            "references": result.get("references", []),
            "filters": {"user_id": user_id, "cat_id": cat_id},
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ===================== Health Check =====================


@app.get("/health/")
async def health():
    """서버 상태 확인"""
    try:
        ollama_status = check_ollama_server()
        return {
            "status": "healthy" if ollama_status["available"] else "degraded",
            "message": "Server is running",
            "models": ollama_status.get("models", []),
            "current_model": CURRENT_MODEL,
            "server_url": LLM_CONFIG.get("base_url", "http://localhost:11434"),
        }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}
