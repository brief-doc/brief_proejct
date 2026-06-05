import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks

# 로컬 모델용 LangChain 모듈
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.summarize import load_summarize_chain

# 설정 및 LLM 모듈 임포트 (절대 경로로 변경)
try:
    # LLM 폴더에서 직접 실행할 때
    from config import API_CONFIG, CHROMA_DB_PATH, EMBEDDING_CONFIG
    from llm_module import get_llm_manager
except ImportError:
    # 상위 폴더에서 실행할 때
    from LLM.config import API_CONFIG, CHROMA_DB_PATH, EMBEDDING_CONFIG
    from LLM.llm_module import get_llm_manager 

# FastAPI 앱 초기화
# - config.py의 API_CONFIG에서 설정을 자동으로 가져옴
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"]
)

def is_valid_pdf(file_path: str) -> tuple[bool, str]:
    """
    PDF 파일 유효성 검증
    - 파일 확장자 확인
    - 파일 헤더(매직 넘버) 확인
    """
    # 1. 확장자 확인
    if not file_path.lower().endswith('.pdf'):
        return False, f"잘못된 파일 확장자입니다. PDF 파일만 업로드 가능합니다. (업로드된 파일: {os.path.splitext(file_path)[1]})"
    
    # 2. 파일 헤더 확인 (PDF는 %PDF로 시작)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                # ZIP 파일인 경우 (XLSX, DOCX, ZIP 등)
                if header.startswith(b'PK\x03\x04'):
                    return False, "ZIP 기반 파일(XLSX, DOCX, ZIP 등)이 감지되었습니다. 순수 PDF 파일을 업로드해주세요."
                else:
                    return False, f"올바른 PDF 파일이 아닙니다. 파일 헤더: {header.hex()}"
    except Exception as e:
        return False, f"파일 검증 실패: {str(e)}"
    
    return True, "OK"

def save_to_vector_db(file_path: str, doc_id: int, user_id: int):
    try:
        # 1. PDF 로드 및 청크 분할
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["user_id"] = user_id

        # 2. 로컬 한국어 임베딩 모델 로드 (config.py의 EMBEDDING_CONFIG에서 가져옴)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model_name"],
            model_kwargs={'device': EMBEDDING_CONFIG["device"]},
            encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
        )

        # 3. Chroma DB에 저장
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print(f"[완료] doc_id: {doc_id} 벡터 DB 저장 성공")
    except Exception as e:
        print(f"[에러] 벡터 DB 저장 실패: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload-and-summarize/", summary="PDF 업로드 및 요약")
async def upload_and_summarize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: int = Form(...),
    user_id: int = Form(...)
):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. PDF 파일 유효성 검증
    is_valid, validation_msg = is_valid_pdf(temp_file_path)
    if not is_valid:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {
            "status": "error",
            "error": validation_msg
        }

    # 2. PDF 로드 시도 (손상된 파일 안전 처리)
    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        if not docs:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return {
                "status": "error",
                "error": "PDF 파일에서 내용을 읽을 수 없습니다. 파일이 손상되었을 수 있습니다."
            }
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {
            "status": "error",
            "error": f"PDF 파일 처리 실패: {str(e)}"
        }

    # --- LLM 모델 연결 ---
    # llm_module에서 LLMManager의 LLM 인스턴스를 가져옴
    llm_manager = get_llm_manager()
    llm = llm_manager.get_llm_instance()
    
    chain = load_summarize_chain(llm, chain_type="stuff")
    
    try:
        # 요약 실행 (로컬 PC 성능에 따라 시간이 조금 걸릴 수 있습니다)
        summary_result = chain.invoke(docs)
        summary_text = summary_result["output_text"]
    except Exception as e:
        summary_text = f"요약 중 오류 발생: {str(e)}"

    background_tasks.add_task(save_to_vector_db, temp_file_path, doc_id, user_id)

    return {
        "status": "success",
        "model_used": llm_manager.get_config_summary(),
        "doc_id": doc_id,
        "summary": summary_text
    }

@app.post("/batch-upload/", summary="여러 PDF 파일 배치 업로드")
async def batch_upload(
    background_tasks: BackgroundTasks,
    files: list = File(...),
    start_doc_id: int = Form(default=0),
    user_id: int = Form(default=1)
):
    """
    여러 개의 PDF 파일을 한꺼번에 벡터 DB에 저장합니다.
    
    Args:
        files: 업로드할 PDF 파일 목록
        start_doc_id: 시작 문서 ID
        user_id: 사용자 ID
    """
    
    if not files:
        return {
            "status": "error",
            "message": "파일이 업로드되지 않았습니다."
        }
    
    results = []
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model_name"],
            model_kwargs={'device': EMBEDDING_CONFIG["device"]},
            encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
        )
    )
    
    total_chunks = 0
    
    for idx, file in enumerate(files):
        doc_id = start_doc_id + idx
        temp_file_path = f"temp_{file.filename}"
        
        try:
            # 파일 저장
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # PDF 유효성 검증
            is_valid, validation_msg = is_valid_pdf(temp_file_path)
            if not is_valid:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": validation_msg
                })
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                continue
            
            # PDF 로드 및 청크 분할
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            if not documents:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "PDF 파일에서 내용을 읽을 수 없습니다."
                })
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                continue
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # 메타데이터 추가
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["user_id"] = user_id
            
            # 벡터 DB에 저장 (config.py의 EMBEDDING_CONFIG 사용)
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_CONFIG["model_name"],
                model_kwargs={'device': EMBEDDING_CONFIG["device"]},
                encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
            )
            vectorstore.add_documents(chunks)
            
            total_chunks += len(chunks)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "doc_id": doc_id,
                "chunks": len(chunks)
            })
            
            # 임시 파일 삭제
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    return {
        "status": "success",
        "total_files": len(files),
        "total_chunks": total_chunks,
        "results": results
    }