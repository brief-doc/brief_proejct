"""
벡터 DB 저장 및 관리 로직
main.py에서 분리된 벡터 DB 관련 작업을 처리합니다.
"""

import os
from pathlib import Path


def save_pdf_to_vector_db(file_path: str, doc_id: int, user_id: int) -> dict:
    """
    PDF 파일을 벡터화하여 ChromaDB에 저장
    
    Args:
        file_path (str): PDF 파일 경로
        doc_id (int): 문서 ID
        user_id (int): 사용자 ID
    
    Returns:
        dict: 저장 결과 {"status": "success"/"error", "detail": "..."}
    """
    try:
        from .ingest import ingest_pdf
        
        if ingest_pdf is None:
            return {"status": "error", "detail": "ingest 모듈을 찾을 수 없습니다"}
        
        # ingest.py의 ingest_pdf 함수 호출
        file_name = os.path.basename(file_path)
        result = ingest_pdf(
            file_path=file_path,
            doc_id=doc_id,
            user_id=user_id,
            doc_name=file_name
        )
        
        return result
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def save_markdown_to_vector_db(
    markdown_content: str,
    doc_id: int,
    user_id: int,
    doc_name: str = "마크다운 문서",
    chunking_method: str = "sections",
    enable_summary: bool = True
) -> dict:
    """
    마크다운 텍스트를 벡터화하여 ChromaDB에 저장
    
    Args:
        markdown_content (str): 마크다운 텍스트
        doc_id (int): 문서 ID
        user_id (int): 사용자 ID
        doc_name (str): 문서 이름
        chunking_method (str): 청킹 방법 ("sections" 또는 "size")
        enable_summary (bool): 요약 생성 여부
    
    Returns:
        dict: 저장 결과 {"status": "success"/"error", "detail": "..."}
    """
    try:
        from .ingest import ingest_markdown
        
        if ingest_markdown is None:
            return {"status": "error", "detail": "마크다운 처리 모듈을 찾을 수 없습니다"}
        
        # ingest.py의 ingest_markdown 함수 호출
        result = ingest_markdown(
            markdown_content=markdown_content,
            doc_id=doc_id,
            user_id=user_id,
            doc_name=doc_name,
            chunking_method=chunking_method,
            enable_summary=enable_summary
        )
        
        return result
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def batch_save_to_vector_db(
    files_data: list,
    start_doc_id: int,
    user_id: int
) -> dict:
    """
    여러 PDF 파일을 배치로 벡터화하여 저장
    
    Args:
        files_data (list): [{"file_path": str, "file_name": str}, ...] 형태의 파일 정보 리스트
        start_doc_id (int): 시작 document ID
        user_id (int): 사용자 ID
    
    Returns:
        dict: 저장 결과 {"status": "success"/"error", "total_chunks": int, "results": [...]}
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        results = []
        total_chunks = 0
        
        for idx, file_data in enumerate(files_data):
            file_path = file_data.get("file_path")
            file_name = file_data.get("file_name", os.path.basename(file_path))
            doc_id = start_doc_id + idx
            
            try:
                # PDF 로드 및 청크 분할
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(documents)
                
                # 메타데이터 추가
                for i, chunk in enumerate(chunks):
                    chunk.metadata['doc_id'] = doc_id
                    chunk.metadata['user_id'] = user_id
                    chunk.metadata['chunk_index'] = i
                
                total_chunks += len(chunks)
                results.append({
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "chunks": len(chunks),
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "chunks": 0,
                    "status": "error",
                    "detail": str(e)
                })
        
        return {
            "status": "success",
            "total_chunks": total_chunks,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
