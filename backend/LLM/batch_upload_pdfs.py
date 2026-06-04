"""
여러 PDF 파일을 벡터 DB에 일괄 저장하는 배치 처리 스크립트
"""

import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime

# config.py에서 설정 가져오기
from LLM.config import EMBEDDING_CONFIG, CHROMA_DB_PATH, TEXT_SPLITTER_CONFIG

# 설정
PDF_FOLDER = "./pdf_files"  # PDF 파일이 있는 폴더

# 임베딩 모델 (config.py의 EMBEDDING_CONFIG 사용)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_CONFIG["model_name"],
    model_kwargs={'device': EMBEDDING_CONFIG["device"]},
    encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
)

# 텍스트 분할기 (config.py의 TEXT_SPLITTER_CONFIG 사용)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
    chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"]
)

def batch_upload_pdfs(pdf_folder: str, start_doc_id: int = 0, user_id: int = 1):
    """
    폴더의 모든 PDF 파일을 벡터 DB에 저장
    
    Args:
        pdf_folder: PDF 파일이 있는 폴더 경로
        start_doc_id: 시작 문서 ID (자동 증가)
        user_id: 사용자 ID
    """
    
    print("=" * 80)
    print("📁 PDF 배치 처리 시작")
    print("=" * 80)
    print(f"📂 대상 폴더: {pdf_folder}")
    print(f"🗄️  벡터 DB: {CHROMA_DB_PATH}")
    print(f"👤 사용자 ID: {user_id}\n")
    
    # 폴더 존재 확인
    if not os.path.exists(pdf_folder):
        print(f"❌ 폴더를 찾을 수 없습니다: {pdf_folder}")
        print(f"💡 먼저 '{pdf_folder}' 폴더를 만들고 PDF 파일을 넣으세요.")
        return
    
    # PDF 파일 목록 조회
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  PDF 파일을 찾을 수 없습니다: {pdf_folder}/*.pdf")
        return
    
    print(f"📄 찾은 PDF 파일 개수: {len(pdf_files)}개\n")
    
    # ChromaDB 초기화
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    
    total_chunks = 0
    successful_files = 0
    failed_files = 0
    
    # 각 PDF 파일 처리
    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_id = start_doc_id + idx - 1
        filename = os.path.basename(pdf_path)
        
        print(f"[{idx}/{len(pdf_files)}] 처리 중: {filename}")
        
        try:
            # 1. PDF 로드
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                print(f"    ⚠️  내용을 읽을 수 없음 (빈 PDF)\n")
                failed_files += 1
                continue
            
            # 2. 청크 분할
            chunks = text_splitter.split_documents(documents)
            
            # 3. 메타데이터 추가
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["user_id"] = user_id
                chunk.metadata["batch_processed_at"] = datetime.now().isoformat()
            
            # 4. 벡터 DB에 저장
            vectorstore.add_documents(chunks)
            
            chunk_count = len(chunks)
            total_chunks += chunk_count
            successful_files += 1
            
            print(f"    ✅ 성공 - {chunk_count}개 청크 저장\n")
            
        except Exception as e:
            print(f"    ❌ 실패 - {str(e)}\n")
            failed_files += 1
    
    # 결과 출력
    print("=" * 80)
    print("📊 배치 처리 완료")
    print("=" * 80)
    print(f"✅ 성공한 파일: {successful_files}개")
    print(f"❌ 실패한 파일: {failed_files}개")
    print(f"📦 총 저장된 청크: {total_chunks}개")
    print(f"🗄️  저장 위치: {CHROMA_DB_PATH}")
    print("=" * 80 + "\n")
    
    if successful_files > 0:
        print("🎉 배치 처리가 완료되었습니다!")
        print(f"💡 다음 문서 ID: {start_doc_id + len(pdf_files)}")


if __name__ == "__main__":
    # 사용 예제
    batch_upload_pdfs(
        pdf_folder=PDF_FOLDER,
        start_doc_id=0,
        user_id=1
    )
