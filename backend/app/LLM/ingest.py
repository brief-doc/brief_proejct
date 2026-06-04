"""
적재/벡터화 모듈 (ingest.py)
=================================
역할: 문서를 청크로 쪼개고 → 벡터화 → Chroma에 저장 (한 번만 실행)

특징:
1. 문서를 1,000자 청크로 분할
2. 페이지별 청킹 (페이지 경계 보존)
3. bge-m3로 임베딩
4. 메타데이터: {문서명, 페이지, 청크 ID}
5. ChromaDB에 저장

사용:
    python ingest.py <pdf_파일_경로> <doc_id> <user_id> [--doc-name "문서명"]
    
예시:
    python ingest.py ./pdf_files/계약서.pdf 1 100 --doc-name "2024년 계약서"
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 설정 import
try:
    from .config import CHROMA_DB_PATH, EMBEDDING_CONFIG
except ImportError:
    print("경고: config.py를 찾을 수 없습니다. 기본값을 사용합니다.")
    CHROMA_DB_PATH = "./chroma_pdf_db"
    EMBEDDING_CONFIG = {"model_name": "BAAI/bge-m3", "device": "cpu"}


def is_valid_pdf(file_path: str) -> Tuple[bool, str]:
    """PDF 파일 유효성 검증"""
    if not file_path.lower().endswith('.pdf'):
        return False, "파일 확장자가 .pdf가 아닙니다"
    
    try:
        if not os.path.exists(file_path):
            return False, "파일을 찾을 수 없습니다"
        
        if os.path.getsize(file_path) == 0:
            return False, "파일이 비어있습니다"
        
        loader = PyPDFLoader(file_path)
        test_docs = loader.load()
        
        if not test_docs:
            return False, "PDF에서 텍스트를 추출할 수 없습니다"
        
        return True, "OK"
    except Exception as e:
        return False, f"PDF 로드 실패: {str(e)[:100]}"


def ingest_pdf(
    file_path: str,
    doc_id: int,
    user_id: int,
    doc_name: str = None
) -> Dict:
    """
    PDF 문서를 벡터화하여 ChromaDB에 저장
    
    Args:
        file_path: PDF 파일 경로
        doc_id: 문서 ID
        user_id: 사용자 ID
        doc_name: 커스텀 문서명 (None이면 파일명 사용)
    
    Returns:
        dict: {
            "status": "success" or "error",
            "file_name": str,
            "total_chunks": int,
            "total_pages": int,
            "detail": str
        }
    """
    print(f"\n{'='*60}")
    print(f"[벡터화 시작]")
    print(f"{'='*60}")
    print(f"파일: {file_path}")
    print(f"doc_id: {doc_id}, user_id: {user_id}")
    
    # 1단계: PDF 검증
    print("\n[1단계] PDF 파일 검증 중...")
    is_valid, msg = is_valid_pdf(file_path)
    if not is_valid:
        print(f"❌ 검증 실패: {msg}")
        return {
            "status": "error",
            "detail": f"유효한 PDF 파일이 아닙니다: {msg}"
        }
    print("✓ PDF 검증 완료")
    
    # 파일명 결정
    file_name = doc_name if doc_name else os.path.basename(file_path)
    print(f"  → 문서명: {file_name}")
    
    # 2단계: PDF 로드
    print("\n[2단계] PDF 로드 중...")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        total_pages = len(documents)
        print(f"✓ PDF 로드 완료 (총 {total_pages}페이지)")
    except Exception as e:
        print(f"❌ PDF 로드 실패: {str(e)[:100]}")
        return {
            "status": "error",
            "detail": f"PDF 로드 오류: {str(e)[:100]}"
        }
    
    # 3단계: 페이지별 청킹
    print("\n[3단계] 페이지별 청킹 중...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    all_chunks = []
    for page_idx, doc in enumerate(documents):
        # 각 페이지 내에서만 청킹 (페이지 경계 보존)
        page_chunks = splitter.split_documents([doc])
        
        for chunk_idx, chunk in enumerate(page_chunks):
            # 메타데이터 추가
            chunk.metadata['doc_id'] = doc_id
            chunk.metadata['user_id'] = user_id
            chunk.metadata['file_name'] = file_name
            chunk.metadata['page_num'] = page_idx + 1  # 1부터 시작
            chunk.metadata['chunk_id'] = f"{doc_id}_p{page_idx+1}_c{chunk_idx}"
            
            all_chunks.append(chunk)
    
    total_chunks = len(all_chunks)
    print(f"✓ 청킹 완료 (총 {total_chunks}개 청크)")
    
    # 청크별 통계
    chunks_per_page = {}
    for chunk in all_chunks:
        page = chunk.metadata['page_num']
        chunks_per_page[page] = chunks_per_page.get(page, 0) + 1
    
    print(f"  → 페이지별 청크 분포:")
    for page in sorted(chunks_per_page.keys())[:5]:  # 처음 5개 페이지만 표시
        print(f"     페이지 {page}: {chunks_per_page[page]}개 청크")
    if len(chunks_per_page) > 5:
        print(f"     ... (외 {len(chunks_per_page)-5}페이지)")
    
    # 4단계: 임베딩 모델 로드
    print("\n[4단계] 임베딩 모델 로드 중...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG.get("model_name", "BAAI/bge-m3"),
            model_kwargs={"device": EMBEDDING_CONFIG.get("device", "cpu")},
            encode_kwargs={"normalize_embeddings": True}
        )
        print(f"✓ 모델 로드 완료: {EMBEDDING_CONFIG.get('model_name', 'BAAI/bge-m3')}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {str(e)[:100]}")
        return {
            "status": "error",
            "detail": f"임베딩 모델 로드 실패: {str(e)[:100]}"
        }
    
    # 5단계: ChromaDB에 저장
    print("\n[5단계] ChromaDB에 벡터 저장 중...")
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        
        db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        print(f"✓ 벡터 저장 완료")
        print(f"  → 저장 위치: {CHROMA_DB_PATH}")
    except Exception as e:
        print(f"❌ 벡터 저장 실패: {str(e)[:100]}")
        return {
            "status": "error",
            "detail": f"벡터 저장 실패: {str(e)[:100]}"
        }
    
    # 완료
    print(f"\n{'='*60}")
    print(f"✓ 벡터화 완료!")
    print(f"{'='*60}")
    print(f"문서명: {file_name}")
    print(f"총 페이지: {total_pages}")
    print(f"총 청크: {total_chunks}")
    print(f"평균 청크/페이지: {total_chunks/total_pages:.1f}")
    print(f"{'='*60}\n")
    
    return {
        "status": "success",
        "file_name": file_name,
        "total_chunks": total_chunks,
        "total_pages": total_pages,
        "detail": f"{file_name} 문서가 {total_chunks}개 청크로 벡터화되어 저장되었습니다."
    }


def ingest_markdown(
    markdown_content: str,
    doc_id: int,
    user_id: int,
    doc_name: str = "마크다운 문서",
    chunking_method: str = "sections",
    enable_summary: bool = True
) -> Dict:
    """
    마크다운 문서를 청킹 → 요약 → ChromaDB에 저장
    
    Args:
        markdown_content: 마크다운 텍스트 내용
        doc_id: 문서 ID
        user_id: 사용자 ID
        doc_name: 문서명
        chunking_method: 청킹 방식 ("sections", "size")
        enable_summary: 요약 생성 여부
    
    Returns:
        dict: {
            "status": "success" or "error",
            "total_chunks": int,
            "doc_name": str,
            "detail": str
        }
    """
    try:
        from .markdown_processor import MarkdownProcessor
    except ImportError:
        return {
            "status": "error",
            "detail": "markdown_processor 모듈을 찾을 수 없습니다"
        }
    
    print(f"\n{'='*60}")
    print(f"[마크다운 처리 시작]")
    print(f"{'='*60}")
    print(f"문서명: {doc_name}")
    print(f"크기: {len(markdown_content)} 문자")
    print(f"doc_id: {doc_id}, user_id: {user_id}")
    
    # 마크다운 처리기 초기화
    processor = MarkdownProcessor()
    
    # 마크다운 처리
    try:
        result = processor.process_markdown(
            markdown_content=markdown_content,
            doc_id=doc_id,
            user_id=user_id,
            doc_name=doc_name,
            chunking_method=chunking_method,
            enable_summary=enable_summary,
            save_to_db=True
        )
        
        print(f"\n{'='*60}")
        print(f"✓ 마크다운 처리 완료!")
        print(f"{'='*60}")
        
        return result
    
    except Exception as e:
        print(f"❌ 마크다운 처리 실패: {str(e)[:100]}")
        return {
            "status": "error",
            "detail": f"마크다운 처리 실패: {str(e)[:100]}"
        }


def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(
        description="PDF 문서를 청크로 쪼개고 벡터화하여 ChromaDB에 저장",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python ingest.py ./pdf_files/계약서.pdf 1 100
  python ingest.py ./pdf_files/판결문.pdf 2 200 --doc-name "2024년 판결문"
        """
    )
    
    parser.add_argument(
        "file_path",
        type=str,
        help="PDF 파일 경로"
    )
    parser.add_argument(
        "doc_id",
        type=int,
        help="문서 ID"
    )
    parser.add_argument(
        "user_id",
        type=int,
        help="사용자 ID"
    )
    parser.add_argument(
        "--doc-name",
        type=str,
        default=None,
        help="커스텀 문서명 (기본값: 파일명)"
    )
    
    args = parser.parse_args()
    
    # 벡터화 실행
    result = ingest_pdf(
        file_path=args.file_path,
        doc_id=args.doc_id,
        user_id=args.user_id,
        doc_name=args.doc_name
    )
    
    # 결과 출력
    if result["status"] == "success":
        print(f"✓ 성공: {result['detail']}")
        sys.exit(0)
    else:
        print(f"❌ 실패: {result['detail']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
