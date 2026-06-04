"""
마크다운 처리 모듈
- 마크다운 텍스트 청킹
- LLM 요약
- ChromaDB에 저장
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from config import CHROMA_DB_PATH, EMBEDDING_CONFIG
except ImportError:
    CHROMA_DB_PATH = "./chroma_pdf_db"
    EMBEDDING_CONFIG = {"model_name": "BAAI/bge-m3", "device": "cpu"}

# 기본 설정
DEFAULT_LLM_MODEL = "llama3.2:1b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class MarkdownProcessor:
    """마크다운 문서 처리 클래스"""
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        llm_model: str = DEFAULT_LLM_MODEL
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        
        # 텍스트 분할기 초기화
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", ""]
        )
        
        # ChromaDB 클라이언트 초기화
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = self.chroma_client.get_collection(name="qa_collection")
        except Exception as e:
            print(f"⚠️  ChromaDB 연결 실패: {e}")
            self.chroma_client = None
            self.collection = None
    
    def chunk_by_sections(self, content: str) -> List[Dict]:
        """섹션별 청킹 (마크다운 헤더 기준)
        
        Args:
            content: 마크다운 내용
        
        Returns:
            청크 리스트
        """
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_section = "본문"
        section_level = 0
        chunk_id = 0
        
        for line in lines:
            # 마크다운 헤더 감지
            if line.startswith('#'):
                # 이전 청크 저장
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text and len(chunk_text) > 10:  # 최소 길이 체크
                        chunks.append({
                            "id": chunk_id,
                            "section": current_section,
                            "level": section_level,
                            "content": chunk_text,
                            "char_count": len(chunk_text)
                        })
                        chunk_id += 1
                
                # 새로운 섹션 시작
                level_count = len(line) - len(line.lstrip('#'))
                current_section = line.lstrip('#').strip()
                section_level = level_count
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # 마지막 청크 저장
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text and len(chunk_text) > 10:
                chunks.append({
                    "id": chunk_id,
                    "section": current_section,
                    "level": section_level,
                    "content": chunk_text,
                    "char_count": len(chunk_text)
                })
        
        return chunks
    
    def chunk_by_size(self, content: str) -> List[Dict]:
        """크기별 청킹 (RecursiveCharacterTextSplitter 사용)
        
        Args:
            content: 마크다운 내용
        
        Returns:
            청크 리스트
        """
        text_chunks = self.splitter.split_text(content)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip() and len(chunk_text) > 10:
                chunks.append({
                    "id": i,
                    "content": chunk_text.strip(),
                    "char_count": len(chunk_text)
                })
        
        return chunks
    
    def summarize_chunk(self, chunk_content: str, max_length: int = 200) -> str:
        """LLM을 사용한 청크 요약
        
        Args:
            chunk_content: 청크 내용
            max_length: 최대 요약 길이 (단어)
        
        Returns:
            요약 텍스트
        """
        # 내용이 너무 길면 자르기
        content_preview = chunk_content[:800]
        
        prompt = f"""다음 텍스트를 {max_length}자 이내로 요약해주세요.
핵심만 간결하게 정리해주세요.

텍스트:
{content_preview}

요약:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "num_predict": 250
                },
                timeout=180
            )
            
            if response.status_code == 200:
                summary = response.json().get("response", "").strip()
                # 요약이 너무 길면 자르기
                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."
                return summary
            else:
                return f"[요약 실패: HTTP {response.status_code}]"
        
        except requests.exceptions.Timeout:
            return "[요약 실패: 타임아웃]"
        except Exception as e:
            return f"[요약 실패: {str(e)[:50]}]"
    
    def process_chunks_with_summary(
        self,
        chunks: List[Dict],
        enable_summary: bool = True,
        summary_length: int = 200
    ) -> List[Dict]:
        """청크에 요약 추가
        
        Args:
            chunks: 청크 리스트
            enable_summary: 요약 생성 여부
            summary_length: 요약 길이
        
        Returns:
            요약이 추가된 청크 리스트
        """
        print(f"\n🤖 {len(chunks)}개 청크 처리 중...")
        
        results = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}/{len(chunks)}] 처리 중... ({chunk.get('char_count', 0)} 문자)")
            
            chunk_copy = chunk.copy()
            
            # 요약 생성
            if enable_summary:
                chunk_copy['summary'] = self.summarize_chunk(
                    chunk.get('content', ''),
                    summary_length
                )
                print(f"           ✓ 요약: {chunk_copy['summary'][:80]}...")
            
            results.append(chunk_copy)
        
        return results
    
    def save_to_chromadb(
        self,
        chunks: List[Dict],
        doc_id: int,
        user_id: int,
        doc_name: str,
        doc_type: str = "markdown"
    ) -> Dict:
        """청크를 ChromaDB에 저장
        
        Args:
            chunks: 청크 리스트
            doc_id: 문서 ID
            user_id: 사용자 ID
            doc_name: 문서명
            doc_type: 문서 타입 ("markdown", "qa_data" 등)
        
        Returns:
            저장 결과
        """
        if not self.collection:
            return {"status": "error", "detail": "ChromaDB 연결 실패"}
        
        try:
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_md_{i}"
                
                ids.append(chunk_id)
                documents.append(chunk.get('content', ''))
                metadatas.append({
                    'doc_id': str(doc_id),
                    'user_id': str(user_id),
                    'doc_name': doc_name,
                    'doc_type': doc_type,
                    'chunk_id': i,
                    'chunk_summary': chunk.get('summary', ''),
                    'section': chunk.get('section', 'N/A'),
                    'source': 'markdown'
                })
            
            # ChromaDB에 저장
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"✓ {len(ids)}개 청크 저장 완료 (doc_id: {doc_id})")
            
            return {
                "status": "success",
                "total_chunks": len(ids),
                "doc_id": doc_id,
                "detail": f"{len(ids)}개 청크가 저장되었습니다"
            }
        
        except Exception as e:
            print(f"✗ ChromaDB 저장 실패: {e}")
            return {
                "status": "error",
                "detail": f"저장 실패: {str(e)[:100]}"
            }
    
    def process_markdown(
        self,
        markdown_content: str,
        doc_id: int,
        user_id: int,
        doc_name: str,
        chunking_method: str = "sections",
        enable_summary: bool = True,
        save_to_db: bool = True
    ) -> Dict:
        """마크다운 문서 처리 (청킹 → 요약 → 저장)
        
        Args:
            markdown_content: 마크다운 텍스트
            doc_id: 문서 ID
            user_id: 사용자 ID
            doc_name: 문서명
            chunking_method: 청킹 방식 ("sections", "size")
            enable_summary: 요약 생성 여부
            save_to_db: ChromaDB 저장 여부
        
        Returns:
            처리 결과
        """
        print(f"\n{'='*60}")
        print(f"마크다운 처리 시작")
        print(f"{'='*60}")
        print(f"문서: {doc_name}")
        print(f"크기: {len(markdown_content)} 문자")
        print(f"doc_id: {doc_id}, user_id: {user_id}")
        
        # 1단계: 청킹
        print(f"\n[1단계] 청킹 중...")
        if chunking_method == "sections":
            chunks = self.chunk_by_sections(markdown_content)
            print(f"✓ 섹션별 청킹 완료: {len(chunks)}개")
        else:
            chunks = self.chunk_by_size(markdown_content)
            print(f"✓ 크기별 청킹 완료: {len(chunks)}개")
        
        if not chunks:
            return {
                "status": "error",
                "detail": "청킹 결과가 없습니다"
            }
        
        # 2단계: 요약
        print(f"\n[2단계] 요약 생성 중...")
        chunks_with_summary = self.process_chunks_with_summary(
            chunks,
            enable_summary=enable_summary
        )
        
        # 3단계: ChromaDB 저장
        if save_to_db:
            print(f"\n[3단계] ChromaDB에 저장 중...")
            save_result = self.save_to_chromadb(
                chunks_with_summary,
                doc_id=doc_id,
                user_id=user_id,
                doc_name=doc_name,
                doc_type="markdown"
            )
            
            return save_result
        
        # DB 저장 없이 반환
        return {
            "status": "success",
            "total_chunks": len(chunks_with_summary),
            "chunks": chunks_with_summary,
            "detail": "마크다운 처리 완료 (DB 저장 안 함)"
        }


def process_markdown_text(
    markdown_text: str,
    doc_id: int,
    user_id: int,
    doc_name: str,
    chunking_method: str = "sections",
    enable_summary: bool = True
) -> Dict:
    """마크다운 텍스트 처리 (편의 함수)
    
    Args:
        markdown_text: 마크다운 텍스트
        doc_id: 문서 ID
        user_id: 사용자 ID
        doc_name: 문서명
        chunking_method: 청킹 방식
        enable_summary: 요약 생성 여부
    
    Returns:
        처리 결과
    """
    processor = MarkdownProcessor()
    return processor.process_markdown(
        markdown_content=markdown_text,
        doc_id=doc_id,
        user_id=user_id,
        doc_name=doc_name,
        chunking_method=chunking_method,
        enable_summary=enable_summary,
        save_to_db=True
    )
