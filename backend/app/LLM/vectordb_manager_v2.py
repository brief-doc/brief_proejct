"""
개선된 벡터 데이터베이스 관리 모듈 (v2)
========================================
역할:
- Chroma DB와 PostgreSQL을 통합 관리
- 메타데이터를 PostgreSQL에 저장하고 벡터는 Chroma에 저장
- 사용자/카테고리별 필터링 검색 기능
- 배치 처리 최적화

핵심 기능:
1. ImprovedVectorDBManager 클래스
   - create_and_save_vector_store(): PDF → 벡터 → DB 저장
   - load_existing_vector_store(): 기존 벡터 DB 로드
   - search_with_filter(): 필터링된 검색 수행
   - get_category_stats(): 카테고리별 통계

2. 데이터 구조
   - Chroma DB: 벡터 + 기본 메타데이터 (로컬 SQLite)
   - PostgreSQL: 상세 메타데이터 (사용자, 카테고리, 문서정보)

3. 사용 예
   from LLM.vectordb_manager_v2 import ImprovedVectorDBManager
   
   manager = ImprovedVectorDBManager()
   manager.create_and_save_vector_store(
       chunks=chunks,
       doc_id=1,
       user_id=100,
       file_name="document.pdf",
       cat_id=2,
       content_sum="요약..."
   )
   
   results = manager.search_with_filter(
       query="검색어",
       user_id=100,
       cat_id=2,
       top_k=5
   )

중요: 모든 모델 설정(임베딩, DB 등)은 config.py에서 중앙화 관리됩니다.
      모델을 변경하려면 config.py만 수정하면 됩니다.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import psycopg2
from datetime import datetime
from .config import EMBEDDING_CONFIG, DB_CONFIG, CHROMA_DB_PATH

class ImprovedVectorDBManager:
    """
    PostgreSQL + Chroma 통합 벡터 데이터베이스 관리자
    
    역할:
    - 두 개의 데이터베이스를 동기화하여 관리
    - 벡터 임베딩: ChromaDB (빠른 검색)
    - 메타데이터: PostgreSQL (구조화된 관리)
    
    이점:
    - 벡터는 로컬에서 빠르게 검색
    - 메타데이터는 중앙화된 DB에서 관리
    - 필터링을 통해 정확한 검색 결과 제공
    """
    
    def __init__(self, chroma_path: str = None, db_config: dict = None):
        """
        VectorDBManager 초기화
        
        역할:
        - 벡터 DB(Chroma)와 메타데이터 DB(PostgreSQL) 초기화
        - 임베딩 모델 로드
        - 모든 설정은 config.py에서 중앙화 관리됨
        
        인자:
        - chroma_path (str): Chroma DB 저장 경로 (기본값: config.py의 CHROMA_DB_PATH)
        - db_config (dict): PostgreSQL 연결 설정 (기본값: config.py의 DB_CONFIG)
          {
              "host": "localhost",
              "port": 5432,
              "database": "pdf_db",
              "user": "postgres",
              "password": "password"
          }
        """
        # 기본값은 config.py에서 가져옴
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self.db_config = db_config or DB_CONFIG
        
        # 한국어 최적화 임베딩 모델 초기화
        # - 모든 설정은 config.py의 EMBEDDING_CONFIG에서 관리됨
        # - 모델 변경 시 config.py만 수정하면 됨
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model_name"],
            model_kwargs={'device': EMBEDDING_CONFIG["device"]},
            encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
        )
    
    def create_and_save_vector_store(self, chunks: list, doc_id: int, user_id: int,
                                    file_name: str, cat_id: int, content_sum: str):
        """
        청크를 벡터 임베딩으로 변환하여 DB에 저장
        
        역할:
        - 분해된 텍스트 청크를 처리
        - 각 청크를 벡터 임베딩으로 변환
        - 메타데이터와 함께 Chroma DB에 저장
        - PostgreSQL에도 메타데이터 저장
        
        처리 단계:
        1. 각 청크에 메타데이터 추가 (사용자, 카테고리 등)
        2. 한국어 임베딩 모델로 벡터 변환
        3. 벡터 + 메타데이터를 Chroma DB에 저장
        4. 문서 정보를 PostgreSQL db.doc 테이블에 저장
        5. 작업 이력을 PostgreSQL job 테이블에 기록
        
        인자:
        - chunks (list): LangChain Document 객체 리스트 (각각 page_content와 metadata 보유)
        - doc_id (int): 고유한 문서 ID
        - user_id (int): 문서 소유자 사용자 ID
        - file_name (str): 원본 파일명 (예: "report.pdf")
        - cat_id (int): 카테고리 ID (분류 및 필터링용)
        - content_sum (str): 문서 요약 또는 전체 내용
        
        반환값:
        - True: 성공
        - False: 실패
        """
        try:
            # 1단계: 각 청크에 검색/필터링을 위한 메타데이터 추가
            for chunk in chunks:
                chunk.metadata.update({
                    "doc_id": doc_id,                         # 문서 ID
                    "user_id": user_id,                       # 사용자 ID (필터링)
                    "file_name": file_name,                   # 파일명
                    "cat_id": cat_id,                         # 카테고리 ID (필터링)
                    "content_sum": content_sum,               # 요약 내용
                    "created_at": datetime.now().isoformat()  # 생성 시간
                })
            
            # 2단계: Chroma DB에 벡터 저장
            # - from_documents: 청크를 임베딩으로 변환하여 저장
            # - persist_directory: 로컬 파일에 저장하여 재시작 후에도 유지
            vectorstore = Chroma.from_documents(
                documents=chunks,                   # 텍스트 청크 리스트
                embedding=self.embeddings,         # 임베딩 모델
                persist_directory=self.chroma_path  # 저장 경로
            )
            print(f"✓ 벡터 저장 완료: {len(chunks)} chunks (doc_id={doc_id})")
            
            # 3. PostgreSQL에 메타데이터 저장
            self._save_to_postgresql(doc_id, file_name, "pdf", content_sum, cat_id, user_id)
            
            return True
        except Exception as e:
            print(f"✗ 벡터 저장 실패: {e}")
            return False
    
    def load_existing_vector_store(self):
        """기존 Chroma DB 로드"""
        try:
            return Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"✗ 벡터 DB 로드 실패: {e}")
            return None
    
    def search_with_filter(self, query: str, user_id: int = None, 
                          cat_id: int = None, top_k: int = 5) -> list:
        """
        사용자/카테고리 필터링을 포함한 의미론적 검색
        
        역할:
        - 사용자 입력 쿼리를 벡터로 변환
        - 저장된 벡터들과 유사도 계산
        - 조건에 맞는 상위 K개 반환
        
        검색 프로세스:
        1. 쿼리를 임베딩 벡터로 변환
        2. 모든 청크 벡터와 코사인 유사도 계산
        3. 유사도 높은 순서로 정렬
        4. 메타데이터 필터링 적용 (user_id, cat_id)
        5. 필터링된 상위 K개 반환
        
        필터링 예시:
        - user_id=100, cat_id=None: 사용자 100의 모든 문서
        - user_id=None, cat_id=2: 모든 사용자의 카테고리 2 문서
        - user_id=100, cat_id=2: 사용자 100의 카테고리 2 문서만
        
        인자:
        - query (str): 검색 쿼리 (예: "매출 증가율은?")
        - user_id (int): 사용자 ID 필터 (None이면 모든 사용자)
        - cat_id (int): 카테고리 ID 필터 (None이면 모든 카테고리)
        - top_k (int): 반환할 결과 개수
        
        반환값:
        - list: 필터링된 문서 딕셔너리 리스트
          [
            {
              "content": "청크 텍스트",
              "metadata": {...},
              "doc_id": 1,
              "file_name": "report.pdf",
              "cat_id": 2
            },
            ...
          ]
        """
        vectorstore = self.load_existing_vector_store()
        if not vectorstore:
            return []
        
        # 1단계: 의미론적 유사도로 상위 K*2개 검색 (필터링 후 K개 남도록)
        results = vectorstore.similarity_search(query, k=top_k * 2)
        
        # 2단계: 메타데이터 기반 필터링
        filtered = []
        for doc in results:
            # 사용자 필터: user_id가 지정되고 다르면 제외
            if user_id and doc.metadata.get("user_id") != user_id:
                continue
            # 카테고리 필터: cat_id가 지정되고 다르면 제외
            if cat_id and doc.metadata.get("cat_id") != cat_id:
                continue
            
            # 필터링 통과: 결과에 추가
            filtered.append({
                "content": doc.page_content,        # 청크 텍스트
                "metadata": doc.metadata,           # 모든 메타데이터
                "doc_id": doc.metadata.get("doc_id"),       # 문서 ID
                "file_name": doc.metadata.get("file_name"), # 파일명
                "cat_id": doc.metadata.get("cat_id")        # 카테고리 ID
            })
            
            # 필요한 개수 도달 시 종료
            if len(filtered) >= top_k:
                break
        
        return filtered
    
    def _save_to_postgresql(self, doc_id: int, file_name: str, file_type: str,
                           content_sum: str, cat_id: int, user_id: int):
        """
        문서 메타데이터를 PostgreSQL db.doc 테이블에 저장
        
        역할:
        - 벡터 저장 이후 메타데이터를 중앙 DB에 기록
        - PostgreSQL 연결 실패 시 경고만 출력 (벡터 DB는 유지)
        - 동일 doc_id가 있으면 UPDATE (덮어쓰기)
        
        저장되는 정보:
        - doc_id: 고유 문서 ID
        - file_name: 원본 파일명
        - file_type: 파일 타입 (현재는 "pdf")
        - content_full: 요약 또는 전체 내용
        - cat_id: 카테고리 ID
        - user_id: 소유자 사용자 ID
        
        인자:
        - doc_id (int): 문서 ID
        - file_name (str): 파일명
        - file_type (str): 파일 타입
        - content_sum (str): 요약 내용
        - cat_id (int): 카테고리 ID
        - user_id (int): 사용자 ID
        """
        try:
            # PostgreSQL 연결
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # SQL: 문서 정보 INSERT 또는 UPDATE
            # ON CONFLICT: 동일 doc_id가 있으면 UPDATE
            cur.execute("""
                INSERT INTO doc (doc_id, file_name, file_type, content_full, cat_id, user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    file_type = EXCLUDED.file_type,
                    content_full = EXCLUDED.content_full,
                    cat_id = EXCLUDED.cat_id
            """, (doc_id, file_name, file_type, content_sum, cat_id, user_id))
            
            conn.commit()
            cur.close()
            conn.close()
            print(f"✓ PostgreSQL 저장 완료: doc_id={doc_id}")
        except psycopg2.OperationalError:
            print("[경고] PostgreSQL 연결 불가능. 벡터 DB에만 저장됩니다.")
        except Exception as e:
            print(f"✗ PostgreSQL 저장 실패: {e}")
    
    def get_category_stats(self, user_id: int) -> dict:
        """사용자별 카테고리 통계"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT c.cat_id, c.main, COUNT(d.doc_id) as count
                FROM doc d
                JOIN category c ON d.cat_id = c.cat_id
                WHERE d.user_id = %s
                GROUP BY c.cat_id, c.main
            """, (user_id,))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            return {"categories": results}
        except Exception as e:
            print(f"✗ 통계 조회 실패: {e}")
            return {}
