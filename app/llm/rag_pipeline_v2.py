"""
개선된 RAG (Retrieval-Augmented Generation) 파이프라인 모듈 (v2)
=================================================================
역할:
- 필터링된 문서 검색 (ChromaDB)
- 검색된 문서를 기반으로 LLM이 답변 생성
- PostgreSQL 메타데이터 활용
- 사용자/카테고리별 개인화 검색

핵심 기능:
1. ImprovedRAGPipeline 클래스
   - query_with_context(): 필터링 검색 + LLM 답변 생성
   - create_retriever(): 필터링 가능한 리트리버 생성
   - get_document_summary(): 문서 요약 조회
   - log_query_history(): 쿼리 이력 저장
   - get_user_statistics(): 사용자 통계

2. RAG 처리 단계
   1. 사용자 질문 받음
   2. 필터링 조건 적용 (사용자, 카테고리)
   3. ChromaDB에서 관련 문서 검색
   4. 검색된 문서를 컨텍스트로 구성
   5. LLM이 답변 생성 (llm_module.py에서 처리)
   6. 답변 + 참고 문서 반환

3. 이점
   - 정확한 답변 (문서 기반)
   - 환각(hallucination) 감소
   - 개인화된 결과 (사용자/카테고리 필터)
   - 답변의 출처 명확 (참고 문서 제시)

4. 사용 예
   from LLM.rag_pipeline_v2 import ImprovedRAGPipeline
   
   pipeline = ImprovedRAGPipeline()
   result = pipeline.query_with_context(
       query="2026년 매출은 얼마인가?",
       user_id=100,
       cat_id=2
   )
   # result = {
   #     "status": "success",
   #     "query": "...",
   #     "answer": "LLM이 생성한 답변",
   #     "references": [...]
   # }

담당 영역:
- [임베딩 팀] 임베딩 모델, ChromaDB, 문서 검색
- [LLM 팀] LLM 초기화 및 답변 생성 (llm_module.py)
- [공용] 설정 (config.py)

중요: 모든 모델 설정(LLM, 임베딩, DB 등)은 config.py에서 중앙화 관리됩니다.
      모델을 변경하려면 config.py만 수정하면 됩니다.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import psycopg2
from datetime import datetime
from config import EMBEDDING_CONFIG, DB_CONFIG, CHROMA_DB_PATH
from llm_module import get_llm_manager

class ImprovedRAGPipeline:
    """
    필터링 가능한 RAG 파이프라인
    
    역할:
    - 사용자 질문에 대해 관련 문서를 검색
    - 검색된 문서를 기반으로 LLM이 답변 생성
    - 결과의 신뢰성과 정확성 보장
    
    특징:
    - 사용자/카테고리별 개인화 검색
    - PostgreSQL 메타데이터 통합
    - 쿼리 이력 관리
    - 사용자 통계 제공
    """
    
    def __init__(self, chroma_path: str = None, db_config: dict = None):
        """
        RAG 파이프라인 초기화
        
        역할:
        - 벡터 DB, 임베딩 모델, DB 연결 초기화
        - LLM은 llm_module.py에서 관리됨
        - 모든 설정은 config.py에서 중앙화 관리됨
        - 모델을 변경하려면 config.py만 수정하면 됨
        
        인자:
        - chroma_path (str): Chroma DB 저장 경로 (기본값: config.py의 CHROMA_DB_PATH)
        - db_config (dict): PostgreSQL 연결 설정 (기본값: config.py의 DB_CONFIG)
        """
        # 기본값은 config.py에서 가져옴
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self.db_config = db_config or DB_CONFIG
        
        # 한국어 임베딩 모델 로드
        # - 모든 설정은 config.py의 EMBEDDING_CONFIG에서 관리됨
        # - 모델 변경 시 config.py만 수정하면 됨
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model_name"],
            model_kwargs={'device': EMBEDDING_CONFIG["device"]},
            encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
        )
        
        # LLM 매니저 로드 (llm_module.py에서 관리)
        self.llm_manager = get_llm_manager()
    
    def create_retriever(self, user_id: int = None, cat_id: int = None):
        """
        필터링 가능한 리트리버 생성
        
        역할:
        - Chroma DB에서 문서를 검색할 수 있는 리트리버 객체 생성
        - 메타데이터 필터를 설정하여 사용자/카테고리별 필터링 가능
        
        인자:
        - user_id (int): 사용자 ID (필터링, None이면 생략)
        - cat_id (int): 카테고리 ID (필터링, None이면 생략)
        
        반환값:
        - retriever 객체 (검색 가능)
        - None: 실패
        """
        try:
            vectorstore = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings
            )
            
            # 리트리버 설정
            search_kwargs = {"k": 3}  # 기본 상위 3개
            
            # 메타데이터 필터 설정 (Chroma의 필터 형식)
            if user_id or cat_id:
                filter_dict = {}
                if user_id:
                    filter_dict["user_id"] = {"$eq": user_id}  # 사용자 ID 필터
                if cat_id:
                    filter_dict["cat_id"] = {"$eq": cat_id}    # 카테고리 ID 필터
                search_kwargs["filter"] = filter_dict
            
            return vectorstore.as_retriever(**search_kwargs)
        except Exception as e:
            print(f"✗ 리트리버 생성 실패: {e}")
            return None
    
    def query_with_context(self, query: str, user_id: int = None, 
                          cat_id: int = None, top_k: int = 3) -> dict:
        """
        필터링된 컨텍스트를 포함한 RAG 질의응답
        
        역할:
        - 사용자 질문과 필터링 조건을 받음
        - 관련 문서를 검색
        - LLM이 문서 기반으로 답변 생성
        - 답변과 참고 문서 반환
        
        RAG 처리 단계:
        1. 질문과 필터링 조건으로 문서 검색
        2. 검색된 문서를 프롬프트에 포함
        3. LLM에 문서와 질문 전달
        4. LLM이 문서 기반 답변 생성
        5. 답변과 참고 문서 정보 반환
        
        인자:
        - query (str): 사용자 질문 (예: "2026년 1분기 매출은?")
        - user_id (int): 사용자 ID (필터링, None이면 전체)
        - cat_id (int): 카테고리 ID (필터링, None이면 전체)
        - top_k (int): 참고할 문서 개수
        
        반환값:
        - dict: {
            "status": "success" or "error",
            "query": str,
            "answer": str (생성된 답변),
            "references": list (참고 문서),
            "filters": dict (적용된 필터)
          }
        """
        try:
            # 1단계: 문서 검색 (필터링 포함)
            vectorstore = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings
            )
            
            docs = vectorstore.similarity_search(query, k=top_k * 2)
            
            # 필터링
            filtered_docs = []
            for doc in docs:
                if user_id and doc.metadata.get("user_id") != user_id:
                    continue
                if cat_id and doc.metadata.get("cat_id") != cat_id:
                    continue
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
            
            if not filtered_docs:
                return {
                    "status": "error",
                    "message": "검색 결과 없음",
                    "answer": None,
                    "references": []
                }
            
            # 2단계: 컨텍스트 텍스트 구성
            # 검색된 각 문서를 명시적으로 표시하여 LLM이 출처를 알 수 있도록
            context_text = "\n\n".join([
                f"[{doc.metadata.get('file_name', 'Unknown')}]\n{doc.page_content}"
                for doc in filtered_docs
            ])
            
            # 3단계: LLM에 전달할 프롬프트 구성
            # - llm_module.py의 LLM 매니저가 프롬프트 생성
            prompt = self.llm_manager.create_rag_prompt(context_text, query)
            
            # 4단계: LLM에 프롬프트 전달하여 답변 생성
            # - llm_module.py의 LLM 매니저가 답변 생성
            answer = self.llm_manager.generate_answer(prompt)
            
            # 5단계: 참고 문서 정보 정리 (사용자가 출처를 알 수 있도록)
            references = [
                {
                    "doc_id": doc.metadata.get("doc_id"),           # 문서 ID
                    "file_name": doc.metadata.get("file_name"),     # 파일명
                    "cat_id": doc.metadata.get("cat_id"),           # 카테고리 ID
                    "snippet": doc.page_content[:150] + "..."       # 처음 150자 (미리보기)
                }
                for doc in filtered_docs
            ]
            
            # 6단계: 최종 결과 반환
            return {
                "status": "success",
                "query": query,
                "answer": answer,
                "references": references,
                "filters": {
                    "user_id": user_id,
                    "cat_id": cat_id
                }
            }
        
        except Exception as e:
            # 에러 발생 시 처리
            return {
                "status": "error",
                "message": str(e),
                "answer": None,
                "references": []
            }
    
    def get_document_summary(self, doc_id: int, user_id: int) -> str:
        """
        PostgreSQL에서 문서 요약 조회
        
        역할:
        - 특정 문서의 요약 내용을 조회
        - 사용자 검증 포함 (사용자만 자신의 문서 조회 가능)
        
        인자:
        - doc_id (int): 문서 ID
        - user_id (int): 사용자 ID (권한 확인용)
        
        반환값:
        - str: 문서 요약 내용
        - None: 문서 없음 또는 권한 없음
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # SQL: 문서 ID와 사용자 ID로 문서 요약 조회
            cur.execute("""
                SELECT content_full FROM doc 
                WHERE doc_id = %s AND user_id = %s
            """, (doc_id, user_id))
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            return result[0] if result else None
        except Exception as e:
            print(f"✗ 요약 조회 실패: {e}")
            return None
    
    def log_query_history(self, user_id: int, query: str, answer: str, 
                         references: list):
        """
        쿼리 이력을 PostgreSQL history 테이블에 저장
        
        역할:
        - 사용자가 수행한 모든 RAG 질의를 기록
        - 감시/분석 목적으로 사용
        - 사용자의 검색 패턴 분석 가능
        
        저장 내용:
        - user_id: 사용자 ID
        - query_text: 질문 내용
        - answer_text: 생성된 답변
        - doc_ids: 참고한 문서 ID들 (쉼표 구분)
        - created_at: 쿼리 실행 시간
        
        인자:
        - user_id (int): 사용자 ID
        - query (str): 사용자의 질문
        - answer (str): 생성된 답변
        - references (list): 참고 문서 리스트 (각각 doc_id 포함)
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # 참고한 문서 ID들을 쉼표로 구분된 문자열로 변환
            ref_ids = ",".join([str(r.get("doc_id")) for r in references])
            
            # SQL: 쿼리 이력 INSERT
            cur.execute("""
                INSERT INTO history (user_id, query_text, answer_text, doc_ids, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, query, answer, ref_ids, datetime.now()))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[경고] 이력 저장 실패: {e}")
    
    def get_user_statistics(self, user_id: int) -> dict:
        """
        사용자별 통계 정보 조회
        
        역할:
        - 사용자가 업로드한 문서 수
        - 카테고리별 문서 분포
        - 수행한 쿼리 개수
        - 사용자의 활동 분석
        
        반환되는 통계:
        - total_documents: 총 문서 수
        - categories: {카테고리명: 문서수} (예: {"계약서": 15, "보고서": 8})
        - total_queries: 총 쿼리 수
        
        인자:
        - user_id (int): 사용자 ID
        
        반환값:
        - dict: 통계 정보
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # 1단계: 사용자의 총 문서 수 조회
            cur.execute("SELECT COUNT(*) FROM doc WHERE user_id = %s", (user_id,))
            doc_count = cur.fetchone()[0]
            
            # 2단계: 카테고리별 문서 수 조회
            # LEFT JOIN으로 카테고리 정보 함께 조회
            cur.execute("""
                SELECT c.main, COUNT(d.doc_id)
                FROM doc d
                LEFT JOIN category c ON d.cat_id = c.cat_id
                WHERE d.user_id = %s
                GROUP BY c.main
            """, (user_id,))
            # 결과를 {카테고리: 개수} 딕셔너리로 변환
            categories = {cat[0] or "미분류": cat[1] for cat in cur.fetchall()}
            
            # 3단계: 사용자의 총 쿼리 기록 수 조회
            cur.execute("""
                SELECT COUNT(*) FROM history WHERE user_id = %s
            """, (user_id,))
            query_count = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            # 4단계: 통계 정보 반환
            return {
                "user_id": user_id,
                "total_documents": doc_count,        # 총 문서 수
                "categories": categories,             # 카테고리별 분포
                "total_queries": query_count          # 총 쿼리 수
            }
        except Exception as e:
            print(f"✗ 통계 조회 실패: {e}")
            return {}
