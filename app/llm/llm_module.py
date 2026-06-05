"""
LLM 모듈 - 대형 언어 모델 관리
=============================
역할:
- LLM 모델 초기화 및 관리
- 프롬프트 기반 답변 생성
- LLM 설정 중앙화 (config.py 참조)

주의:
- 이 모듈은 LLM 팀이 담당합니다
- 임베딩, RAG 파이프라인은 다른 팀이 관리합니다
- 설정 변경은 config.py에서만 수정하세요
"""

from langchain_ollama import ChatOllama
from config import LLM_CONFIG


class LLMManager:
    """
    LLM 모델 관리 및 답변 생성 클래스
    
    역할:
    - ChatOllama 모델 초기화
    - 주어진 프롬프트에 대한 답변 생성
    - LLM 설정 중앙화 (config.py에서 모두 관리)
    """
    
    def __init__(self):
        """
        LLM 모델 초기화
        
        역할:
        - ChatOllama 모델 로드
        - 모든 설정은 config.py의 LLM_CONFIG에서 관리됨
        - 모델 변경 시 config.py만 수정하면 됨
        """
        try:
            self.llm = ChatOllama(
                model=LLM_CONFIG["model_name"],          # 사용할 모델명 (config.py에서 설정)
                temperature=LLM_CONFIG["temperature"],   # 온도 (config.py에서 설정)
                base_url=LLM_CONFIG["base_url"]          # Ollama 서버 주소 (config.py에서 설정)
            )
            print(f"✓ LLM 모델 로드 성공: {LLM_CONFIG['model_name']}")
        except Exception as e:
            print(f"✗ LLM 모델 로드 실패: {e}")
            raise
    
    def generate_answer(self, prompt: str) -> str:
        """
        주어진 프롬프트에 대한 LLM 답변 생성
        
        역할:
        - 프롬프트를 LLM에 전달
        - LLM의 응답을 처리하여 반환
        
        인자:
        - prompt (str): LLM에 전달할 프롬프트
        
        반환값:
        - str: LLM이 생성한 답변
        """
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer
        except Exception as e:
            print(f"✗ LLM 답변 생성 실패: {e}")
            return f"에러: {str(e)}"
    
    def create_rag_prompt(self, context_text: str, query: str) -> str:
        """
        RAG 기반 질의응답을 위한 프롬프트 생성
        
        역할:
        - 문서 컨텍스트와 사용자 질문을 포함한 프롬프트 생성
        - LLM이 신뢰성 있는 답변을 생성하도록 지시
        
        인자:
        - context_text (str): 검색된 문서 컨텍스트 (RAG 파이프라인에서 제공)
        - query (str): 사용자의 질문
        
        반환값:
        - str: LLM에 전달할 프롬프트
        """
        prompt = f"""다음 문서를 기반으로 질문에 답변하세요.

[참고 문서]
{context_text}

[질문]
{query}

[지시사항]
- 문서에 없는 정보는 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변
- 정확하고 간결하게 답변
- 필요하면 문서의 내용을 인용

[답변]"""
        return prompt
    
    def get_config_summary(self) -> str:
        """
        현재 LLM 설정을 사용자가 읽기 좋게 반환
        
        반환값:
        - str: LLM 설정 요약
        """
        return f"모델: {LLM_CONFIG['model_name']}, 온도: {LLM_CONFIG['temperature']}, URL: {LLM_CONFIG['base_url']}"
    
    def get_llm_instance(self):
        """
        내부 LLM 인스턴스 반환
        
        역할:
        - LangChain의 load_summarize_chain 등에서 직접 LLM이 필요한 경우 사용
        
        주의:
        - 일반적으로 generate_answer()를 사용할 것을 권장합니다
        
        반환값:
        - ChatOllama 인스턴스
        """
        return self.llm


# 전역 LLM 매니저 인스턴스 (처음 사용할 때 초기화)
_llm_manager = None


def get_llm_manager() -> LLMManager:
    """
    LLM 매니저 싱글톤 인스턴스 반환
    
    역할:
    - 전체 애플리케이션에서 LLM 매니저를 단 하나만 사용하도록 보장
    - 처음 호출 시 초기화, 이후는 기존 인스턴스 반환
    
    사용 예:
    from LLM.llm_module import get_llm_manager
    
    llm_manager = get_llm_manager()
    answer = llm_manager.generate_answer(prompt)
    """
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def generate_llm_answer(prompt: str) -> str:
    """
    편의 함수: LLM 답변 생성
    
    사용 예:
    from LLM.llm_module import generate_llm_answer
    
    answer = generate_llm_answer(prompt)
    """
    return get_llm_manager().generate_answer(prompt)
