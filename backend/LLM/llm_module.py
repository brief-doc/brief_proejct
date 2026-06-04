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
from config import LLM_CONFIG, CURRENT_MODEL
import requests
import time


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
            print(f"[DEBUG] LLM 호출 시작: 모델={LLM_CONFIG['model_name']}, URL={LLM_CONFIG['base_url']}")
            response = self.llm.invoke(prompt)
            
            # response 디버깅
            print(f"[DEBUG] LLM 응답 타입: {type(response)}")
            print(f"[DEBUG] LLM 응답: {response}")
            
            if response is None:
                print("[ERROR] LLM이 None을 반환했습니다")
                return "에러: LLM이 응답하지 않았습니다"
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            if not answer:
                print("[ERROR] LLM 응답 내용이 비어있습니다")
                return "에러: LLM이 비어있는 응답을 반환했습니다"
            
            print(f"[DEBUG] 생성된 답변: {answer[:100]}...")
            return answer
        except Exception as e:
            print(f"✗ LLM 답변 생성 실패: {e}")
            import traceback
            traceback.print_exc()
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
        prompt = f"""다음 법적 문서를 기반으로 질문에 답변하세요.

[참고 문서]
{context_text}

[질문]
{query}

[지시사항]
- 참고 문서의 내용을 바탕으로 질문에 최대한 상세하고 정확하게 답변
- 문서에 명확히 없는 정보만 "문서에서 확인할 수 없습니다"라고 표시
- 판례, 법규, 사실 관계 등을 문서에서 직접 인용하여 설명
- 법적 용어와 판례 내용을 명확하게 전달

[답변]"""  
        return prompt
    
    def get_config_summary(self) -> str:
        """
        현재 LLM 설정을 사용자가 읽기 좋게 반환
        
        반환값:
        - str: LLM 설정 요약
        """
        return f"모델: {CURRENT_MODEL}, 온도: {LLM_CONFIG['temperature']}, URL: {LLM_CONFIG['base_url']}"
    
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


def check_ollama_server() -> dict:
    """
    Ollama 서버 상태 확인
    
    역할:
    - Ollama 서버가 정상 작동하는지 확인
    - 설치된 모델 목록 조회
    
    반환값:
    - {
        "status": "ok" or "error",
        "message": "상태 메시지",
        "models": ["모델명1", "모델명2"],
        "server_url": "http://..."
      }
    """
    server_url = LLM_CONFIG["base_url"]
    
    try:
        # 1. Ollama 서버 연결 확인
        response = requests.get(f"{server_url}/api/tags", timeout=5)
        
        if response.status_code != 200:
            return {
                "status": "error",
                "message": f"Ollama 서버 응답 오류 (상태코드: {response.status_code})",
                "server_url": server_url
            }
        
        # 2. 설치된 모델 목록 조회
        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        
        if not models:
            return {
                "status": "warning",
                "message": "Ollama 서버 연결됨, 하지만 설치된 모델이 없습니다",
                "models": [],
                "server_url": server_url
            }
        
        # 3. 현재 사용 중인 모델이 설치되어 있는지 확인
        current_model = LLM_CONFIG["model_name"]
        model_found = any(current_model in model for model in models)
        
        return {
            "status": "ok",
            "message": f"✓ Ollama 서버 정상 작동 (모델 {len(models)}개)",
            "models": models,
            "server_url": server_url,
            "current_model": current_model,
            "model_found": model_found
        }
        
    except requests.ConnectionError:
        return {
            "status": "error",
            "message": f"Ollama 서버에 연결할 수 없습니다\n\n[해결 방법]\n1. 터미널에서 'ollama serve' 실행\n2. 또는 Ollama 앱 실행",
            "server_url": server_url
        }
    except requests.Timeout:
        return {
            "status": "error",
            "message": "Ollama 서버 응답 타임아웃 (서버가 응답하지 않음)",
            "server_url": server_url
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ollama 서버 상태 확인 실패: {str(e)}",
            "server_url": server_url
        }


def generate_llm_answer(prompt: str) -> str:
    """
    편의 함수: LLM 답변 생성
    
    사용 예:
    from LLM.llm_module import generate_llm_answer
    
    answer = generate_llm_answer(prompt)
    """
    return get_llm_manager().generate_answer(prompt)
