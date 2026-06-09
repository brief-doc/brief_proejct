"""Ollama LLM 싱글톤"""

from langchain_ollama import ChatOllama

from .config import LLM_CONFIG

_llm = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(**LLM_CONFIG)
    return _llm


# 하위 호환
get_llm_manager = get_llm


def generate_llm_answer(prompt: str) -> str:
    return get_llm().invoke(prompt).content
