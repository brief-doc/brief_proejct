"""LLM 싱글톤 — Ollama / HuggingFace 공급자 자동 분기

레고 교체 방법:
    .env 또는 환경변수에서 LLM_PROVIDER 만 바꾸면 됩니다.

    LLM_PROVIDER=ollama          → Ollama 로컬 서버 사용 (기본값)
    LLM_PROVIDER=huggingface     → HuggingFace transformers 로컬 모델 사용

HuggingFace 모델 변경:
    HF_MODEL_ID=rudalson/Llama-3.2-3B-Instruct-Legal-Chatbot

공개 API:
    get_llm()         → RAG 답변용 LLM
    get_summary_llm() → 요약 전용 LLM (num_predict 축소)
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import (
    HF_LLM_CONFIG,
    HF_SUMMARY_LLM_CONFIG,
    LLM_CONFIG,
    LLM_PROVIDER,
    SUMMARY_LLM_CONFIG,
)

# ── 싱글톤 저장소 ─────────────────────────────────────────────────────────────
_llm: BaseChatModel | None = None
_summary_llm: BaseChatModel | None = None


# ── 공급자별 LLM 생성 ─────────────────────────────────────────────────────────
def _build_ollama(cfg: dict) -> BaseChatModel:
    """Ollama ChatOllama 생성."""
    from langchain_ollama import ChatOllama

    return ChatOllama(**cfg)


def _build_huggingface(cfg: dict) -> BaseChatModel:
    """HuggingFace 로컬 모델을 ChatHuggingFace 로 래핑하여 반환.

    transformers.pipeline → HuggingFacePipeline → ChatHuggingFace 순으로 래핑합니다.
    ChatHuggingFace 는 ChatOllama 와 동일한 인터페이스(invoke/stream)를 가집니다.
    """
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id: str = cfg["model_id"]
    max_new_tokens: int = cfg.get("max_new_tokens", 512)
    temperature: float = cfg.get("temperature", 0.1)
    device: str = cfg.get("device", "auto")

    # dtype 결정: GPU 없으면 자동으로 float32 로 전환
    dtype_str: str = cfg.get("torch_dtype", "float16")
    if dtype_str == "float16":
        torch_dtype = torch.float16
    elif dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # CPU 강제 시 float16 → float32 (CPU 는 float16 미지원)
    if device == "cpu":
        torch_dtype = torch.float32

    print(f"[llm] HuggingFace 모델 로딩 중: {model_id} (dtype={torch_dtype}, device={device})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,  # temperature=0 이면 greedy decoding
        return_full_text=False,  # 입력 프롬프트를 출력에 포함하지 않음
        pad_token_id=tokenizer.eos_token_id,
    )

    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    # ChatHuggingFace: ChatOllama 와 동일한 메시지 기반 인터페이스 제공
    return ChatHuggingFace(llm=hf_pipeline)


def _build_llm(cfg_ollama: dict, cfg_hf: dict) -> BaseChatModel:
    """LLM_PROVIDER 에 따라 적절한 LLM 을 생성합니다."""
    if LLM_PROVIDER == "huggingface":
        return _build_huggingface(cfg_hf)
    return _build_ollama(cfg_ollama)


# ── 공개 API ──────────────────────────────────────────────────────────────────
def get_llm() -> BaseChatModel:
    """RAG 질의응답용 LLM 싱글톤.

    LLM_PROVIDER 환경변수에 따라 Ollama 또는 HuggingFace 모델을 반환합니다.
    """
    global _llm
    if _llm is None:
        _llm = _build_llm(LLM_CONFIG, HF_LLM_CONFIG)
        print(f"[llm] get_llm 초기화 완료 (provider={LLM_PROVIDER})")
    return _llm


def get_summary_llm() -> BaseChatModel:
    """요약 전용 LLM 싱글톤 — max_new_tokens/num_predict 를 줄여 속도 최적화."""
    global _summary_llm
    if _summary_llm is None:
        _summary_llm = _build_llm(SUMMARY_LLM_CONFIG, HF_SUMMARY_LLM_CONFIG)
        print(f"[llm] get_summary_llm 초기화 완료 (provider={LLM_PROVIDER})")
    return _summary_llm


def reload_llm() -> None:
    """LLM 싱글톤을 초기화합니다 (모델 변경 후 재로드 시 사용)."""
    global _llm, _summary_llm
    _llm = None
    _summary_llm = None
    print("[llm] 싱글톤 초기화 완료 — 다음 호출 시 재생성됩니다.")


# 하위 호환
get_llm_manager = get_llm


def generate_llm_answer(prompt: str) -> str:
    return get_llm().invoke(prompt).content
