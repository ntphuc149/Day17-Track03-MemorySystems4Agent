from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Cấu hình provider dùng chung cho main model và judge model."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Chuẩn hóa tên provider, xử lý typo phổ biến."""
    aliases = {
        "anthorpic": "anthropic",
        "antrhropic": "anthropic",
        "open_ai": "openai",
        "open-ai": "openai",
        "gpt": "openai",
        "google": "gemini",
        "google-genai": "gemini",
    }
    v = value.lower().strip()
    return aliases.get(v, v)


def build_chat_model(config: ProviderConfig):
    """Khởi tạo LangChain chat model theo provider được chọn.

    Trả về None nếu package chưa cài (để offline mode vẫn chạy được).
    """
    provider = normalize_provider(config.provider)

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )

        if provider == "custom":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or "custom",
                base_url=config.base_url,
            )

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=config.api_key,
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url or "http://localhost:11434",
            )

        if provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    except ImportError:
        # Package chưa cài -> chạy offline mode
        return None

    raise ValueError(f"Provider không hỗ trợ: {config.provider!r}")
