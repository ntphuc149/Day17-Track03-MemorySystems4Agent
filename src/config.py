from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int   # số token tối đa trước khi compact
    compact_keep_messages: int       # số message gần nhất giữ lại sau compact
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Đọc env vars và trả về LabConfig đầy đủ."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Tìm file .env ở root repo nếu có
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # --- Provider chính ---
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    model_name = os.getenv("LLM_MODEL", _default_model(provider))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    api_key = _api_key_for(provider)
    base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # --- Judge model (dùng để đánh giá response quality nếu cần) ---
    judge_provider = os.getenv("JUDGE_PROVIDER", provider).lower().strip()
    judge_model_name = os.getenv("JUDGE_MODEL", _default_model(judge_provider))
    judge_api_key = _api_key_for(judge_provider)

    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=os.getenv("CUSTOM_BASE_URL") or os.getenv("OLLAMA_BASE_URL"),
    )

    # --- Compact memory defaults ---
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # --- Tạo thư mục state nếu chưa có ---
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_cfg,
        judge_model=judge_cfg,
    )


def _default_model(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-haiku-4-5-20251001",
        "ollama": "llama3",
        "openrouter": "openai/gpt-4o-mini",
        "custom": "gpt-4o-mini",
    }
    return defaults.get(provider, "gpt-4o-mini")


def _api_key_for(provider: str) -> str | None:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "ollama": None,
    }
    env_var = mapping.get(provider)
    return os.getenv(env_var) if env_var else None
