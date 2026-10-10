"""Настройки из .env. Ключи не печатаются и не хранятся в коде."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _required(name: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise RuntimeError(f"В .env не задано значение {name}.")
    return cleaned


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_api_key: str
    openai_base_url: str
    chat_model: str
    embedding_model: str
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str


def load_settings() -> Settings:
    proxy_key = os.getenv("PROXYAPI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip() or proxy_key
    # Haystack читает ключ OpenAI из окружения, если он не передан явно.
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    if pinecone_key:
        os.environ["PINECONE_API_KEY"] = pinecone_key

    index = os.getenv("PINECONE_INDEX_NAME", "").strip() or os.getenv("PINECONE_INDEX", "").strip()

    return Settings(
        telegram_token=_required("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", "")),
        openai_api_key=_required("PROXYAPI_API_KEY или OPENAI_API_KEY", openai_key),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/v1").strip(),
        chat_model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small").strip(),
        pinecone_api_key=_required("PINECONE_API_KEY", pinecone_key),
        pinecone_index=_required("PINECONE_INDEX_NAME", index),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "team-chat").strip() or "team-chat",
    )
