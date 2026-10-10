"""Пайплайны Haystack: запись в Pinecone, RAG-ответ и резюме обсуждения.

Схема повторяет учебный RAG-пайплайн из ноутбука
(text embedder → retriever → ChatPromptBuilder → ChatGenerator),
но хранилище — PineconeDocumentStore, а эмбеддинги и генерация идут
через OpenAI-совместимый ProxyAPI (api_base_url).
"""

from __future__ import annotations

import logging

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from team_bot.config import Settings

logger = logging.getLogger(__name__)

# text-embedding-3-small по умолчанию отдаёт 1536 чисел. Если индекс уже
# существует, размерность берём у него: модель умеет ужимать вектор.
DEFAULT_EMBEDDING_DIMENSION = 1536

RAG_TEMPLATE = [
    ChatMessage.from_user(
        """
Ты помощник рабочей команды. Отвечай по-русски, с правильным согласованием.
Ниже текущее обсуждение этого чата и отдельно найденные более ранние фрагменты.

Если спрашивают мнение («что думаешь», «как считаешь», «что лучше» и похожие формулировки),
отвечай по сути обсуждения: кто что предложил и когда, затем своё мнение — что лучше сделать и почему.
Не пиши, что в обсуждениях нет самого вопроса. Вопрос — это просьба высказаться о репликах вокруг.
Если спрашивают факт, ответь по источникам и укажи, кто и когда это сказал.
Только если ни в текущем обсуждении, ни во фрагментах нет ничего по теме, напиши, что в сохранённых обсуждениях этого нет.
Не выдумывай реплики.

Текущее обсуждение:
{{ recent }}

Найденные фрагменты:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Вопрос: {{ question }}
Ответ:
"""
    )
]

SUMMARY_TEMPLATE = [
    ChatMessage.from_user(
        """
Ты помощник рабочей команды. Ниже стенограмма одного отрезка обсуждения,
который записывали между командами «начать» и «остановить».

Сначала определи характер разговора:
- спор или несогласие;
- решение о действиях;
- смешанный разговор или обычное обсуждение без явного итога.

Дальше ответь по-русски, коротко и по делу.

Если это спор: изложи позиции участников (кто что утверждал) и дай своё мнение —
на чьей стороне более сильные аргументы и почему.
Если это решение о действиях: прорезюмируй договорённость и выдай финальный результат —
что решили, кто за что отвечает и какие следующие шаги.
Если разговор смешанный: сделай оба блока, «Спор» и «Решение».
Если явного спора и решения нет: дай краткое резюме, о чём договорились или что осталось открытым.

Опирайся только на стенограмму. Не приписывай людям слова, которых там нет.

Стенограмма:
{{ dialogue }}
"""
    )
]


def open_document_store(settings: Settings) -> tuple[PineconeDocumentStore, int]:
    """Подключается к индексу и возвращает фактическую размерность векторов.

    У text-embedding-3 её можно сузить под уже существующий индекс.
    Если индекса нет, PineconeDocumentStore создаёт его на 1536.
    """
    document_store = build_document_store(settings, DEFAULT_EMBEDDING_DIMENSION)
    document_store.count_documents()
    dimension = int(document_store.dimension)
    if dimension > DEFAULT_EMBEDDING_DIMENSION and "large" not in settings.embedding_model:
        raise RuntimeError(
            f"Индекс {settings.pinecone_index} хранит векторы размерности {dimension}, "
            f"а модель {settings.embedding_model} не отдаёт векторы длиннее {DEFAULT_EMBEDDING_DIMENSION}."
        )
    logger.info(
        "Pinecone: индекс %s, пространство %s, размерность %s.",
        settings.pinecone_index,
        settings.pinecone_namespace,
        dimension,
    )
    return document_store, dimension


def build_document_store(settings: Settings, dimension: int) -> PineconeDocumentStore:
    return PineconeDocumentStore(
        api_key=Secret.from_token(settings.pinecone_api_key),
        index=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
        dimension=dimension,
        metric="cosine",
        show_progress=False,
    )


def _openai_key(settings: Settings) -> Secret:
    return Secret.from_token(settings.openai_api_key)


def build_indexing_pipeline(settings: Settings, document_store: PineconeDocumentStore, dimension: int) -> Pipeline:
    """Эмбеддинг документов и запись в Pinecone. Так устроен indexing pipeline в документации 2026."""
    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=_openai_key(settings),
            model=settings.embedding_model,
            api_base_url=settings.openai_base_url,
            dimensions=dimension,
            progress_bar=False,
            raise_on_failure=True,
        ),
    )
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
    )
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


def build_rag_pipeline(settings: Settings, document_store: PineconeDocumentStore, dimension: int) -> Pipeline:
    """Тот же каркас, что в ноутбуке 27_First_RAG_Pipeline: embedder, retriever, prompt, llm."""
    pipeline = Pipeline()
    pipeline.add_component(
        "text_embedder",
        OpenAITextEmbedder(
            api_key=_openai_key(settings),
            model=settings.embedding_model,
            api_base_url=settings.openai_base_url,
            dimensions=dimension,
        ),
    )
    pipeline.add_component(
        "retriever",
        PineconeEmbeddingRetriever(document_store=document_store, top_k=8),
    )
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(template=RAG_TEMPLATE, required_variables="*"),
    )
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=_openai_key(settings),
            model=settings.chat_model,
            api_base_url=settings.openai_base_url,
        ),
    )
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


def build_summary_pipeline(settings: Settings) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(template=SUMMARY_TEMPLATE, required_variables="*"),
    )
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=_openai_key(settings),
            model=settings.chat_model,
            api_base_url=settings.openai_base_url,
        ),
    )
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


def index_documents(pipeline: Pipeline, documents: list[Document]) -> int:
    if not documents:
        return 0
    result = pipeline.run({"embedder": {"documents": documents}})
    return int(result["writer"]["documents_written"])


def answer_question(pipeline: Pipeline, question: str, chat_id: int, recent_dialogue: str = "") -> str:
    stored = recent_messages_text(pipeline.get_component("retriever").document_store, chat_id)
    recent = recent_dialogue.strip() or stored or "Пока в этом чате нет сохранённых реплик."
    search_text = f"{recent[-1500:]}\n{question}" if stored or recent_dialogue.strip() else question
    result = pipeline.run(
        {
            "text_embedder": {"text": search_text},
            "retriever": {"filters": chat_filters(chat_id), "top_k": 8},
            "prompt_builder": {"question": question, "recent": recent},
        }
    )
    return _reply_text(result)


def recent_messages_text(document_store: PineconeDocumentStore, chat_id: int, limit: int = 12) -> str:
    """Последние реплики чата. Вопрос вроде «что думаешь?» сам по себе их не находит."""
    try:
        documents = document_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.chat_id", "operator": "==", "value": str(chat_id)},
                    {"field": "meta.kind", "operator": "==", "value": "message"},
                ],
            }
        )
    except Exception:
        logger.exception("Не удалось прочитать последние сообщения чата %s.", chat_id)
        return ""
    documents.sort(key=lambda doc: (str(doc.meta.get("timestamp", "")), int(doc.meta.get("message_id") or 0)))
    lines = [doc.content.strip() for doc in documents[-limit:] if doc.content]
    return "\n".join(lines)


def summarize_dialogue(pipeline: Pipeline, dialogue: str) -> str:
    result = pipeline.run({"prompt_builder": {"dialogue": dialogue}})
    return _reply_text(result)


def chat_filters(chat_id: int) -> dict:
    return {
        "operator": "AND",
        "conditions": [
            {"field": "meta.chat_id", "operator": "==", "value": str(chat_id)},
        ],
    }


def _reply_text(result: dict) -> str:
    reply = result["llm"]["replies"][0]
    text = getattr(reply, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts = getattr(reply, "_content", []) or []
    joined = "".join(getattr(part, "text", "") for part in parts).strip()
    if joined:
        return joined
    return "Модель вернула пустой ответ."
