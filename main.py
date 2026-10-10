"""Точка входа командного Telegram-бота."""

from __future__ import annotations

from team_bot.config import load_settings
from team_bot.logging_setup import CorrespondenceLog, setup_logging
from team_bot.pipelines import (
    build_indexing_pipeline,
    build_rag_pipeline,
    build_summary_pipeline,
    open_document_store,
)
from team_bot.telegram_bot import TeamAssistantBot


def main() -> None:
    setup_logging()
    settings = load_settings()
    document_store, dimension = open_document_store(settings)
    bot = TeamAssistantBot(
        token=settings.telegram_token,
        indexing_pipeline=build_indexing_pipeline(settings, document_store, dimension),
        rag_pipeline=build_rag_pipeline(settings, document_store, dimension),
        summary_pipeline=build_summary_pipeline(settings),
        correspondence=CorrespondenceLog(),
    )
    bot.run()


if __name__ == "__main__":
    main()
