"""Журнал работы бота и файл переписки."""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

from team_bot.config import ROOT

LOG_DIR = ROOT / "logs"
BOT_LOG = LOG_DIR / "bot.log"
CORRESPONDENCE_LOG = LOG_DIR / "correspondence.log"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(BOT_LOG, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console)


class CorrespondenceLog:
    """Дописывает каждую входящую реплику и каждый ответ бота в один текстовый файл."""

    def __init__(self, path: Path = CORRESPONDENCE_LOG) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def incoming(self, chat_id: int, chat_title: str, author: str, text: str) -> None:
        self._write("входящее", chat_id, chat_title, author, text)

    def outgoing(self, chat_id: int, chat_title: str, author: str, text: str) -> None:
        self._write("исходящее", chat_id, chat_title, author, text)

    def _write(self, direction: str, chat_id: int, chat_title: str, author: str, text: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = f" «{chat_title}»" if chat_title else ""
        body = " / ".join(line.strip() for line in text.splitlines() if line.strip())
        line = f"[{stamp}] чат {chat_id}{title} | {direction} | {author}: {body}\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
