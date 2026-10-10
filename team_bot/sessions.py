"""Активные окна записи: от /listen до /stop, отдельно для каждого чата."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock


def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


@dataclass
class RecordedMessage:
    author: str
    username: str
    text: str
    timestamp: str
    message_id: int

    def as_line(self) -> str:
        handle = f" (@{self.username})" if self.username else ""
        return f"[{self.timestamp}] {self.author}{handle}: {self.text}"


@dataclass
class ListeningSession:
    started_at: str
    messages: list[RecordedMessage] = field(default_factory=list)

    def transcript(self) -> str:
        if not self.messages:
            return ""
        participants = []
        seen = set()
        for item in self.messages:
            if item.author not in seen:
                seen.add(item.author)
                participants.append(item.author)
        header = (
            f"Начало: {self.started_at}. Конец: {now_label()}.\n"
            f"Участники: {', '.join(participants)}."
        )
        body = "\n".join(item.as_line() for item in self.messages)
        return f"{header}\n\n{body}"


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[int, ListeningSession] = {}
        self._lock = Lock()

    def start(self, chat_id: int) -> bool:
        """Возвращает False, если запись в этом чате уже идёт."""
        with self._lock:
            if chat_id in self._sessions:
                return False
            self._sessions[chat_id] = ListeningSession(started_at=now_label())
            return True

    def is_active(self, chat_id: int) -> bool:
        with self._lock:
            return chat_id in self._sessions

    def transcript(self, chat_id: int) -> str:
        with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                return ""
            return session.transcript()

    def add(self, chat_id: int, message: RecordedMessage) -> bool:
        with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                return False
            session.messages.append(message)
            return True

    def stop(self, chat_id: int) -> ListeningSession | None:
        with self._lock:
            return self._sessions.pop(chat_id, None)
