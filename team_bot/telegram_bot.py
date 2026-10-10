"""Telegram-бот на PyTelegramBotAPI.

/listen включает запись сообщений чата в Pinecone.
/stop останавливает запись, сохраняет диалог и просит модель подвести итог.
Упоминание бота (@username или ответ на его сообщение) запускает поиск по
уже сохранённому контексту этого чата.
"""

from __future__ import annotations

import logging
from threading import Lock

import telebot
from haystack import Document
from telebot.types import BotCommand, Message

from team_bot.logging_setup import CorrespondenceLog
from team_bot.pipelines import answer_question, index_documents, summarize_dialogue
from team_bot.sessions import RecordedMessage, SessionStore, now_label

logger = logging.getLogger(__name__)

TELEGRAM_TEXT_LIMIT = 4000
DIALOGUE_CHUNK_CHARS = 1500


class TeamAssistantBot:
    def __init__(
        self,
        token: str,
        indexing_pipeline,
        rag_pipeline,
        summary_pipeline,
        correspondence: CorrespondenceLog,
    ) -> None:
        self.bot = telebot.TeleBot(token, parse_mode=None)
        self.indexing_pipeline = indexing_pipeline
        self.rag_pipeline = rag_pipeline
        self.summary_pipeline = summary_pipeline
        self.correspondence = correspondence
        self.sessions = SessionStore()
        self._pipeline_lock = Lock()
        self.username = ""
        self.bot_id = 0
        self._register_handlers()

    def run(self) -> None:
        me = self.bot.get_me()
        self.username = me.username or ""
        self.bot_id = me.id
        self.bot.set_my_commands(
            [
                BotCommand("listen", "Начать запись обсуждения"),
                BotCommand("stop", "Остановить запись и подвести итог"),
            ]
        )
        logger.info("Бот @%s запущен. Жду сообщения.", self.username)
        self.bot.infinity_polling(skip_pending=True, allowed_updates=["message"])

    def _register_handlers(self) -> None:
        @self.bot.message_handler(commands=["start", "help"])
        def on_help(message: Message) -> None:
            self._note_incoming(message)
            self._reply(
                message,
                "Я записываю рабочие обсуждения и отвечаю по ним.\n\n"
                "/listen — начать слушать чат и складывать реплики в векторную базу.\n"
                "/stop — остановить, сохранить диалог и прислать итог: "
                "мнение, если был спор, или финальное решение, если договорились о действиях.\n\n"
                "Чтобы спросить по уже сохранённому контексту, упомяните меня в сообщении "
                "или ответьте на моё сообщение.\n\n"
                "В группе у бота должен быть выключен privacy mode в @BotFather "
                "(Bot Settings → Group Privacy → Turn off), иначе Telegram не отдаёт чужие реплики.",
            )

        @self.bot.message_handler(commands=["listen"])
        def on_listen(message: Message) -> None:
            self._note_incoming(message)
            if self.sessions.start(message.chat.id):
                self._reply(message, "Слушаю. Следующие сообщения чата записываю, пока не придёт /stop.")
                return
            self._reply(message, "Запись в этом чате уже идёт. Остановить её можно командой /stop.")

        @self.bot.message_handler(commands=["stop"])
        def on_stop(message: Message) -> None:
            self._note_incoming(message)
            session = self.sessions.stop(message.chat.id)
            if session is None:
                self._reply(message, "Сейчас я не записываю этот чат. Начать можно командой /listen.")
                return
            if not session.messages:
                self._reply(message, "За время записи новых сообщений не было. В базу ничего не добавил.")
                return
            self._finish_session(message, session)

        @self.bot.message_handler(content_types=["text", "photo", "video", "document", "voice", "audio", "sticker"])
        def on_message(message: Message) -> None:
            if message.from_user is None or message.from_user.is_bot:
                return
            self._note_incoming(message)
            text = message_text(message)
            if not text:
                return
            if self._is_addressed(message):
                self._answer(message, strip_mention(text, self.username))
            if self.sessions.is_active(message.chat.id):
                self._record(message, text)
            if not self._is_addressed(message):
                logger.info(
                    "Сообщение %s в чате %s без упоминания бота, ответ не отправляю.",
                    message.message_id,
                    message.chat.id,
                )

    def _record(self, message: Message, text: str) -> None:
        user = message.from_user
        author = user.full_name or user.username or str(user.id)
        recorded = RecordedMessage(
            author=author,
            username=user.username or "",
            text=text,
            timestamp=now_label(),
            message_id=message.message_id,
        )
        self.sessions.add(message.chat.id, recorded)
        document = Document(
            id=f"msg-{message.chat.id}-{message.message_id}",
            content=recorded.as_line(),
            meta=message_meta(message, author, kind="message"),
        )
        try:
            with self._pipeline_lock:
                index_documents(self.indexing_pipeline, [document])
        except Exception:
            logger.exception("Не удалось записать сообщение %s в Pinecone.", message.message_id)
            self._reply(message, "Сообщение осталось в текущей записи, но в Pinecone оно сейчас не попало.")

    def _finish_session(self, message: Message, session) -> None:
        self.bot.send_chat_action(message.chat.id, "typing")
        dialogue = session.transcript()
        try:
            with self._pipeline_lock:
                summary = summarize_dialogue(self.summary_pipeline, dialogue)
        except Exception:
            logger.exception("Не удалось подвести итог обсуждения в чате %s.", message.chat.id)
            self._reply(message, "Сообщения за этот отрезок уже лежат в базе по одной реплике, но общее резюме сейчас не собралось.")
            return
        self._reply(message, summary)
        try:
            with self._pipeline_lock:
                documents = dialogue_documents(message, dialogue, summary, session.started_at)
                index_documents(self.indexing_pipeline, documents)
        except Exception:
            logger.exception("Не удалось сохранить итог обсуждения чата %s в Pinecone.", message.chat.id)
            self._reply(message, "Итог выше уже отправлен, но целиком в векторную базу этот диалог сейчас не записался.")

    def _answer(self, message: Message, question: str) -> None:
        question = question.strip()
        if not question:
            self._reply(message, "Напишите вопрос сразу после упоминания, и я поищу ответ в сохранённых обсуждениях.")
            return
        self.bot.send_chat_action(message.chat.id, "typing")
        try:
            with self._pipeline_lock:
                answer = answer_question(
                    self.rag_pipeline,
                    question,
                    message.chat.id,
                    recent_dialogue=self.sessions.transcript(message.chat.id),
                )
        except Exception:
            logger.exception("Не удалось ответить на вопрос в чате %s.", message.chat.id)
            self._reply(message, "Не получилось просмотреть сохранённый контекст. Попробуйте ещё раз чуть позже.")
            return
        self._reply(message, answer)

    def _is_addressed(self, message: Message) -> bool:
        if message.chat.type == "private":
            return True
        reply = message.reply_to_message
        if reply is not None and reply.from_user is not None and reply.from_user.id == self.bot_id:
            return True
        username = self.username.lower()
        if not username:
            return False
        needle = f"@{username}"
        text = message.text or message.caption or ""
        if needle in text.lower():
            return True
        for entity in message.entities or message.caption_entities or []:
            if entity.type == "text_mention" and entity.user is not None and entity.user.id == self.bot_id:
                return True
        return False

    def _note_incoming(self, message: Message) -> None:
        self.correspondence.incoming(
            message.chat.id,
            message.chat.title or "",
            author_label(message),
            message_text(message) or f"[{message.content_type}]",
        )

    def _reply(self, message: Message, text: str) -> None:
        author = f"@{self.username}" if self.username else "бот"
        self.correspondence.outgoing(message.chat.id, message.chat.title or "", author, text)
        for chunk in split_text(text):
            try:
                self.bot.reply_to(message, chunk)
            except Exception:
                logger.exception("Не удалось отправить ответ в чат %s.", message.chat.id)


def author_label(message: Message) -> str:
    user = message.from_user
    if user is None:
        return "неизвестно"
    name = user.full_name or user.username or str(user.id)
    if user.username:
        return f"{name} (@{user.username})"
    return name


def message_text(message: Message) -> str:
    if message.text:
        return message.text.strip()
    if message.caption:
        return message.caption.strip()
    labels = {
        "photo": "[фото без подписи]",
        "video": "[видео без подписи]",
        "document": "[файл без подписи]",
        "voice": "[голосовое сообщение]",
        "audio": "[аудио без подписи]",
        "sticker": "[стикер]",
    }
    return labels.get(message.content_type, "")


def strip_mention(text: str, username: str) -> str:
    if not username:
        return text
    cleaned = text.replace(f"@{username}", "").replace(f"@{username.lower()}", "")
    return " ".join(cleaned.split())


def message_meta(message: Message, author: str, kind: str) -> dict:
    user = message.from_user
    meta = {
        "kind": kind,
        "author": author,
        "author_id": int(user.id),
        "chat_id": str(message.chat.id),
        "chat_title": message.chat.title or message.chat.type or "",
        "message_id": int(message.message_id),
        "timestamp": now_label(),
    }
    if user.username:
        meta["username"] = user.username
    return meta


def dialogue_documents(message: Message, dialogue: str, summary: str, started_at: str) -> list[Document]:
    user = message.from_user
    author = user.full_name or user.username or str(user.id)
    stamp = now_label()
    base_meta = message_meta(message, author, kind="dialogue")
    base_meta["session_started_at"] = started_at
    documents: list[Document] = []
    chunks = split_text(dialogue, DIALOGUE_CHUNK_CHARS)
    for index, chunk in enumerate(chunks, start=1):
        documents.append(
            Document(
                id=f"dialogue-{message.chat.id}-{message.message_id}-{index}",
                content=f"Фрагмент диалога {index}/{len(chunks)}, запись с {started_at} до {stamp}.\n{chunk}",
                meta={**base_meta, "part": index},
            )
        )
    documents.append(
        Document(
            id=f"summary-{message.chat.id}-{message.message_id}",
            content=f"Итог обсуждения с {started_at} до {stamp}.\n{summary}",
            meta={**base_meta, "kind": "summary"},
        )
    )
    return documents


def split_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return [cleaned]
    parts: list[str] = []
    rest = cleaned
    while rest:
        if len(rest) <= limit:
            parts.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return [part for part in parts if part]
