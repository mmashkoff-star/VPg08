# Командный Telegram-бот

Бот для рабочего чата. Между командами `/listen` и `/stop` он сохраняет реплики в Pinecone, а по запросу отвечает по этому контексту: кто и когда что сказал, и что из обсуждения следует. Если отрезок был спором, в итоге бот даёт своё мнение. Если договорились о действиях — кратко фиксирует решение и следующие шаги.

Эмбеддинги и генерация текста идут через OpenAI-совместимый [ProxyAPI](https://proxyapi.ru). Пайплайны собраны на Haystack: запись в индекс, поиск с `ChatPromptBuilder` и отдельное резюме диалога.

Аккаунт бота: [@MMVPf02_bot](https://t.me/MMVPf02_bot).

## Команды

| Команда | Что делает |
| --- | --- |
| `/listen` | Начинает запись сообщений этого чата в Pinecone. |
| `/stop` | Останавливает запись, сохраняет диалог и присылает итог. |
| `/start`, `/help` | Короткая справка. |

В группе вопрос задают так: `@MMVPf02_bot что думаешь?` или ответом на сообщение бота. В личном чате упоминание не нужно: бот отвечает на каждое сообщение.

На вопрос вроде «что думаешь?» бот опирается на последние реплики чата, а не ищет в базе саму эту фразу.

## Подготовка группы

1. Добавьте `@MMVPf02_bot` в группу.
2. В @BotFather откройте бота → Bot Settings → Group Privacy → **Turn off**. Иначе Telegram отдаёт боту только команды и упоминания. Тот же доступ даёт роль администратора.
3. В группе отправьте `/listen`. В базу попадают только сообщения после этой команды и до `/stop`.

## Запуск

Нужен Python 3.14. Зависимости стоят в виртуальном окружении `.venv`, не в системном Python.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Если запустить `main.py` системным интерпретатором, появится `ModuleNotFoundError: No module named 'haystack'`.

## Настройки

Скопируйте значения в файл `.env` в корне проекта. Файл не коммитится.

```env
PROXYAPI_API_KEY=
OPENAI_BASE_URL=https://api.proxyapi.ru/v1
OPENAI_MODEL=openai/gpt-4o-mini
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small

PINECONE_API_KEY=
PINECONE_INDEX_NAME=vpg08
# PINECONE_NAMESPACE=team-chat

TELEGRAM_BOT_TOKEN=
```

`OPENAI_API_KEY` можно указать вместо `PROXYAPI_API_KEY`. Если `PINECONE_NAMESPACE` пустой, используется пространство `team-chat`. Размерность эмбеддинга подстраивается под уже существующий индекс. Для `text-embedding-3-small` индекс не должен быть шире 1536.

## Журналы

| Файл | Содержимое |
| --- | --- |
| `logs/bot.log` | Запуск, ошибки, обращения к Pinecone и ProxyAPI. |
| `logs/correspondence.log` | Входящие сообщения и ответы бота. |

Окно `/listen` хранится в памяти процесса. После перезапуска запись нужно включить снова. Уже записанные реплики в Pinecone остаются.

## Состав

- `main.py` — запуск.
- `team_bot/telegram_bot.py` — команды, упоминания, запись переписки.
- `team_bot/pipelines.py` — пайплайны Haystack: индексация, ответ по контексту, резюме.
- `team_bot/sessions.py` — активные окна записи по чатам.
- `requirements.txt` — `haystack-ai`, `pinecone-haystack`, `pyTelegramBotAPI`, `python-dotenv`.
