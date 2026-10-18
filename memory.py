"""Память диалогов: история сообщений по каждому чату.

Хранится в оперативной памяти и дублируется в JSON-файл, чтобы история
переживала перезапуск бота.
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path

log = logging.getLogger("bot")

Role = str  # "user" | "assistant"
Turn = dict[str, str]  # {"role": ..., "content": ...}


class MemoryStore:
    def __init__(self, limit: int, path: Path) -> None:
        self._limit = limit
        self._path = path
        self._lock = asyncio.Lock()
        self._histories: dict[int, list[Turn]] = {}
        self._load()

    # — публичный интерфейс —

    async def append(self, chat_id: int, role: Role, content: str) -> None:
        async with self._lock:
            history = self._histories.setdefault(chat_id, [])
            history.append({"role": role, "content": content})
            del history[: max(0, len(history) - self._limit)]
            self._save()

    async def get_history(self, chat_id: int) -> list[Turn]:
        """История для отправки в API.

        Anthropic требует, чтобы список начинался с роли user, — после
        обрезки по лимиту ведущие assistant-сообщения отбрасываются.
        """
        async with self._lock:
            history = list(self._histories.get(chat_id, []))
        while history and history[0]["role"] != "user":
            history.pop(0)
        return history

    async def clear(self, chat_id: int) -> None:
        async with self._lock:
            self._histories.pop(chat_id, None)
            self._save()

    def chats_count(self) -> int:
        return len(self._histories)

    # — сохранение на диск —

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._histories = {
                int(chat_id): [
                    {"role": t["role"], "content": t["content"]}
                    for t in turns
                    if t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str)
                ]
                for chat_id, turns in raw.items()
            }
            log.info("Загружена история %d чат(ов) из %s", len(self._histories), self._path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning("Не удалось прочитать %s (%s) — начинаем с чистой истории.", self._path, exc)
            self._histories = {}

    def _save(self) -> None:
        """Атомарная запись: сначала временный файл, потом замена."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            tmp = Path(tmp_name)
            with open(fd, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self._histories.items()}, f, ensure_ascii=False)
            tmp.replace(self._path)
        except OSError as exc:
            log.warning("Не удалось сохранить историю на диск: %s", exc)
