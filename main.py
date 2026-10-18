"""Telegram-бот автоответчик с ИИ (Claude / Mistral).

Запуск:  python main.py
Остановка: Ctrl+C
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import ErrorEvent, Message

from config import BASE_DIR, ConfigError, load_config
from memory import MemoryStore
from providers import AIError, AIProvider, build_provider

log = logging.getLogger("bot")

TELEGRAM_TEXT_LIMIT = 4096


def split_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Режет длинный ответ на части по границам абзацев/слов."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


class TypingIndicator:
    """Показывает «печатает…», пока ИИ думает (действие живёт ~5 секунд)."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            try:
                await self._bot.send_chat_action(self._chat_id, ChatAction.TYPING)
            except Exception:  # noqa: BLE001 — индикация не должна ронять бота
                return
            await asyncio.sleep(4)

    def __enter__(self) -> "TypingIndicator":
        self._task = asyncio.create_task(self._loop())
        return self

    def __exit__(self, *exc_info) -> None:
        if self._task:
            self._task.cancel()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    )
    logging.getLogger("aiogram").setLevel(logging.WARNING)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"\n[ошибка настройки] {exc}\n")
        return

    echo_mode = os.getenv("ECHO_MODE", "").strip().lower() in ("1", "true", "yes", "on")
    provider = build_provider(cfg, echo=echo_mode)
    memory = MemoryStore(limit=cfg.history_limit, path=BASE_DIR / "data" / "history.json")

    bot = Bot(token=cfg.bot_token, default=DefaultBotProperties())
    dp = Dispatcher()

    # Один чат обрабатывается строго последовательно: история не перепутается,
    # если пользователь пришлёт несколько сообщений подряд.
    chat_locks: dict[int, asyncio.Lock] = {}

    def lock_for(chat_id: int) -> asyncio.Lock:
        return chat_locks.setdefault(chat_id, asyncio.Lock())

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        await message.answer(
            "Привет! Я автоответчик на базе ИИ. Напишите мне любое сообщение — отвечу.\n"
            "Команда /reset очищает историю нашего диалога."
        )

    @dp.message(Command("reset"))
    async def cmd_reset(message: Message, memory: MemoryStore) -> None:
        await memory.clear(message.chat.id)
        await message.answer("История диалога очищена. Начнём заново.")

    @dp.message(F.text)
    async def handle_text(message: Message, provider: AIProvider, memory: MemoryStore) -> None:
        chat_id = message.chat.id
        text = (message.text or "").strip()
        if not text:
            return

        async with lock_for(chat_id):
            await memory.append(chat_id, "user", text)
            history = await memory.get_history(chat_id)

            with TypingIndicator(bot, chat_id):
                try:
                    answer = await provider.generate(cfg.system_prompt, history)
                except AIError as exc:
                    log.error("Ошибка генерации для чата %s: %s", chat_id, exc)
                    await message.answer(
                        "Извините, сейчас не удалось подготовить ответ. Попробуйте чуть позже."
                    )
                    return
                except Exception:  # noqa: BLE001
                    log.exception("Неожиданная ошибка при генерации ответа")
                    await message.answer(
                        "Произошла непредвиденная ошибка. Я записал её в журнал."
                    )
                    return

            await memory.append(chat_id, "assistant", answer)

            for part in split_text(answer):
                await message.answer(part)

        log.info("Чат %s: ответ отправлен (%d симв.)", chat_id, len(answer))

    @dp.message()
    async def handle_other(message: Message) -> None:
        await message.answer("Я пока понимаю только текстовые сообщения.")

    @dp.errors()
    async def on_error(event: ErrorEvent) -> None:
        log.exception("Необработанная ошибка: %s", event.exception)

    me = await bot.get_me()
    log.info(
        "Бот @%s запущен. Провайдер: %s (%s). Чатов в памяти: %d",
        me.username, provider.name, cfg.model_name, memory.chats_count(),
    )

    try:
        await dp.start_polling(bot, provider=provider, memory=memory, cfg=cfg)
    finally:
        await provider.close()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Бот остановлен.")
