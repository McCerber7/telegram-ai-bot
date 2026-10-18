"""Смоук-тесты логики, не требующие API-ключей и сети.

Запуск: python -m pytest -q
"""
from __future__ import annotations

import asyncio

import pytest

from config import Config, ConfigError
from main import split_text
from memory import MemoryStore
from providers import EchoProvider, build_provider


def _cfg(**overrides) -> Config:
    base = dict(
        bot_token="x",
        provider="claude",
        anthropic_api_key="k",
        claude_model="claude-opus-5-5",
        claude_effort="low",
        use_fallbacks=True,
        mistral_api_key="m",
        mistral_model="mistral-small-latest",
        mistral_temperature=0.7,
        system_prompt="role",
        history_limit=20,
        max_tokens=4096,
    )
    base.update(overrides)
    return Config(**base)


# --- split_text ---

def test_split_short_text_unchanged():
    assert split_text("привет") == ["привет"]


def test_split_long_text_respects_limit():
    text = "слово " * 2000  # много слов
    parts = split_text(text, limit=4096)
    assert all(len(p) <= 4096 for p in parts)
    assert "".join(parts).replace(" ", "").replace("\n", "") == text.replace(" ", "")


def test_split_prefers_paragraph_boundary():
    text = ("a" * 100 + "\n") * 60
    parts = split_text(text, limit=4096)
    assert len(parts) >= 2
    assert all(len(p) <= 4096 for p in parts)


# --- MemoryStore ---

def test_memory_append_and_trim(tmp_path):
    async def run():
        store = MemoryStore(limit=4, path=tmp_path / "h.json")
        for i in range(6):
            await store.append(1, "user", f"m{i}")
        history = await store.get_history(1)
        return [t["content"] for t in history]

    result = asyncio.run(run())
    assert result == ["m2", "m3", "m4", "m5"]  # лимит 4, старое обрезано


def test_memory_history_starts_with_user(tmp_path):
    async def run():
        store = MemoryStore(limit=10, path=tmp_path / "h.json")
        await store.append(1, "assistant", "leading")  # ведущее assistant
        await store.append(1, "user", "hello")
        await store.append(1, "assistant", "hi")
        return await store.get_history(1)

    history = asyncio.run(run())
    assert history[0]["role"] == "user"


def test_memory_persists_across_instances(tmp_path):
    path = tmp_path / "h.json"

    async def write():
        store = MemoryStore(limit=10, path=path)
        await store.append(7, "user", "запомни")
        await store.append(7, "assistant", "запомнил")

    async def read():
        store = MemoryStore(limit=10, path=path)
        return await store.get_history(7)

    asyncio.run(write())
    history = asyncio.run(read())
    assert [t["content"] for t in history] == ["запомни", "запомнил"]


def test_memory_clear(tmp_path):
    async def run():
        store = MemoryStore(limit=10, path=tmp_path / "h.json")
        await store.append(3, "user", "x")
        await store.clear(3)
        return await store.get_history(3)

    assert asyncio.run(run()) == []


# --- providers ---

def test_build_provider_claude_and_mistral():
    assert build_provider(_cfg(provider="claude")).name == "claude"
    assert build_provider(_cfg(provider="mistral")).name == "mistral"


def test_build_provider_echo_override():
    assert build_provider(_cfg(), echo=True).name == "echo"


def test_build_provider_unknown_raises():
    with pytest.raises(ValueError):
        build_provider(_cfg(provider="gpt"))


def test_echo_provider_returns_last_user():
    async def run():
        p = EchoProvider()
        return await p.generate("role", [{"role": "user", "content": "тест"}])

    out = asyncio.run(run())
    assert "тест" in out


# --- config validation ---

def test_config_rejects_missing_required_key(monkeypatch, tmp_path):
    monkeypatch.setenv("PROVIDER", "claude")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("BOT_TOKEN", "tok")
    # prompt file must exist
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", "prompt.txt")

    import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "BASE_DIR", tmp_path)
    (tmp_path / "prompt.txt").write_text("role", encoding="utf-8")

    with pytest.raises(ConfigError):
        cfg_mod.load_config()
