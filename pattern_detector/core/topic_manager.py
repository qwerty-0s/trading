"""
core/topic_manager.py
---------------------
Создаёт forum topics в каждой супергруппе и кэширует thread_id в файл
topics_cache.json — при повторном запуске темы НЕ пересоздаются.

Не использует get_forum_topics (метод отсутствует в python-telegram-bot).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional

from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)

# Названия тем и цвета иконок
TF_TOPIC_NAMES: Dict[int, str] = {
    10:  "10 min",
    15:  "15 min",
    30:  "30 min",
    60:  "1 hour",
    120: "2 hours",
    180: "3 hours",
    240: "4 hours",
}

TF_TOPIC_COLORS: Dict[int, int] = {
    10:  7322096,   # синий
    15:  9367192,   # зелёный
    30:  16766590,  # жёлтый
    60:  13338331,  # фиолетовый
    120: 16749490,  # розовый
    180: 16478047,  # красный
    240: 6134861,   # серый
}

# Путь к файлу кэша (рядом с main.py)
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "topics_cache.json")

# chat_id (str в JSON) → {tf_str → thread_id}
TopicMap = Dict[int, Dict[int, int]]


class TopicManager:
    def __init__(self, bot: Bot, timeframes: List[int]) -> None:
        self._bot        = bot
        self._timeframes = timeframes
        self.topic_map: TopicMap = {}

    async def setup(self, assets: list) -> None:
        """Вызвать один раз при старте."""
        self._load_cache()
        logger.info("TopicManager: setting up topics for %d assets …", len(assets))
        await asyncio.gather(
            *[self._setup_group(asset) for asset in assets],
            return_exceptions=True,
        )
        self._save_cache()
        logger.info("TopicManager: ready. topic_map keys: %s",
                    list(self.topic_map.keys()))

    def get_thread_id(self, chat_id: int, tf: int) -> Optional[int]:
        return self.topic_map.get(chat_id, {}).get(tf)

    # ── cache ─────────────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if not os.path.exists(CACHE_FILE):
            return
        try:
            with open(CACHE_FILE, "r") as f:
                raw: dict = json.load(f)
            # JSON keys are always strings — convert back to int
            self.topic_map = {
                int(chat_id): {int(tf): int(tid) for tf, tid in tfs.items()}
                for chat_id, tfs in raw.items()
            }
            logger.info("TopicManager: loaded cache from %s", CACHE_FILE)
        except Exception as exc:
            logger.warning("TopicManager: cache load failed (%s), will recreate", exc)
            self.topic_map = {}

    def _save_cache(self) -> None:
        try:
            # Convert int keys to str for JSON
            serialisable = {
                str(chat_id): {str(tf): tid for tf, tid in tfs.items()}
                for chat_id, tfs in self.topic_map.items()
            }
            with open(CACHE_FILE, "w") as f:
                json.dump(serialisable, f, indent=2)
            logger.info("TopicManager: cache saved to %s", CACHE_FILE)
        except Exception as exc:
            logger.error("TopicManager: cache save failed: %s", exc)

    # ── per-group setup ───────────────────────────────────────────────────────

    async def _setup_group(self, asset) -> None:
        chat_id  = asset.tg_chat_id
        existing = self.topic_map.get(chat_id, {})

        # Only create missing TFs
        missing = [tf for tf in self._timeframes if tf not in existing]
        if not missing:
            logger.info("[%s] all topics already cached, skipping", asset.ticker)
            return

        self.topic_map.setdefault(chat_id, {})

        for tf in missing:
            name  = TF_TOPIC_NAMES[tf]
            color = TF_TOPIC_COLORS[tf]
            tid   = await self._create_topic(chat_id, asset.ticker, tf, name, color)
            if tid is not None:
                self.topic_map[chat_id][tf] = tid

    async def _create_topic(
        self, chat_id: int, ticker: str, tf: int, name: str, color: int
    ) -> Optional[int]:
        try:
            topic = await self._bot.create_forum_topic(
                chat_id    = chat_id,
                name       = name,
                icon_color = color,
            )
            logger.info("[%s] created topic '%s' → thread_id=%d",
                        ticker, name, topic.message_thread_id)
            return topic.message_thread_id
        except TelegramError as exc:
            logger.error("[%s] createForumTopic '%s' failed: %s", ticker, name, exc)
            return None
