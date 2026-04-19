"""
TelegramRouter
--------------
Отправляет сигналы в нужную супергруппу в нужную тему (message_thread_id).
Формат сообщения адаптирован под паттерны PatternDetector.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from config import TELEGRAM_BOT_TOKEN, TG_SEND_TIMEOUT
from core.candle_aggregator import Candle

logger = logging.getLogger(__name__)

_BULLISH_KEYWORDS = {'bull', 'hammer', 'morning', 'soldier', 'piercing', 'inverted hammer'}
_BEARISH_KEYWORDS = {'bear', 'hanging', 'shooting', 'evening', 'crows', 'dark cloud'}


def _direction(pattern: str) -> tuple[str, str]:
    low = pattern.lower()
    if any(k in low for k in _BULLISH_KEYWORDS):
        return "LONG 📈", "🟢"
    if any(k in low for k in _BEARISH_KEYWORDS):
        return "SHORT 📉", "🔴"
    return "NEUTRAL", "⚪"


class TelegramRouter:
    def __init__(self, token: str = TELEGRAM_BOT_TOKEN) -> None:
        self._bot = Bot(token=token)

    # ── public API ───────────────────────────────────────────────────────────

    async def send_signal(
        self,
        chat_id:          int,
        ticker:           str,
        tf_label:         str,
        candle:           Candle,
        pattern:          str,
        chart_path:       Optional[str] = None,
        message_thread_id: Optional[int] = None,   # ← тема внутри супергруппы
    ) -> None:
        direction, emoji = _direction(pattern)
        caption = (
            f"{emoji} <b>{ticker}</b> [{tf_label}]\n"
            f"<b>{pattern}</b>\n"
            f"\n"
            f"<b>Направление:</b> {direction}\n"
            f"<b>O:</b> {candle.open:.2f}  "
            f"<b>H:</b> {candle.high:.2f}  "
            f"<b>L:</b> {candle.low:.2f}  "
            f"<b>C:</b> {candle.close:.2f}\n"
            f"<b>Объём:</b> {candle.volume:,}\n"
            f"<i>🕐 {candle.open_time.strftime('%H:%M %d.%m.%Y')}</i>"
        )

        if chart_path and os.path.exists(chart_path):
            await self._send_photo(chat_id, chart_path, caption, message_thread_id)
        else:
            await self._send_text(chat_id, caption, message_thread_id)

    async def send_message(
        self,
        chat_id: int,
        text: str,
        message_thread_id: Optional[int] = None,
    ) -> None:
        await self._send_text(chat_id, text, message_thread_id)

    @property
    def bot(self) -> Bot:
        """Прямой доступ к Bot — нужен TopicManager."""
        return self._bot

    # ── private ──────────────────────────────────────────────────────────────

    async def _send_photo(
        self,
        chat_id: int,
        path: str,
        caption: str,
        thread_id: Optional[int],
    ) -> None:
        try:
            async with asyncio.timeout(TG_SEND_TIMEOUT):
                with open(path, "rb") as f:
                    await self._bot.send_photo(
                        chat_id            = chat_id,
                        photo              = f,
                        caption            = caption,
                        parse_mode         = ParseMode.HTML,
                        message_thread_id  = thread_id,
                    )
        except (TelegramError, TimeoutError) as exc:
            logger.error("send_photo → chat=%d thread=%s failed: %s",
                         chat_id, thread_id, exc)
            await self._send_text(chat_id, caption, thread_id)

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        thread_id: Optional[int],
    ) -> None:
        try:
            async with asyncio.timeout(TG_SEND_TIMEOUT):
                await self._bot.send_message(
                    chat_id            = chat_id,
                    text               = text,
                    parse_mode         = ParseMode.HTML,
                    message_thread_id  = thread_id,
                )
        except (TelegramError, TimeoutError) as exc:
            logger.error("send_message → chat=%d thread=%s failed: %s",
                         chat_id, thread_id, exc)
