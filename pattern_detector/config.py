"""
config.py — центральная конфигурация
FIGIs для фьючей резолвятся динамически при старте через T-Invest Instruments API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ── T-Invest ──────────────────────────────────────────────────────────────────
TINKOFF_TOKEN: str = os.environ["TINKOFF_TOKEN"]

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]

# ── Таймфреймы (минуты) ───────────────────────────────────────────────────────
TIMEFRAMES: List[int] = [10, 15, 30, 60, 120, 180, 240]

# Глубина истории для детектора
CANDLE_HISTORY: int = 200

# ── Параметры детектора ───────────────────────────────────────────────────────
@dataclass
class ScannerConfig:
    long_body_coeff:  float = 1.5
    short_body_coeff: float = 0.5
    # Индикатор — меняй здесь: NoIndicator / RSIIndicator(14) / MACDIndicator()
    indicator: object = field(default_factory=lambda: _default_indicator())

def _default_indicator():
    from indicators.base import NoIndicator
    return NoIndicator()

# ── Активы (фьючи MOEX) ───────────────────────────────────────────────────────
@dataclass
class AssetConfig:
    ticker: str       # тикер как в T-Invest, напр. "SiM4", "BRM4"
    tg_chat_id: int
    figi: str = ""    # заполняется автоматически InstrumentResolver при старте

ASSETS: List[AssetConfig] = [
    AssetConfig(ticker="SiM6",  tg_chat_id=-1003916417055),
    AssetConfig(ticker="KCJ6",  tg_chat_id=-1003716674818),
    AssetConfig(ticker="BRK6",  tg_chat_id=-1003750564099),
    AssetConfig(ticker="CCJ6",  tg_chat_id=-1003944653501),
    AssetConfig(ticker="SVM6",  tg_chat_id=-1003994847530),
    AssetConfig(ticker="GDM6",  tg_chat_id=-1003949836612),
]

# ── Misc ──────────────────────────────────────────────────────────────────────
STREAM_RECONNECT_DELAY: int = 5
TG_SEND_TIMEOUT: int        = 15
SCREENSHOT_DIR: str         = "/tmp/trade_charts"
