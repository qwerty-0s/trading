"""
indicators/base.py
------------------
Минимальный интерфейс индикаторов.
Перенесён из morris_bot.indicators.base — сюда же добавляй свои RSI/MACD/etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseIndicator(ABC):
    """Базовый класс индикатора."""

    @property
    @abstractmethod
    def column_name(self) -> str:
        """Имя колонки в DataFrame (напр. 'rsi14', 'macd')."""

    @property
    @abstractmethod
    def plot_label(self) -> str:
        """Подпись на графике."""

    @abstractmethod
    def compute(self, df) -> "pd.DataFrame":
        """Добавить колонку индикатора в DataFrame и вернуть его."""

    def confirms_bullish(self, value: float) -> bool:
        """Индикатор подтверждает бычий разворот?"""
        return True

    def confirms_bearish(self, value: float) -> bool:
        """Индикатор подтверждает медвежий разворот?"""
        return True

    def get_level_lines(self) -> List[dict]:
        """Горизонтальные уровни для графика (напр. RSI 30/70)."""
        return []


class NoIndicator(BaseIndicator):
    """Заглушка — сигналы проходят без фильтра по индикатору."""

    @property
    def column_name(self) -> str:
        return "__no_indicator__"

    @property
    def plot_label(self) -> str:
        return ""

    def compute(self, df):
        return df

    def confirms_bullish(self, value: float) -> bool:
        return True

    def confirms_bearish(self, value: float) -> bool:
        return True


class RSIIndicator(BaseIndicator):
    """RSI с настраиваемым периодом и уровнями подтверждения."""

    def __init__(
        self,
        period: int = 14,
        bullish_below: float = 40.0,
        bearish_above: float = 60.0,
    ) -> None:
        self.period        = period
        self.bullish_below = bullish_below
        self.bearish_above = bearish_above

    @property
    def column_name(self) -> str:
        return f"rsi{self.period}"

    @property
    def plot_label(self) -> str:
        return f"RSI({self.period})"

    def compute(self, df):
        import pandas as pd
        delta = df["close"].diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()
        rs  = avg_gain / avg_loss.replace(0, float("nan"))
        df[self.column_name] = 100 - 100 / (1 + rs)
        return df

    def confirms_bullish(self, value: float) -> bool:
        return value <= self.bullish_below

    def confirms_bearish(self, value: float) -> bool:
        return value >= self.bearish_above

    def get_level_lines(self):
        return [
            {"value": 30,  "color": "#26a69a", "dash": "dash"},
            {"value": 70,  "color": "#ef5350",  "dash": "dash"},
            {"value": 50,  "color": "#888888",  "dash": "dot"},
        ]


class MACDIndicator(BaseIndicator):
    """MACD-гистограмма."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    @property
    def column_name(self) -> str:
        return "macd_hist"

    @property
    def plot_label(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"

    def compute(self, df):
        ema_fast   = df["close"].ewm(span=self.fast,   adjust=False).mean()
        ema_slow   = df["close"].ewm(span=self.slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_ln  = macd_line.ewm(span=self.signal,   adjust=False).mean()
        df[self.column_name] = macd_line - signal_ln
        return df

    def confirms_bullish(self, value: float) -> bool:
        return value > 0

    def confirms_bearish(self, value: float) -> bool:
        return value < 0

    def get_level_lines(self):
        return [{"value": 0, "color": "#888888", "dash": "dot"}]
