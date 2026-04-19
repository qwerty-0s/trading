"""
CandleAggregator
----------------
• Принимает закрытые 1-минутные свечи из gRPC стрима
• Агрегирует их в N-минутные свечи для каждого TF
• Хранит скользящую историю и умеет отдавать её как DataFrame
  с колонками: datetime, open, high, low, close, volume, ema10
  — именно этот формат ожидает PatternDetector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """Внутреннее представление свечи (TF-агностик)."""
    ticker:    str
    timeframe: int       # минуты
    open_time: datetime
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    int

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


@dataclass
class _TFBucket:
    """Накапливает 1-мин свечи и испускает закрытую N-мин свечу."""
    tf_minutes: int
    _buf: List[Candle] = field(default_factory=list)
    _bucket_start: Optional[datetime] = None

    def _aligned_start(self, ts: datetime) -> datetime:
        epoch = int(ts.timestamp())
        step  = self.tf_minutes * 60
        return datetime.fromtimestamp((epoch // step) * step, tz=timezone.utc)

    def push(self, c1m: Candle) -> Optional[Candle]:
        bstart = self._aligned_start(c1m.open_time)
        closed: Optional[Candle] = None

        # Смена бакета — испустить предыдущий (если был заполнен)
        if self._bucket_start is not None and bstart != self._bucket_start:
            if len(self._buf) > 0:
                closed = self._build()
            self._buf.clear()

        if not self._buf:
            self._bucket_start = bstart
        self._buf.append(c1m)

        # Бакет заполнен точно по количеству 1-мин свечей
        if len(self._buf) >= self.tf_minutes:
            closed = self._build()
            self._buf.clear()
            self._bucket_start = None

        return closed

    def _build(self) -> Optional[Candle]:
        if not self._buf:
            return None
        return Candle(
            ticker    = self._buf[0].ticker,
            timeframe = self.tf_minutes,
            open_time = self._buf[0].open_time,
            open      = self._buf[0].open,
            high      = max(c.high   for c in self._buf),
            low       = min(c.low    for c in self._buf),
            close     = self._buf[-1].close,
            volume    = sum(c.volume for c in self._buf),
        )


class CandleAggregator:
    """
    Один агрегатор на актив.
    Хранит историю по каждому TF и умеет отдавать её как DataFrame
    с ema10 — готово для PatternDetector.
    """

    def __init__(self, ticker: str, timeframes: List[int], history_size: int = 200) -> None:
        self.ticker       = ticker
        self._history_sz  = history_size
        self._buckets:  Dict[int, _TFBucket]     = {tf: _TFBucket(tf) for tf in timeframes}
        self._history:  Dict[int, List[Candle]]  = {tf: []            for tf in timeframes}

    # ── public API ───────────────────────────────────────────────────────────

    def push(self, c1m: Candle) -> List[Candle]:
        """
        Принять закрытую 1-мин свечу.
        Вернуть список только что закрытых N-мин свечей (может быть пустым).
        """
        result: List[Candle] = []
        for tf, bucket in self._buckets.items():
            closed = bucket.push(c1m)
            if closed is not None:
                hist = self._history[tf]
                hist.append(closed)
                if len(hist) > self._history_sz:
                    del hist[0]
                result.append(closed)
                logger.debug("Closed %s %dm @ %s", self.ticker, tf,
                             closed.open_time.strftime("%H:%M"))
        return result

    def to_dataframe(self, tf: int, indicator=None) -> pd.DataFrame:
        """
        Собрать DataFrame из истории заданного TF.
        Добавляет колонки: datetime, ema10.
        Если передан indicator — вычисляет его колонку тоже.
        Возвращает DataFrame с integer-индексом (как ожидает детектор).
        """
        hist = self._history[tf]
        if not hist:
            return pd.DataFrame()

        df = pd.DataFrame({
            "datetime": [c.open_time for c in hist],
            "open":     [c.open      for c in hist],
            "high":     [c.high      for c in hist],
            "low":      [c.low       for c in hist],
            "close":    [c.close     for c in hist],
            "volume":   [c.volume    for c in hist],
        })

        # EMA10 — нужна детектору для определения тренда
        df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()

        # Дополнительный индикатор (RSI, MACD, …)
        if indicator is not None:
            df = indicator.compute(df)

        df.reset_index(drop=True, inplace=True)
        return df

    def history_len(self, tf: int) -> int:
        return len(self._history[tf])
