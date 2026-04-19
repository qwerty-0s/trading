"""
AssetWorker
-----------
Один воркер на актив. Получает 1-мин свечи из asyncio.Queue,
агрегирует их, запускает детектор и отправляет сигнал в нужную
тему (message_thread_id) нужной супергруппы.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timezone
from typing import TYPE_CHECKING, List, Optional

from config import AssetConfig, CANDLE_HISTORY, TIMEFRAMES, ScannerConfig
from core.candle_aggregator import Candle, CandleAggregator
from patterns.detector import PatternDetector
from sending.telegram_router import TelegramRouter
from visualization.visualisation import ChartVisualizer

if TYPE_CHECKING:
    from core.topic_manager import TopicManager

logger = logging.getLogger(__name__)

_TF_LABEL = {
    10: "10min", 15: "15min", 30: "30min",
    60: "1h", 120: "2h", 180: "3h", 240: "4h",
}


class AssetWorker:
    def __init__(
        self,
        asset:          AssetConfig,
        router:         TelegramRouter,
        visualizer:     ChartVisualizer,
        scanner_config: ScannerConfig,
        topic_manager:  "TopicManager",
        timeframes:     List[int] = TIMEFRAMES,
    ) -> None:
        self.asset          = asset
        self.router         = router
        self.visualizer     = visualizer
        self.scanner_config = scanner_config
        self.topic_manager  = topic_manager
        self.timeframes     = timeframes

        self.queue: asyncio.Queue[Candle] = asyncio.Queue()

        self._aggregator = CandleAggregator(
            ticker       = asset.ticker,
            timeframes   = timeframes,
            history_size = CANDLE_HISTORY,
        )
        self._detectors = {
            tf: PatternDetector(scanner_config) for tf in timeframes
        }

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("[%s] worker started", self.asset.ticker)
        try:
            while True:
                candle_1m: Candle = await self.queue.get()
                await self._handle_1m(candle_1m)
        except asyncio.CancelledError:
            logger.info("[%s] worker stopped", self.asset.ticker)
            raise

    # ── internal ─────────────────────────────────────────────────────────────

    async def _handle_1m(self, candle_1m: Candle) -> None:
        closed = self._aggregator.push(candle_1m)
        if not closed:
            return
        await asyncio.gather(
            *[self._process_tf_candle(c) for c in closed],
            return_exceptions=True,
        )

    async def _process_tf_candle(self, candle: Candle) -> None:
        tf        = candle.timeframe
        tf_label  = _TF_LABEL.get(tf, f"{tf}m")
        detector  = self._detectors[tf]
        indicator = self.scanner_config.indicator

        # DataFrame + ema10 + индикатор
        try:
            df = await asyncio.to_thread(
                self._aggregator.to_dataframe, tf, indicator
            )
        except Exception as exc:
            logger.error("[%s %s] dataframe error: %s", self.asset.ticker, tf_label, exc)
            return

        if df.empty or len(df) < 13:
            return

        last_idx = len(df) - 1

        # Детектор паттернов
        try:
            patterns: list[str] = await asyncio.to_thread(
                detector.get_pattern_at_index, df, last_idx
            )
        except Exception as exc:
            logger.error("[%s %s] detector error: %s", self.asset.ticker, tf_label, exc)
            return

        if not patterns:
            return

        signal_time = (
            candle.open_time.replace(tzinfo=timezone.utc)
            if candle.open_time.tzinfo is None
            else candle.open_time
        )

        # Получаем thread_id темы для этого TF
        thread_id: Optional[int] = self.topic_manager.get_thread_id(
            self.asset.tg_chat_id, tf
        )

        logger.info(
            "[%s %s] patterns=%s thread_id=%s",
            self.asset.ticker, tf_label, patterns, thread_id,
        )

        await asyncio.gather(
            *[
                self._send_pattern(df, candle, tf_label, pattern,
                                   signal_time, indicator, thread_id)
                for pattern in patterns
            ],
            return_exceptions=True,
        )

    async def _send_pattern(
        self, df, candle, tf_label, pattern,
        signal_time, indicator, thread_id: Optional[int],
    ) -> None:
        # График
        try:
            chart_path = await asyncio.to_thread(
                self.visualizer.create_screenshot,
                df, self.asset.ticker, tf_label,
                pattern, signal_time, indicator,
            )
        except Exception as exc:
            logger.error("[%s %s] chart error: %s", self.asset.ticker, tf_label, exc)
            chart_path = None

        # Отправка в нужную тему
        try:
            await self.router.send_signal(
                chat_id           = self.asset.tg_chat_id,
                ticker            = self.asset.ticker,
                tf_label          = tf_label,
                candle            = candle,
                pattern           = pattern,
                chart_path        = chart_path,
                message_thread_id = thread_id,
            )
        except Exception as exc:
            logger.error("[%s %s] telegram error: %s", self.asset.ticker, tf_label, exc)
