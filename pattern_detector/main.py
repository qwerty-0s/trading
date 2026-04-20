"""
main.py — точка входа
1. Резолвим FIGI фьючей через T-Invest API
2. Создаём / проверяем темы в Telegram-супергруппах (TopicManager)
3. Создаём воркеры
4. Prefill истории через REST API → детектор готов сразу
5. Запускаем gRPC стрим
6. Graceful shutdown по SIGINT/SIGTERM
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

from config import ASSETS, TIMEFRAMES, ScannerConfig
from core.asset_worker import AssetWorker
from core.topic_manager import TopicManager
from data_loader.loader import InstrumentResolver, StreamLoader
from sending.telegram_router import TelegramRouter
from visualization.visualisation import ChartVisualizer

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
logger = logging.getLogger(__name__)


async def _run() -> None:
    # ── 1. Резолвим FIGI ──────────────────────────────────────────────────────
    logger.info("Resolving FIGIs for %d assets …", len(ASSETS))
    await InstrumentResolver.resolve_all(ASSETS)

    active_assets = [a for a in ASSETS if a.figi]
    skipped = [a.ticker for a in ASSETS if not a.figi]
    if skipped:
        logger.warning("Could not resolve FIGIs for: %s — skipped", skipped)
    if not active_assets:
        logger.error("No active assets — exiting")
        return

    # ── 2. Shared зависимости ─────────────────────────────────────────────────
    router         = TelegramRouter()
    visualizer     = ChartVisualizer()
    scanner_config = ScannerConfig()

    # ── 3. Создаём / проверяем темы в супергруппах ───────────────────────────
    topic_manager = TopicManager(bot=router.bot, timeframes=TIMEFRAMES)
    await topic_manager.setup(active_assets)

    # ── 4. Воркеры ────────────────────────────────────────────────────────────
    workers_by_figi: dict[str, AssetWorker] = {}
    for asset in active_assets:
        workers_by_figi[asset.figi] = AssetWorker(
            asset          = asset,
            router         = router,
            visualizer     = visualizer,
            scanner_config = scanner_config,
            topic_manager  = topic_manager,
            timeframes     = TIMEFRAMES,
        )

    # ── 5. Prefill истории ────────────────────────────────────────────────────
    loader = StreamLoader(
        assets     = active_assets,
        workers    = workers_by_figi,
        timeframes = TIMEFRAMES,
    )
    await loader.prefill_history()

    # ── 6. Запускаем воркеры и стрим ─────────────────────────────────────────
    worker_tasks = [
        asyncio.create_task(w.run(), name=f"worker:{a.ticker}")
        for a, w in zip(active_assets, workers_by_figi.values())
    ]
    stream_task = asyncio.create_task(loader.run_forever(), name="stream")
    all_tasks   = worker_tasks + [stream_task]

    logger.info(
        "✅ Running: %s | TFs: %s",
        [a.ticker for a in active_assets],
        TIMEFRAMES,
    )

    # ── 7. Ждём отмены ───────────────────────────────────────────────────────
    try:
        await asyncio.gather(*all_tasks)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down …")
        for t in all_tasks:
            t.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        logger.info("Stopped.")


def _shutdown(loop: asyncio.AbstractEventLoop) -> None:
    logger.info("Stop signal received")
    for task in asyncio.all_tasks(loop):
        task.cancel()


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
