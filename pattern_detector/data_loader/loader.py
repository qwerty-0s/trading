"""
data_loader/loader.py
---------------------
Два компонента:

1. InstrumentResolver — при старте ищет актуальный FIGI для каждого фьюча
   через T-Invest InstrumentsService (futures меняют FIGI при экспирации).

2. StreamLoader — gRPC MarketDataStream, подписка на 1-мин закрытые свечи,
   роутинг в asyncio.Queue нужного AssetWorker.
   Автоматически переподключается при обрыве.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from t_tech.invest import (
    AsyncClient,
    CandleInstrument,
    MarketDataRequest,
    SubscribeCandlesRequest,
    SubscriptionAction,
    SubscriptionInterval,
)
from t_tech.invest.utils import quotation_to_decimal

from config import AssetConfig, STREAM_RECONNECT_DELAY, TINKOFF_TOKEN
from core.candle_aggregator import Candle
from core.asset_worker import AssetWorker

logger = logging.getLogger(__name__)

_1MIN = SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE


# ── FIGI Resolver ─────────────────────────────────────────────────────────────

class InstrumentResolver:
    """
    Находит актуальный FIGI для фьючерсного тикера.
    T-Invest возвращает все активные контракты по базовому тикеру
    (напр. "Si" → SiM5, SiH5, …). Берём ближайший по дате экспирации.
    """

    @staticmethod
    async def resolve_all(assets: List[AssetConfig]) -> None:
        """
        Заполняет поле .figi у каждого AssetConfig на месте (in-place).
        Вызывать один раз при старте.
        """
        async with AsyncClient(TINKOFF_TOKEN) as client:
            for asset in assets:
                try:
                    figi = await InstrumentResolver._find_figi(client, asset.ticker)
                    asset.figi = figi
                    logger.info("Resolved %s → %s", asset.ticker, figi)
                except Exception as exc:
                    logger.error("Cannot resolve FIGI for %s: %s", asset.ticker, exc)

    @staticmethod
    async def _find_figi(client, ticker: str) -> str:
        from t_tech.invest import InstrumentStatus
        resp = await client.instruments.find_instrument(query=ticker)
        instruments = [
            i for i in resp.instruments
            if i.ticker.upper().startswith(ticker.upper())
            and i.instrument_type == "futures"
        ]
        if not instruments:
            raise ValueError(f"No futures found for ticker '{ticker}'")

        # Среди найденных берём ближайший к экспирации (наименьший expiry_date > now)
        from datetime import datetime, timezone
        now = datetime.now(tz=timezone.utc)
        future_contracts = [
            i for i in instruments
            if hasattr(i, 'expiry_date') and i.expiry_date and i.expiry_date > now
        ]
        if future_contracts:
            nearest = min(future_contracts, key=lambda i: i.expiry_date)
        else:
            nearest = instruments[0]   # fallback

        return nearest.figi


# ── Stream Loader ─────────────────────────────────────────────────────────────

class StreamLoader:
    """
    Держит один gRPC stream на все активы.
    При разрыве — переподключается через STREAM_RECONNECT_DELAY секунд.
    """

    def __init__(
        self,
        assets:  List[AssetConfig],
        workers: Dict[str, AssetWorker],   # figi → worker
    ) -> None:
        self._assets  = assets
        self._workers = workers
        self._figi_to_ticker = {a.figi: a.ticker for a in assets}

    async def run_forever(self) -> None:
        while True:
            try:
                await self._stream_once()
            except asyncio.CancelledError:
                logger.info("StreamLoader: cancelled")
                raise
            except Exception as exc:
                logger.error(
                    "Stream error: %s — reconnecting in %ds …",
                    exc, STREAM_RECONNECT_DELAY,
                )
                await asyncio.sleep(STREAM_RECONNECT_DELAY)
                # Обновить маппинг на случай смены FIGI после экспирации
                self._figi_to_ticker = {a.figi: a.ticker for a in self._assets}

    async def _stream_once(self) -> None:
        figis = [a.figi for a in self._assets if a.figi]
        if not figis:
            raise RuntimeError("No FIGIs resolved — cannot subscribe")

        async with AsyncClient(TINKOFF_TOKEN) as client:
            logger.info(
                "gRPC connected — subscribing to %d instruments: %s",
                len(figis), [a.ticker for a in self._assets if a.figi],
            )

            subscribe_req = MarketDataRequest(
                subscribe_candles_request=SubscribeCandlesRequest(
                    subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                    instruments=[
                        CandleInstrument(figi=figi, interval=_1MIN)
                        for figi in figis
                    ],
                    waiting_close=True,   # ← только закрытые свечи
                )
            )

            async def _requests():
                yield subscribe_req
                await asyncio.Future()   # держим соединение открытым

            async for response in client.market_data_stream.market_data_stream(_requests()):
                await self._dispatch(response)

    async def _dispatch(self, response) -> None:
        ev = response.candle
        if ev is None:
            return

        worker = self._workers.get(ev.figi)
        if worker is None:
            return

        try:
            candle = Candle(
                ticker    = self._figi_to_ticker[ev.figi],
                timeframe = 1,
                open_time = ev.time,
                open      = float(quotation_to_decimal(ev.open)),
                high      = float(quotation_to_decimal(ev.high)),
                low       = float(quotation_to_decimal(ev.low)),
                close     = float(quotation_to_decimal(ev.close)),
                volume    = ev.volume,
            )
        except Exception as exc:
            logger.warning("Candle parse error: %s", exc)
            return

        await worker.queue.put(candle)
        logger.debug("→ %s 1m @ %s", candle.ticker, candle.open_time.strftime("%H:%M"))
