"""
data_loader/loader.py
---------------------
1. InstrumentResolver — резолвит FIGI фьючей при старте.
2. StreamLoader       — gRPC MarketDataStream (1min, waiting_close=True)
                        + prefill_history() через REST API.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from t_tech.invest import (
    AsyncClient,
    CandleInstrument,
    CandleInterval,
    MarketDataRequest,
    SubscribeCandlesRequest,
    SubscriptionAction,
    SubscriptionInterval,
)
from t_tech.invest.utils import quotation_to_decimal

from config import AssetConfig, CANDLE_HISTORY, STREAM_RECONNECT_DELAY, TIMEFRAMES, TINKOFF_TOKEN
from core.candle_aggregator import Candle
from core.asset_worker import AssetWorker

logger = logging.getLogger(__name__)

_1MIN = SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE


# ── FIGI Resolver ─────────────────────────────────────────────────────────────

class InstrumentResolver:
    @staticmethod
    async def resolve_all(assets: List[AssetConfig]) -> None:
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
        resp = await client.instruments.find_instrument(query=ticker)
        instruments = [
            i for i in resp.instruments
            if i.ticker.upper().startswith(ticker.upper())
            and i.instrument_type == "futures"
        ]
        if not instruments:
            raise ValueError(f"No futures found for ticker '{ticker}'")

        now = datetime.now(tz=timezone.utc)
        future_contracts = [
            i for i in instruments
            if hasattr(i, 'expiry_date') and i.expiry_date and i.expiry_date > now
        ]
        nearest = min(future_contracts, key=lambda i: i.expiry_date) \
                  if future_contracts else instruments[0]
        return nearest.figi


# ── Stream Loader ─────────────────────────────────────────────────────────────

class StreamLoader:
    def __init__(
        self,
        assets:     List[AssetConfig],
        workers:    Dict[str, AssetWorker],
        timeframes: List[int] = TIMEFRAMES,
    ) -> None:
        self._assets          = assets
        self._workers         = workers
        self._timeframes      = timeframes
        self._figi_to_ticker  = {a.figi: a.ticker for a in assets}

    # ── Prefill ───────────────────────────────────────────────────────────────

    async def prefill_history(self) -> None:
        max_tf       = max(self._timeframes)
        minutes_back = int(max_tf * CANDLE_HISTORY * 1.1)
        from_dt      = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)

        tf_coverage = {tf: minutes_back // tf for tf in self._timeframes}
        logger.info(
            "Prefilling history: last %d min (~%.1fh) per asset | expected candles: %s",
            minutes_back, minutes_back / 60, tf_coverage,
        )

        async with AsyncClient(TINKOFF_TOKEN) as client:
            for asset in self._assets:
                if not asset.figi:
                    continue
                worker = self._workers.get(asset.figi)
                if not worker:
                    continue
                await self._prefill_asset(client, asset, worker, from_dt)

        logger.info("Prefill complete — detectors ready")

    async def _prefill_asset(
        self,
        client,
        asset:   AssetConfig,
        worker:  AssetWorker,
        from_dt: datetime,
    ) -> None:
        try:
            all_candles = []
            chunk_start = from_dt
            now         = datetime.now(timezone.utc)

            while chunk_start < now:
                # Строгое ограничение T-Invest: не более 1 дня за запрос для 1min
                chunk_end = min(chunk_start + timedelta(hours=24), now)
                
                resp = await client.market_data.get_candles(
                    figi     = asset.figi,
                    from_    = chunk_start,
                    to       = chunk_end,
                    interval = CandleInterval.CANDLE_INTERVAL_1_MIN,
                )
                batch = [c for c in resp.candles if c.is_complete]
                all_candles.extend(batch)
                
                chunk_start = chunk_end
                # Обязательная пауза, чтобы не упереться в ratelimit при загрузке 36 дней х 6 активов
                await asyncio.sleep(0.15) 

            if not all_candles:
                logger.warning("[%s] prefill: no candles received", asset.ticker)
                return

            seen  = set()
            dedup = []
            for c in sorted(all_candles, key=lambda x: x.time):
                if c.time not in seen:
                    seen.add(c.time)
                    dedup.append(c)

            logger.info("[%s] prefill: %d raw 1min candles loaded", asset.ticker, len(dedup))

            for c in dedup:
                candle = Candle(
                    ticker    = asset.ticker,
                    timeframe = 1,
                    open_time = c.time,
                    open      = float(quotation_to_decimal(c.open)),
                    high      = float(quotation_to_decimal(c.high)),
                    low       = float(quotation_to_decimal(c.low)),
                    close     = float(quotation_to_decimal(c.close)),
                    volume    = c.volume,
                )
                worker._aggregator.push(candle)

            summary = {tf: worker._aggregator.history_len(tf) for tf in self._timeframes}
            logger.info("[%s] prefill done: %s", asset.ticker, summary)

        except Exception as exc:
            logger.error("[%s] prefill error: %s", asset.ticker, exc)

    # ── Stream ────────────────────────────────────────────────────────────────

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
                    waiting_close=True,
                )
            )

            async def _requests():
                yield subscribe_req
                await asyncio.Future()

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
