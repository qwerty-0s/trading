import asyncio
import copy
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from moexalgo import Ticker

from morris_bot.config import ScannerConfig
from morris_bot.indicators import RSIIndicator, MACDIndicator, NoIndicator
from morris_bot.indicators.base import BaseIndicator
from morris_bot.indicators.dual import DualConfirmIndicator
from morris_bot.patterns.detector import PatternDetector
# filter_confirmed используется только в backtest, не в боте
from morris_bot.visualization.chart import ChartVisualizer
from morris_bot.bot.router import TelegramRouter

load_dotenv()


class MorrisBot:
    def __init__(self,
                 token: str = None,
                 chat_id: str = None,
                 output_dir: str = "/tmp"):
        token   = token   or os.getenv("TG_BOT_TOKEN", "")
        chat_id = chat_id or os.getenv("TG_GROUP_ID", "")

        self.router     = TelegramRouter(token, chat_id)
        self.detectors: Dict[str, PatternDetector] = {}
        # Лок нужен только для инициализации детектора — потом читаем без блокировки.
        self._detectors_lock = asyncio.Lock()
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Детекторы
    # ------------------------------------------------------------------

    async def get_detector(self, ticker: str, tf: str = '',
                           indicator: BaseIndicator = None) -> PatternDetector:
        """
        Создаёт детектор для пары ticker × tf.
        Ключ включает tf, чтобы два таймфрейма одного тикера
        не делили один объект индикатора (защита от Race Condition).

        Приоритет:
          1. indicator — явно переданный (из run() или вызова напрямую)
          2. Встроенные дефолты по тикеру (SBER → RSI, GAZP → MACD)
          3. NoIndicator — если ничего не задано
        """
        key = f"{ticker}_{tf}" if tf else ticker

        # Быстрый путь без лока — детектор уже создан
        if key in self.detectors:
            return self.detectors[key]

        # Медленный путь — создаём под локом, чтобы не дублировать объект
        async with self._detectors_lock:
            if key in self.detectors:          # повторная проверка после захвата лока
                return self.detectors[key]

            if indicator is not None:
                config = ScannerConfig(indicator=indicator)
            elif "SBER" in ticker:
                config = ScannerConfig(
                    long_body_coeff=1.4,
                    indicator=RSIIndicator(period=14, oversold=35, overbought=65)
                )
            elif "GAZP" in ticker:
                config = ScannerConfig(
                    indicator=MACDIndicator(fast=12, slow=26, signal=9)
                )
            else:
                config = ScannerConfig()       # NoIndicator по умолчанию

            self.detectors[key] = PatternDetector(config)

        return self.detectors[key]

    # ------------------------------------------------------------------
    # Форматирование
    # ------------------------------------------------------------------

    def _format_ind_str(self, detector: PatternDetector, df: pd.DataFrame, idx: int) -> str:
        """Форматирует строку со значением индикатора для Telegram-сообщения."""
        ind     = detector.config.indicator
        ind_col = ind.column_name

        if isinstance(ind, NoIndicator) or ind_col not in df.columns:
            return ""

        # DualConfirmIndicator хранит bit-encoded число — показываем sub-индикаторы
        if isinstance(ind, DualConfirmIndicator):
            parts = []
            for sub in (ind.ind1, ind.ind2):
                sub_col = sub.column_name
                if sub_col in df.columns:
                    v = df.loc[idx, sub_col]
                    if pd.notna(v):
                        parts.append(f"{sub.plot_label}: `{v:.1f}`")
            return ("\n📈 " + " | ".join(parts)) if parts else ""

        # Обычный индикатор
        v = df.loc[idx, ind_col]
        if pd.isna(v):
            return ""
        return f"\n📈 {ind.plot_label}: `{v:.1f}`"

    # ------------------------------------------------------------------
    # Подготовка данных (CPU-bound → выполняется в thread-pool)
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_df_sync(df: pd.DataFrame, detector: PatternDetector) -> pd.DataFrame:
        """
        Синхронная версия подготовки DataFrame.
        Вызывается через asyncio.to_thread, чтобы не блокировать event loop.
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['begin'])
        df['ema10']    = df['close'].ewm(span=10, adjust=False).mean()

        ind = detector.config.indicator
        df[ind.column_name] = ind.compute(df)

        if isinstance(ind, DualConfirmIndicator):
            df[ind.ind1.column_name] = ind.ind1.compute(df)
            df[ind.ind2.column_name] = ind.ind2.compute(df)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Таймауты опроса
    # ------------------------------------------------------------------

    TF_SLEEP: Dict[str, int] = {
        '1min':  30,
        '5min':  60,
        '15min': 60,
        '30min': 120,
        '1h':    300,
        '4h':    900,
        '1d':    3600,
    }

    # ------------------------------------------------------------------
    # Воркер (async-корутина вместо потока)
    # ------------------------------------------------------------------

    async def _worker(self, ticker: str, tf: str, indicator: BaseIndicator = None):
        """
        Асинхронный воркер для одной пары ticker × tf.

        Вся блокирующая работа (сеть, диск, pandas) уходит в asyncio.to_thread,
        поэтому event loop не замерзает и asyncio.sleep() точен до миллисекунд
        вместо неопределённых задержек от планировщика ОС у threading.
        """
        sleep_sec    = self.TF_SLEEP.get(tf, 60)
        last_signals: Dict[str, datetime] = {}

        print(f"[Worker] Запуск: {ticker} | {tf} | интервал {sleep_sec}с")

        while True:
            try:
                # fetch_data — сетевой вызов → thread-pool, event loop свободен
                df_raw = await asyncio.to_thread(self.fetch_data, ticker, tf)

                if df_raw.empty:
                    print(f"[Worker] Нет данных: {ticker} {tf}")
                else:
                    detector = await self.get_detector(ticker, tf, indicator)

                    # CPU-bound pandas → thread-pool
                    df = await asyncio.to_thread(
                        self._prepare_df_sync, df_raw, detector
                    )

                    # len(df)-1 может быть формирующейся свечой — берём len(df)-2
                    # как последнюю гарантированно закрытую.
                    # filter_confirmed НЕ используется в боте: он ждёт следующей
                    # закрытой свечи и добавляет 1 полный таймфрейм задержки (30+ мин).
                    # Фильтр индикатора уже встроен в detector через ScannerConfig.
                    pattern_idx = len(df) - 2
                    last_idx    = len(df) - 2

                    patterns = detector.get_pattern_at_index(df, pattern_idx)

                    # Обрабатываем все паттерны конкурентно
                    tasks = [
                        self._handle_pattern(
                            pattern, ticker, tf, df, pattern_idx, last_idx,
                            detector, last_signals
                        )
                        for pattern in patterns
                    ]
                    if tasks:
                        await asyncio.gather(*tasks)

            except asyncio.CancelledError:
                print(f"[Worker] Остановка: {ticker} {tf}")
                return
            except Exception as e:
                print(f"[Worker] Ошибка {ticker} {tf}: {e}")

            # asyncio.sleep точнее time.sleep — не зависит от планировщика ОС
            await asyncio.sleep(sleep_sec)

    async def _handle_pattern(
        self,
        pattern: str,
        ticker: str,
        tf: str,
        df: pd.DataFrame,
        pattern_idx: int,
        last_idx: int,
        detector: PatternDetector,
        last_signals: Dict[str, datetime],
    ):
        """
        Обрабатывает один паттерн: рисует график и отправляет сообщение.
        Вынесено отдельно, чтобы asyncio.gather мог запускать паттерны параллельно.
        """
        sig_key = f"{ticker}_{tf}_{pattern}"
        c_time  = df.loc[pattern_idx, 'datetime']

        if last_signals.get(sig_key) == c_time:
            return

        # Рендер графика — CPU + диск → thread-pool
        img_path = await asyncio.to_thread(
            ChartVisualizer.create_screenshot,
            df, ticker, tf, pattern, c_time,
            detector.config.indicator,
            output_dir=self.output_dir,
        )

        is_bullish      = any(x in pattern.lower() for x in
                              ['bull', 'hammer', 'morning', 'soldier', 'piercing'])
        direction_emoji = "🟢" if is_bullish else "🔴"
        ind_str         = self._format_ind_str(detector, df, last_idx)

        msg = (
            f"{direction_emoji} *{pattern}*\n"
            f"📊 `{ticker}` | `{tf}`\n"
            f"💰 Цена: `{df.loc[last_idx, 'close']}`"
            f"{ind_str}"
        )

        # Отправка в Telegram — сетевой вызов → thread-pool
        if img_path:
            await asyncio.to_thread(self.router.send_photo, ticker, tf, msg, img_path)
            os.remove(img_path)
        else:
            await asyncio.to_thread(self.router.send_message, ticker, tf, msg)

        last_signals[sig_key] = c_time

    # ------------------------------------------------------------------
    # Запуск
    # ------------------------------------------------------------------

    def run(self, config: Dict[str, List[str]],
            indicators: Dict[str, BaseIndicator] = None):
        """
        Запускает event loop и создаёт asyncio.Task для каждой пары ticker × tf.

        Args:
            config:     {ticker: [tf, ...]}
            indicators: {ticker: BaseIndicator} — опциональный индикатор на тикер.
                        Если не задан — используются встроенные дефолты (SBER→RSI, GAZP→MACD)
                        или NoIndicator для остальных.

        Примеры:
            # Дефолты — SBER получит RSI, остальные NoIndicator
            bot.run({'SBER': ['15min'], 'BRJ6': ['15min']})

            # Явные индикаторы
            from morris_bot.indicators import DualConfirmIndicator, BollingerPercentBIndicator, AdaptiveMFIIndicator
            dual = DualConfirmIndicator(BollingerPercentBIndicator(), AdaptiveMFIIndicator())
            bot.run(
                {'SiM6': ['15min'], 'BRJ6': ['15min'], 'NGJ6': ['15min']},
                indicators={
                    'SiM6': dual,
                    'BRJ6': dual,
                    'NGJ6': RSIIndicator(14),
                }
            )
        """
        asyncio.run(self._run_async(config, indicators))

    async def _run_async(self, config: Dict[str, List[str]],
                         indicators: Optional[Dict[str, BaseIndicator]] = None):
        """Внутренний async-метод: создаёт задачи и ждёт их."""
        indicators = indicators or {}
        tasks: List[asyncio.Task] = []

        for ticker, timeframes in config.items():
            for tf in timeframes:
                # Каждая задача получает глубокую копию индикатора,
                # чтобы избежать гонок при параллельном compute().
                # deepcopy корректно копирует DualConfirmIndicator с sub-объектами.
                ind_factory = indicators.get(ticker)
                ind = copy.deepcopy(ind_factory) if ind_factory else None

                task = asyncio.create_task(
                    self._worker(ticker, tf, ind),
                    name=f"{ticker}_{tf}",
                )
                tasks.append(task)

        total = len(tasks)
        print(f"[MorrisBot] Запущено {total} задач: "
              f"{', '.join(t.get_name() for t in tasks)}")

        try:
            # Ждём все задачи — они бесконечны, пока не придёт прерывание
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            # Корректная отмена всех задач при Ctrl+C
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print("[MorrisBot] Остановка по Ctrl+C")

    # ------------------------------------------------------------------
    # Получение данных (синхронный, вызывается через asyncio.to_thread)
    # ------------------------------------------------------------------

    def fetch_data(self, ticker: str, tf: str, days: int = 5) -> pd.DataFrame:
        try:
            t     = Ticker(ticker)
            start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            # moexalgo трактует end как исключающую правую границу,
            # +2 дня гарантирует включение сегодняшних свечей при любом UTC-сдвиге
            end   = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
            data  = t.candles(start=start, end=end, period=tf)
            df    = pd.DataFrame(data)
            if df.empty:
                print(f"[fetch_data] Пустой ответ для {ticker}")
                return df
            first = pd.to_datetime(df['begin'].min()).strftime('%Y-%m-%d %H:%M')
            last  = pd.to_datetime(df['begin'].max()).strftime('%Y-%m-%d %H:%M')
            print(f"[fetch_data] {ticker} {tf}: {len(df)} свечей  |  {first} → {last}")
            return df
        except Exception as e:
            print(f"[fetch_data] Ошибка загрузки {ticker}: {e}")
            return pd.DataFrame()