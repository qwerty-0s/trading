import copy
import os
import time
import threading
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
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def get_detector(self, ticker: str, tf: str = '',
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
        if key not in self.detectors:
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
                config = ScannerConfig()  # NoIndicator по умолчанию

            self.detectors[key] = PatternDetector(config)
        return self.detectors[key]

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

    def _prepare_df(self, df: pd.DataFrame, detector: PatternDetector) -> pd.DataFrame:
        from morris_bot.indicators.dual import DualConfirmIndicator

        df = df.copy()
        df['datetime'] = pd.to_datetime(df['begin'])
        df['ema10']    = df['close'].ewm(span=10, adjust=False).mean()

        ind = detector.config.indicator
        df[ind.column_name] = ind.compute(df)

        # Для DualConfirmIndicator дополнительно записываем колонки sub-индикаторов
        # — нужны для панелей визуализации в visual_backtest_dual
        if isinstance(ind, DualConfirmIndicator):
            df[ind.ind1.column_name] = ind.ind1.compute(df)
            df[ind.ind2.column_name] = ind.ind2.compute(df)

        return df.reset_index(drop=True)

    TF_SLEEP: Dict[str, int] = {
        '1min':  30,
        '5min':  60,
        '15min': 60,
        '30min': 120,
        '1h':    300,
        '4h':    900,
        '1d':    3600,
    }


    def _worker(self, ticker: str, tf: str, indicator: BaseIndicator = None):
        """Воркер для одной пары ticker × tf. Работает в отдельном потоке."""
        sleep_sec    = self.TF_SLEEP.get(tf, 60)
        last_signals: Dict[str, datetime] = {}

        print(f"[Worker] Запуск: {ticker} | {tf} | интервал {sleep_sec}с")

        while True:
            try:
                df_raw = self.fetch_data(ticker, tf)
                if df_raw.empty:
                    print(f"[Worker] Нет данных: {ticker} {tf}")
                else:
                    detector = self.get_detector(ticker, tf, indicator)
                    df       = self._prepare_df(df_raw, detector)

                    # len(df)-1 может быть формирующейся свечой — берём len(df)-2
                    # как последнюю гарантированно закрытую.
                    # filter_confirmed НЕ используется в боте: он ждёт следующей
                    # закрытой свечи и добавляет 1 полный таймфрейм задержки (30+ мин).
                    # Фильтр индикатора уже встроен в detector через ScannerConfig.
                    pattern_idx = len(df) - 2
                    last_idx    = len(df) - 2

                    patterns = detector.get_pattern_at_index(df, pattern_idx)

                    for pattern in patterns:
                        sig_key = f"{ticker}_{tf}_{pattern}"
                        c_time  = df.loc[pattern_idx, 'datetime']

                        if last_signals.get(sig_key) == c_time:
                            continue

                        img_path = ChartVisualizer.create_screenshot(
                            df, ticker, tf, pattern, c_time,
                            detector.config.indicator,
                            output_dir=self.output_dir
                        )

                        is_bullish      = any(x in pattern.lower() for x in
                                              ['bull', 'hammer', 'morning', 'soldier', 'piercing'])
                        direction_emoji = "🟢" if is_bullish else "🔴"
                        ind_str = self._format_ind_str(detector, df, last_idx)

                        msg = (
                            f"{direction_emoji} *{pattern}*\n"
                            f"📊 `{ticker}` | `{tf}`\n"
                            f"💰 Цена: `{df.loc[last_idx, 'close']}`"
                            f"{ind_str}"
                        )

                        if img_path:
                            self.router.send_photo(ticker, tf, msg, img_path)
                            os.remove(img_path)
                        else:
                            self.router.send_message(ticker, tf, msg)

                        last_signals[sig_key] = c_time

            except Exception as e:
                print(f"[Worker] Ошибка {ticker} {tf}: {e}")

            time.sleep(sleep_sec)

    def run(self, config: Dict[str, List[str]],
            indicators: Dict[str, BaseIndicator] = None):
        """
        Запускает отдельный поток для каждой пары ticker × tf.

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
        indicators = indicators or {}
        threads = []
        for ticker, timeframes in config.items():
            for tf in timeframes:
                # Каждый поток получает глубокую копию индикатора
                # чтобы избежать Race Condition при параллельном compute().
                # deepcopy корректно копирует DualConfirmIndicator с sub-объектами.
                ind_factory = indicators.get(ticker)
                ind = copy.deepcopy(ind_factory) if ind_factory else None
                t = threading.Thread(
                    target=self._worker,
                    args=(ticker, tf, ind),
                    name=f"{ticker}_{tf}",
                    daemon=True
                )
                threads.append(t)
                t.start()

        total = len(threads)
        print(f"[MorrisBot] Запущено {total} потоков: "
              f"{', '.join(t.name for t in threads)}")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[MorrisBot] Остановка по Ctrl+C")

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