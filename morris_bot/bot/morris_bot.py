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
from morris_bot.patterns.detector import PatternDetector
from morris_bot.patterns.confirmation import filter_confirmed
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

    def get_detector(self, ticker: str) -> PatternDetector:
        """Создаёт детектор с индивидуальными настройками для тикера."""
        if ticker not in self.detectors:
            if "SBER" in ticker:
                config = ScannerConfig(
                    long_body_coeff=1.4,
                    indicator=RSIIndicator(period=14, oversold=35, overbought=65)
                )
            elif "GAZP" in ticker:
                config = ScannerConfig(
                    indicator=MACDIndicator(fast=12, slow=26, signal=9)
                )
            else:
                config = ScannerConfig()

            self.detectors[ticker] = PatternDetector(config)
        return self.detectors[ticker]

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

    def _worker(self, ticker: str, tf: str):
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
                    detector = self.get_detector(ticker)
                    df       = self._prepare_df(df_raw, detector)

                    # Ищем паттерны на предпоследней закрытой свече,
                    # чтобы последняя закрытая ([last_idx]) была подтверждающей.
                    pattern_idx = len(df) - 3
                    last_idx    = len(df) - 2

                    raw_patterns = detector.get_pattern_at_index(df, pattern_idx)

                    # filter_confirmed использует df.iloc[pattern_idx + 1] как confirm
                    patterns = filter_confirmed(raw_patterns, df, pattern_idx)

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

                        is_bullish = any(x in pattern.lower() for x in
                                         ['bull', 'hammer', 'morning', 'soldier', 'piercing'])
                        direction_emoji = "🟢" if is_bullish else "🔴"
                        ind_col   = detector.config.indicator.column_name
                        ind_value = df.loc[last_idx, ind_col] if ind_col in df.columns else None
                        ind_str   = (
                            f"\n📈 {detector.config.indicator.plot_label}: `{ind_value:.1f}`"
                            if ind_value and not pd.isna(ind_value) else ""
                        )

                        # Добавляем пометку ✅ если паттерн прошёл подтверждение
                        from morris_bot.patterns.confirmation import needs_confirmation
                        confirmed_mark = " ✅" if needs_confirmation(pattern) else ""

                        msg = (
                            f"{direction_emoji} *{pattern}*{confirmed_mark}\n"
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

    def run(self, config: Dict[str, List[str]]):
        """
        Запускает отдельный поток для каждой пары ticker × tf.

        Пример:
            bot.run({
                'SBER': ['15min', '1h'],
                'GAZP': ['15min', '1h'],
            })
        """
        threads = []
        for ticker, timeframes in config.items():
            for tf in timeframes:
                t = threading.Thread(
                    target=self._worker,
                    args=(ticker, tf),
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
