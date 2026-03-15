import os
import time
import threading
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from moexalgo import Ticker
from datetime import datetime, timedelta
import requests
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# ИНДИКАТОРЫ (паттерн Strategy)
# ==============================================================================

class BaseIndicator(ABC):
    """Абстрактный базовый класс для всех индикаторов-фильтров."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Вычислить значения индикатора по всему датафрейму."""
        pass

    @abstractmethod
    def confirms_bullish(self, value: float) -> bool:
        """Подтверждает ли значение индикатора бычий сигнал?"""
        pass

    @abstractmethod
    def confirms_bearish(self, value: float) -> bool:
        """Подтверждает ли значение индикатора медвежий сигнал?"""
        pass

    @property
    @abstractmethod
    def column_name(self) -> str:
        """Имя колонки, которую индикатор добавляет в df."""
        pass

    @property
    @abstractmethod
    def plot_label(self) -> str:
        """Подпись для оси Y на графике."""
        pass

    def get_level_lines(self) -> List[dict]:
        """
        Горизонтальные уровни для отрисовки на графике.
        Возвращает список dict: {"value": float, "color": str, "dash": str}
        """
        return []


class NoIndicator(BaseIndicator):
    """
    Индикатор-заглушка — отключает фильтрацию полностью.
    Используется для чистого теста паттернов без подтверждения.
    На графике ничего не рисует.
    """

    @property
    def column_name(self) -> str:
        return "_no_indicator"

    @property
    def plot_label(self) -> str:
        return ""

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([np.nan] * len(df), index=df.index)

    def confirms_bullish(self, value: float) -> bool:
        return True

    def confirms_bearish(self, value: float) -> bool:
        return True


class RSIIndicator(BaseIndicator):
    """
    RSI (Relative Strength Index).
    Подтверждает бычий сигнал при oversold, медвежий — при overbought.
    """

    def __init__(self,
                 period: int = 14,
                 oversold: float = 30.0,
                 overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def column_name(self) -> str:
        return f"rsi_{self.period}"

    @property
    def plot_label(self) -> str:
        return f"RSI({self.period})"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def confirms_bullish(self, value: float) -> bool:
        return value <= self.oversold

    def confirms_bearish(self, value: float) -> bool:
        return value >= self.overbought

    def get_level_lines(self) -> List[dict]:
        return [
            {"value": self.overbought, "color": "red",   "dash": "dash"},
            {"value": self.oversold,   "color": "green", "dash": "dash"},
            {"value": 50,              "color": "gray",  "dash": "dot"},
        ]


class MACDIndicator(BaseIndicator):
    """
    MACD (Moving Average Convergence Divergence).
    Бычий сигнал: гистограмма растёт и MACD-линия выше нуля или пересекает снизу.
    Медвежий сигнал: гистограмма падает и MACD-линия ниже нуля или пересекает сверху.
    """

    def __init__(self,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def column_name(self) -> str:
        return "macd_hist"

    @property
    def plot_label(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        # Сохраняем дополнительные колонки в df напрямую
        # (метод compute вернёт гистограмму как основную серию)
        return macd_line - signal_line  # Гистограмма

    def confirms_bullish(self, value: float) -> bool:
        return value > 0

    def confirms_bearish(self, value: float) -> bool:
        return value < 0

    def get_level_lines(self) -> List[dict]:
        return [{"value": 0, "color": "gray", "dash": "dot"}]


class StochasticIndicator(BaseIndicator):
    """
    Stochastic Oscillator (%K).
    Бычий: %K < oversold. Медвежий: %K > overbought.
    """

    def __init__(self,
                 k_period: int = 14,
                 d_period: int = 3,
                 oversold: float = 20.0,
                 overbought: float = 80.0):
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def column_name(self) -> str:
        return f"stoch_k_{self.k_period}"

    @property
    def plot_label(self) -> str:
        return f"Stoch({self.k_period},{self.d_period})"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        low_min  = df['low'].rolling(window=self.k_period).min()
        high_max = df['high'].rolling(window=self.k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
        return k

    def confirms_bullish(self, value: float) -> bool:
        return value <= self.oversold

    def confirms_bearish(self, value: float) -> bool:
        return value >= self.overbought

    def get_level_lines(self) -> List[dict]:
        return [
            {"value": self.overbought, "color": "red",   "dash": "dash"},
            {"value": self.oversold,   "color": "green", "dash": "dash"},
        ]


# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================

class ScannerConfig:
    def __init__(self,
                 long_body_coeff: float = None,
                 short_body_coeff: float = None,
                 shadow_limit: float = None,
                 indicator: BaseIndicator = None):
        self.long_body_coeff  = long_body_coeff  or float(os.getenv("DEFAULT_LONG_BODY_COEFF", 1.3))
        self.short_body_coeff = short_body_coeff or float(os.getenv("DEFAULT_SHORT_BODY_COEFF", 0.5))
        self.shadow_limit     = shadow_limit     or float(os.getenv("DEFAULT_SHADOW_LIMIT", 0.1))
        self.indicator: BaseIndicator = indicator or NoIndicator()


# ==============================================================================
# ДЕТЕКТОР ПАТТЕРНОВ
# ==============================================================================

class PatternDetector:
    def __init__(self, config: ScannerConfig):
        self.config = config
        self.avg_body = 0

    def _update_context(self, df: pd.DataFrame, idx: int):
        past_bodies = (df['close'].iloc[idx-10:idx] - df['open'].iloc[idx-10:idx]).abs()
        self.avg_body = past_bodies.mean()

    def is_long(self, body_size: float) -> bool:
        return body_size > (self.avg_body * self.config.long_body_coeff)

    def is_short(self, body_size: float) -> bool:
        return body_size < (self.avg_body * self.config.short_body_coeff)
    
    def is_dodji(self, candle):
        full_range = candle.high - candle.low
        body       = abs(candle.close - candle.open)
        if full_range == 0:
            return False
        # Тело ≤ 15% от полного диапазона — стандартный порог для дожи/крест харами
        return body <= (full_range * 0.15)

    def get_pattern_at_index(self, df: pd.DataFrame, idx: int) -> List[str]:
        if idx < 12:
            return []
        self._update_context(df, idx)

        c  = df.iloc[idx]
        p  = df.iloc[idx - 1]
        pp = df.iloc[idx - 2]

        c_body         = abs(c.close - c.open)
        c_range        = c.high - c.low
        c_top          = max(c.open, c.close)
        c_bottom       = min(c.open, c.close)
        c_upper_shadow = c.high - c_top
        c_lower_shadow = c_bottom - c.low

        p_body   = abs(p.close - p.open)
        p_top    = max(p.open, p.close)
        p_bottom = min(p.open, p.close)
        p_mid    = (p.open + p.close) / 2

        pp_body   = abs(pp.close - pp.open)
        pp_mid    = (pp.open + pp.close) / 2
        pp_top    = max(pp.open, pp.close)
        pp_bottom = min(pp.open, pp.close)

        c_is_white, c_is_black   = c.close > c.open, c.close < c.open
        p_is_white, p_is_black   = p.close > p.open, p.close < p.open
        pp_is_white, pp_is_black = pp.close > pp.open, pp.close < pp.open

        signals      = []
        ema          = c.ema10
        ind_col      = self.config.indicator.column_name
        ind_value    = c[ind_col] if ind_col in df.columns else None

        def bullish_confirmed() -> bool:
            if ind_value is None or np.isnan(ind_value):
                return True  # Нет данных — не блокируем
            return self.config.indicator.confirms_bullish(ind_value)

        def bearish_confirmed() -> bool:
            if ind_value is None or np.isnan(ind_value):
                return True
            return self.config.indicator.confirms_bearish(ind_value)

        # === ТРЕНД ВНИЗ — бычьи развороты ===
        if c.close < ema:

            if bullish_confirmed():

                if c_lower_shadow >= (c_body * 2) and c_upper_shadow <= (c_range * 0.1) and c_body > 0:
                    signals.append("Hammer (Молот)")

                if c_upper_shadow >= (c_body * 2) and c_lower_shadow <= (c_range * 0.1) and c_body > 0:
                    signals.append("Inverted Hammer (Перевернутый молот)")

                if c_is_white and p_is_black and c_top >= p_top and c_bottom <= p_bottom and p_body <= (c_body * 0.8):
                    signals.append("Bullish Engulfing (Бычье поглощение)")

                if c_is_white and p_is_black and c_body <= (p_body * 0.8) and c_top <= p_top and c_bottom >= p_bottom: 
                    signals.append("Bullish Harami (Бычье Харами)")
                
                if self.is_dodji(c) and p_is_black and p_body > self.avg_body and c_top <= p_top and c_bottom >= p_bottom: 
                    signals.append("Bullish Harami Cross (Бычий Крест Харами)")

                if self.is_long(p_body) and p_is_black and c_is_white:
                    if c.open < p.close and c.close > p_mid:
                        signals.append("Piercing Line (Просвет в облаках)")

                if pp_is_black and self.is_long(pp_body) and self.is_short(p_body):
                      if p_top <= pp_bottom and c_is_white and c.close >= pp_mid:
                        signals.append("Morning Star (Утренняя звезда)")

                if all([pp_is_white, p_is_white, c_is_white]):
                    if all([self.is_long(b) for b in [pp_body, p_body, c_body]]):
                        signals.append("Three White Soldiers (Три белых солдата)")

        # === ТРЕНД ВВЕРХ — медвежьи развороты ===
        elif c.close > ema:

            if bearish_confirmed():

                if c_lower_shadow >= (c_body * 2) and c_upper_shadow <= (c_range * 0.1) and c_body > 0:
                    signals.append("Hanging Man (Висельник)")

                if c_upper_shadow >= (c_body * 2) and c_lower_shadow <= (c_range * 0.1) and c_body > 0:
                    signals.append("Shooting Star (Падающая звезда)")

                if c_is_black and p_is_white and p_body <= (c_body * 0.8) and c_top >= p_top and c_bottom <= p_bottom: 
                    signals.append("Bearish Engulfing (Медвежье поглощение)")
                    
                if c_is_black and p_is_white and c_body <= (p_body * 0.8) and c_top <= p_top and c_bottom >= p_bottom: 
                    signals.append("Bearish Harami (Медвежье Харами)")
                
                if self.is_dodji(c) and p_is_white and p_body > self.avg_body and c_top <= p_top and c_bottom >= p_bottom: 
                    signals.append("Bearish Harami Cross (Медвежий Крест Харами)")

                if self.is_long(p_body) and p_is_white and c_is_black:
                    if c.open > p.close and c.close < p_mid:
                        signals.append("Dark Cloud Cover (Темные облака)")

                if pp_is_white and self.is_long(pp_body) and self.is_short(p_body):
                     if p_bottom >= pp_top and c_is_black and c.close < pp_mid:
                        signals.append("Evening Star (Вечерняя звезда)")

                if all([pp_is_black, p_is_black, c_is_black]):
                    if all([self.is_long(b) for b in [pp_body, p_body, c_body]]):
                        signals.append("Three Black Crows (Три черные вороны)")

        return signals


# ==============================================================================
# ВИЗУАЛИЗАЦИЯ
# ==============================================================================

class ChartVisualizer:

    @staticmethod
    def create_screenshot(df: pd.DataFrame,
                          ticker: str,
                          tf: str,
                          pattern: str,
                          signal_time: datetime,
                          indicator: BaseIndicator,
                          output_dir: str = "/tmp") -> Optional[str]:
        try:
            matches = df.index[df['datetime'] == signal_time].tolist()
            if not matches:
                return None
            idx      = matches[0]
            plot_df  = df.iloc[max(0, idx - 30):min(len(df), idx + 5)].copy()

            ind_col  = indicator.column_name
            has_ind  = ind_col in plot_df.columns and not isinstance(indicator, NoIndicator)

            # 2 строки если есть индикатор, иначе 1
            rows        = 2 if has_ind else 1
            row_heights = [0.7, 0.3] if has_ind else [1.0]

            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights,
                subplot_titles=[f"{ticker} {tf}", indicator.plot_label if has_ind else ""]
            )

            # --- Свечи ---
            fig.add_trace(go.Candlestick(
                x=plot_df['datetime'],
                open=plot_df['open'], high=plot_df['high'],
                low=plot_df['low'],   close=plot_df['close'],
                name=ticker,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            ), row=1, col=1)

            # --- EMA10 ---
            fig.add_trace(go.Scatter(
                x=plot_df['datetime'], y=plot_df['ema10'],
                line=dict(color='orange', width=1.5),
                name='EMA10'
            ), row=1, col=1)

            # --- Аннотация паттерна ---
            is_bullish = any(x in pattern.lower() for x in ['bull', 'hammer', 'morning', 'soldier', 'piercing'])
            color  = "#00e676" if is_bullish else "#ff1744"
            y_val  = plot_df.loc[idx, 'low'] if is_bullish else plot_df.loc[idx, 'high']
            ay_val = -40 if is_bullish else 40

            fig.add_annotation(
                x=signal_time, y=y_val,
                text=f"<b>{pattern}</b>",
                showarrow=True, arrowhead=2,
                arrowcolor=color, bgcolor=color,
                font=dict(color="black", size=10),
                ay=ay_val, row=1, col=1
            )

            # --- Индикатор (нижняя панель) ---
            if has_ind:
                ind_values = plot_df[ind_col]

                # Цвет баров для MACD-гистограммы, обычная линия для остальных
                if "macd" in ind_col:
                    bar_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in ind_values]
                    fig.add_trace(go.Bar(
                        x=plot_df['datetime'], y=ind_values,
                        marker_color=bar_colors, name=indicator.plot_label
                    ), row=2, col=1)
                else:
                    fig.add_trace(go.Scatter(
                        x=plot_df['datetime'], y=ind_values,
                        line=dict(color='#7c4dff', width=1.5),
                        name=indicator.plot_label
                    ), row=2, col=1)

                # Горизонтальные уровни (oversold/overbought и т.д.)
                for level in indicator.get_level_lines():
                    fig.add_hline(
                        y=level["value"],
                        line=dict(color=level["color"], dash=level["dash"], width=1),
                        row=2, col=1
                    )

            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                title=dict(text=f"{ticker} | {tf} | {pattern}", font=dict(size=14)),
                height=600,
                showlegend=False,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='#2a2a2a')

            # Уникальное имя файла — нет коллизий при параллельных экземплярах
            safe_pattern = pattern.replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(output_dir, f"alert_{ticker}_{tf}_{safe_pattern}.png")
            fig.write_image(path, scale=2)
            return path

        except Exception as e:
            print(f"[ChartVisualizer] Ошибка: {e}")
            return None


# ==============================================================================
# TELEGRAM ROUTER
# ==============================================================================

class TelegramRouter:
    """
    Определяет нужный message_thread_id по паре (ticker, timeframe).
    ID загружаются из .env по паттерну: TOPIC_{TICKER}_{TF}
    Пример: TOPIC_SBER_15MIN=123
    """

    def __init__(self, bot_token: str, group_id: str):
        self.bot_token = bot_token
        self.group_id  = group_id
        self._base_url = f"https://api.telegram.org/bot{bot_token}"
        self._topics: Dict[str, int] = self._load_topics()

    def _load_topics(self) -> Dict[str, int]:
        topics = {}
        for key, value in os.environ.items():
            if key.startswith("TOPIC_"):
                # TOPIC_SBER_15MIN -> "SBER_15MIN"
                route_key = key[len("TOPIC_"):]
                try:
                    topics[route_key] = int(value)
                except ValueError:
                    print(f"[TelegramRouter] Некорректный thread_id для {key}: {value}")
        return topics

    def _get_thread_id(self, ticker: str, tf: str) -> Optional[int]:
        # Нормализуем: "15min" -> "15MIN", "SBER" -> "SBER"
        key = f"{ticker.upper()}_{tf.upper().replace('MIN', 'MIN')}"
        thread_id = self._topics.get(key)
        if thread_id is None:
            print(f"[TelegramRouter] Тема не найдена для ключа: {key}. Отправляю в общий чат.")
        return thread_id

    def send_message(self, ticker: str, tf: str, text: str):
        thread_id = self._get_thread_id(ticker, tf)
        payload = {
            "chat_id":    self.group_id,
            "text":       text,
            "parse_mode": "Markdown",
        }
        if thread_id:
            payload["message_thread_id"] = thread_id

        try:
            resp = requests.post(f"{self._base_url}/sendMessage", data=payload, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[TelegramRouter] Ошибка sendMessage: {e}")

    def send_photo(self, ticker: str, tf: str, caption: str, img_path: str):
        thread_id = self._get_thread_id(ticker, tf)
        payload = {
            "chat_id":    self.group_id,
            "caption":    caption,
            "parse_mode": "Markdown",
        }
        if thread_id:
            payload["message_thread_id"] = thread_id

        try:
            with open(img_path, 'rb') as f:
                resp = requests.post(
                    f"{self._base_url}/sendPhoto",
                    data=payload,
                    files={"photo": f},
                    timeout=15
                )
                resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[TelegramRouter] Ошибка sendPhoto: {e}")
        except FileNotFoundError:
            print(f"[TelegramRouter] Файл не найден: {img_path}")


# ==============================================================================
# ГЛАВНЫЙ КЛАСС БОТА
# ==============================================================================

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
                config = ScannerConfig()  # Дефолт из .env

            self.detectors[ticker] = PatternDetector(config)
        return self.detectors[ticker]

    def _prepare_df(self, df: pd.DataFrame, detector: PatternDetector) -> pd.DataFrame:
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['begin'])
        df['ema10']    = df['close'].ewm(span=10, adjust=False).mean()

        # Вычисляем индикатор конкретного детектора
        ind = detector.config.indicator
        df[ind.column_name] = ind.compute(df)

        return df.reset_index(drop=True)  # Гарантируем целочисленный индекс

    # Интервал сна в секундах для каждого таймфрейма
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
                    last_idx = len(df) - 2  # Последняя закрытая свеча

                    patterns = detector.get_pattern_at_index(df, last_idx)

                    for pattern in patterns:
                        sig_key = f"{ticker}_{tf}_{pattern}"
                        c_time  = df.loc[last_idx, 'datetime']

                        if last_signals.get(sig_key) == c_time:
                            continue  # Уже отправляли этот сигнал

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
                        ind_str   = f"\n📈 {detector.config.indicator.plot_label}: `{ind_value:.1f}`" if ind_value else ""

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

    def run(self, config: Dict[str, List[str]]):
        """
        Запускает отдельный поток для каждой пары ticker × tf.

        Пример:
            bot.run({
                'SBER': ['15min', '1h'],
                'GAZP': ['15min', '1h'],
                'VTBR': ['15min'],
            })
        """
        threads = []
        for ticker, timeframes in config.items():
            for tf in timeframes:
                t = threading.Thread(
                    target=self._worker,
                    args=(ticker, tf),
                    name=f"{ticker}_{tf}",
                    daemon=True  # Поток завершится вместе с основным процессом
                )
                threads.append(t)
                t.start()

        total = len(threads)
        print(f"[MorrisBot] Запущено {total} потоков: "
              f"{', '.join(t.name for t in threads)}")

        # Держим главный поток живым
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[MorrisBot] Остановка по Ctrl+C")

    def fetch_data(self, ticker: str, tf: str, days: int = 5) -> pd.DataFrame:
        try:
            t     = Ticker(ticker)
            start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end   = datetime.now().strftime('%Y-%m-%d')
            data  = t.candles(start=start, end=end, period=tf)
            df    = pd.DataFrame(data)
            if df.empty:
                print(f"[fetch_data] Пустой ответ для {ticker}")
            return df
        except Exception as e:
            print(f"[fetch_data] Ошибка загрузки {ticker}: {e}")
            return pd.DataFrame()


# ==============================================================================
# БЭКТЕСТ
# ==============================================================================

def visual_backtest(ticker: str = 'SBER',
                    tf: str = '15min',
                    days_back: int = 10,
                    indicator: BaseIndicator = None):
    """
    Визуальный бэктест с отрисовкой паттернов, EMA10 и индикатора.
    """
    bot = MorrisBot("", "")

    # Можно передать кастомный индикатор для теста
    if indicator:
        bot.detectors[ticker] = PatternDetector(ScannerConfig(indicator=indicator))

    df_raw = bot.fetch_data(ticker, tf, days=days_back)
    if df_raw.empty:
        print("Нет данных для бэктеста.")
        return

    detector = bot.get_detector(ticker)
    df       = bot._prepare_df(df_raw, detector)
    ind      = detector.config.indicator
    ind_col  = ind.column_name
    has_ind  = ind_col in df.columns and not isinstance(ind, NoIndicator)

    rows        = 2 if has_ind else 1
    row_heights = [0.7, 0.3] if has_ind else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[f"Backtest {ticker} {tf}", ind.plot_label if has_ind else ""]
    )

    # Свечи
    fig.add_trace(go.Candlestick(
        x=df.datetime, open=df.open, high=df.high, low=df.low, close=df.close,
        name=ticker,
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
    ), row=1, col=1)

    # EMA10
    fig.add_trace(go.Scatter(
        x=df.datetime, y=df.ema10,
        line=dict(color='orange', width=1.5), name='EMA10'
    ), row=1, col=1)

    # Паттерны
    for i in range(12, len(df)):
        patterns = detector.get_pattern_at_index(df, i)
        for p in patterns:
            is_bullish = any(x in p.lower() for x in ['bull', 'hammer', 'morning', 'soldier', 'piercing'])
            color  = "lime" if is_bullish else "red"
            y_val  = df.loc[i, 'low'] if is_bullish else df.loc[i, 'high']
            ay_val = -30 if is_bullish else 30
            fig.add_annotation(
                x=df.loc[i, 'datetime'], y=y_val,
                text=p, showarrow=True, arrowhead=2,
                arrowcolor=color, bgcolor=color,
                font=dict(color="black", size=9),
                ay=ay_val, row=1, col=1
            )

    # Индикатор
    if has_ind:
        if "macd" in ind_col:
            bar_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df[ind_col]]
            fig.add_trace(go.Bar(
                x=df.datetime, y=df[ind_col],
                marker_color=bar_colors, name=ind.plot_label
            ), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=df.datetime, y=df[ind_col],
                line=dict(color='#7c4dff', width=1.5), name=ind.plot_label
            ), row=2, col=1)

        for level in ind.get_level_lines():
            fig.add_hline(
                y=level["value"],
                line=dict(color=level["color"], dash=level["dash"], width=1),
                row=2, col=1
            )

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        title=f"Backtest {ticker} | {tf}",
        height=700,
        showlegend=True,
    )
    fig.show()



def test_telegram(ticker: str, tf: str):
    """
    Отправляет тестовое сообщение в тему группы для пары ticker x tf.
    Используйте перед запуском бота чтобы убедиться что роутинг работает.

    Пример:
        test_telegram('SIM6', '15min')
    """
    token    = os.getenv("TG_BOT_TOKEN", "")
    group_id = os.getenv("TG_GROUP_ID", "")

    if not token or token == "your_bot_token_here":
        print("TG_BOT_TOKEN не задан в .env")
        return
    if not group_id or group_id == "-1001234567890":
        print("TG_GROUP_ID не задан в .env")
        return

    router    = TelegramRouter(token, group_id)
    key       = f"{ticker.upper()}_{tf.upper()}"
    thread_id = router._topics.get(key)

    print(f"[test_telegram] Тикер: {ticker} | ТФ: {tf} | Ключ: {key} | thread_id: {thread_id}")

    if thread_id is None:
        print(f"Ключ '{key}' не найден в .env")
        print(f"Нужна строка: TOPIC_{key}=<thread_id>")
        print(f"Доступные ключи в .env: {list(router._topics.keys())}")
        return

    msg = (
        f"*Тест подключения*\n"
        f"Тикер: `{ticker}` | ТФ: `{tf}`\n"
        f"Ключ: `{key}` thread\\_id: `{thread_id}`\n"
        f"Бот работает корректно"
    )
    router.send_message(ticker, tf, msg)
    print(f"Сообщение отправлено в тему thread_id={thread_id}")
# ==============================================================================
# ТОЧКА ВХОДА
# ==============================================================================

if __name__ == "__main__":

    # --- Бэктест БЕЗ индикатора (только паттерны + EMA10) ---
    #visual_backtest('CCH6', '30min', 5, indicator=NoIndicator())

    # --- Бэктест с RSI (по умолчанию если не передавать indicator) ---
    # visual_backtest('SBER', '15min', 10)

    # --- Бэктест с другим индикатором ---
    # visual_backtest('GAZP', '1h', 20, indicator=MACDIndicator())
    # visual_backtest('VTBR', '15min', 10, indicator=StochasticIndicator())
    
    #test_telegram('BRH6', '15min')

    bot = MorrisBot()
    bot.run({
        'SiM6': ['15min'],
        'BRJ6': ['15min'],
        'CCH6': ['15min'],
        'NGH6': ['15min'],
        'KCJ6': ['15min']
     })
     