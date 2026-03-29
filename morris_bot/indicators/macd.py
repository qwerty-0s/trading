from typing import List
import pandas as pd

from .base import BaseIndicator


class MACDIndicator(BaseIndicator):
    """
    MACD (Moving Average Convergence Divergence).
    Бычий сигнал: гистограмма > 0. Медвежий: гистограмма < 0.
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
        return macd_line - signal_line  # гистограмма

    def confirms_bullish(self, value: float) -> bool:
        return value > 0

    def confirms_bearish(self, value: float) -> bool:
        return value < 0

    def get_level_lines(self) -> List[dict]:
        return [{"value": 0, "color": "gray", "dash": "dot"}]
