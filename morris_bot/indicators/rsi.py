from typing import List
import pandas as pd
import numpy as np

from .base import BaseIndicator


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
