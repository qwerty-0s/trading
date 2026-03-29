from typing import List
import pandas as pd
import numpy as np

from .base import BaseIndicator


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
