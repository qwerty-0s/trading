import numpy as np
import pandas as pd
from typing import List

from .base import BaseIndicator


class BollingerPercentBIndicator(BaseIndicator):
    """
    Bollinger %B — порт Void Lines из TV_ALGO (HomelessLemon).

        basis = SMA(close, period)
        upper = basis + mult × σ
        lower = basis − mult × σ
        %B    = (close − lower) / (upper − lower)

    Диапазон: 0 = нижняя полоса, 1 = верхняя, >1 / <0 = выход за полосу.

    Фильтр min_bandwidth_pct
    ────────────────────────
    В боковике BB сжимаются: (upper − lower) / basis < threshold.
    Экстремум %B в таком режиме — шум, а не перекупленность.
    Для таких свечей compute() возвращает NaN → PatternDetector не блокирует
    сигнал (NaN трактуется как «нет данных»), но и не использует ложный экстремум.
    """

    def __init__(self,
                 period: int = 20,
                 mult: float = 2.0,
                 oversold_band: float = 0.2,
                 overbought_band: float = 0.8,
                 min_bandwidth_pct: float = 0.01):
        self.period = period
        self.mult = mult
        self.oversold_band = oversold_band
        self.overbought_band = overbought_band
        self.min_bandwidth_pct = min_bandwidth_pct

    @property
    def column_name(self) -> str:
        return f"bb_pctB_{self.period}"

    @property
    def plot_label(self) -> str:
        return f"Bollinger %B ({self.period}, {self.mult}σ)"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        basis = df['close'].rolling(self.period).mean()
        std   = df['close'].rolling(self.period).std()
        upper = basis + self.mult * std
        lower = basis - self.mult * std
        band  = upper - lower

        too_narrow = band < (basis * self.min_bandwidth_pct)
        pct_b = (df['close'] - lower) / band.replace(0, np.nan)
        pct_b[too_narrow] = np.nan
        return pct_b

    def confirms_bullish(self, value: float) -> bool:
        return value <= self.oversold_band

    def confirms_bearish(self, value: float) -> bool:
        return value >= self.overbought_band

    def get_level_lines(self) -> List[dict]:
        return [
            {"value": self.overbought_band, "color": "red",   "dash": "dash"},
            {"value": self.oversold_band,   "color": "green", "dash": "dash"},
            {"value": 0.5,                  "color": "gray",  "dash": "dot"},
        ]
