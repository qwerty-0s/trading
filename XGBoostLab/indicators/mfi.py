import numpy as np
import pandas as pd
from typing import List

from .base import BaseIndicator


class AdaptiveMFIIndicator(BaseIndicator):
    """
    Adaptive MFI с K-means кластеризацией — порт K-MEAN.pine (AlgoAlpha).

    Алгоритм
    ────────
    1. Вычисляется классический MFI(mfi_len) на hlc3 × volume.
    2. На скользящем окне training_size свечей запускается K-means (3 кластера):
       overbought (ob), neutral (ne), oversold (os).
       Начальные центроиды — фиксированные 80/50/20, затем сходятся за iterations шагов.
    3. Текущий MFI нормализуется: position = 100 × (mfi − os) / (ob − os).
       Диапазон 0..100 с adaptive уровнями (не фиксированными 80/20).

    Подтверждение:
      Бычий  ← position ≤ 20
      Медвежий ← position ≥ 80
    """

    def __init__(self,
                 mfi_len: int = 14,
                 training_size: int = 300,
                 iterations: int = 5,
                 init_ob: float = 80.0,
                 init_ne: float = 50.0,
                 init_os: float = 20.0,
                 bullish_threshold: float = 20.0,
                 bearish_threshold: float = 80.0):
        self.mfi_len = mfi_len
        self.training_size = training_size
        self.iterations = iterations
        self.init_ob = init_ob
        self.init_ne = init_ne
        self.init_os = init_os
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

    @property
    def column_name(self) -> str:
        return "adaptive_mfi"

    @property
    def plot_label(self) -> str:
        return f"AI MFI K-Means({self.mfi_len})"

    def _raw_mfi(self, df: pd.DataFrame) -> pd.Series:
        tp       = (df['high'] + df['low'] + df['close']) / 3
        raw_mf   = tp * df['volume']
        pos_flow = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_flow = raw_mf.where(tp < tp.shift(1), 0.0)
        pos_sum  = pos_flow.rolling(self.mfi_len).sum()
        neg_sum  = neg_flow.rolling(self.mfi_len).sum()
        return 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))

    def compute(self, df: pd.DataFrame) -> pd.Series:
        mfi    = self._raw_mfi(df)
        n      = len(mfi)
        result = pd.Series(np.nan, index=df.index)

        for i in range(self.mfi_len + 1, n):
            window = mfi.iloc[max(0, i - self.training_size):i + 1].dropna().values
            if len(window) < 10:
                continue

            a, b, c = self.init_ob, self.init_ne, self.init_os
            for _ in range(self.iterations):
                ob_v, ne_v, os_v = [], [], []
                for v in window:
                    d_a, d_b, d_c = abs(v - a), abs(v - b), abs(v - c)
                    if d_b < d_a and d_b < d_c:
                        ne_v.append(v)
                    elif d_a < d_b and d_a < d_c:
                        ob_v.append(v)
                    else:
                        os_v.append(v)
                if ob_v: a = float(np.mean(ob_v))
                if ne_v: b = float(np.mean(ne_v))
                if os_v: c = float(np.mean(os_v))

            cur = mfi.iloc[i]
            if not np.isnan(cur) and (a - c) != 0:
                result.iloc[i] = 100.0 * (cur - c) / (a - c)

        return result

    def confirms_bullish(self, value: float) -> bool:
        return value <= self.bullish_threshold

    def confirms_bearish(self, value: float) -> bool:
        return value >= self.bearish_threshold

    def get_level_lines(self) -> List[dict]:
        return [
            {"value": self.bearish_threshold, "color": "red",   "dash": "dash"},
            {"value": self.bullish_threshold, "color": "green", "dash": "dash"},
            {"value": 50, "color": "gray",  "dash": "dot"},
        ]
