import numpy as np
import pandas as pd
from typing import List

from .base import BaseIndicator


class DualConfirmIndicator(BaseIndicator):
    """
    Composite-индикатор: подтверждает сигнал только если ОБА внутренних
    индикатора согласны. Прозрачен для PatternDetector — ведёт себя как один.

    Архитектурное решение: bit-encoding
    ────────────────────────────────────
    Интерфейс BaseIndicator предполагает один float из compute().
    DualConfirmIndicator кодирует состояние обоих инд. в битовую маску:

        bit 0 (1):  ind1 подтверждает бычий
        bit 1 (2):  ind2 подтверждает бычий
        bit 2 (4):  ind1 подтверждает медвежий
        bit 3 (8):  ind2 подтверждает медвежий

        confirms_bullish → bits 0 и 1 → value & 3  == 3
        confirms_bearish → bits 2 и 3 → value & 12 == 12

    Визуализация
    ────────────
    visual_backtest_dual определяет DualConfirmIndicator через isinstance() и
    рисует отдельные панели для ind1 и ind2 (не бессмысленный bit-encoded график).
    prepare_df при обнаружении DualConfirmIndicator также записывает в df
    колонки sub-индикаторов для этих панелей.

    Пример
    ──────
        ind = DualConfirmIndicator(
            BollingerPercentBIndicator(period=20, mult=2.0),
            AdaptiveMFIIndicator(mfi_len=14),
        )
        config = ScannerConfig(indicator=ind)   # без изменений в PatternDetector
    """

    def __init__(self, ind1: BaseIndicator, ind2: BaseIndicator):
        self.ind1 = ind1
        self.ind2 = ind2

    @property
    def column_name(self) -> str:
        return f"dual__{self.ind1.column_name}__{self.ind2.column_name}"

    @property
    def plot_label(self) -> str:
        return f"{self.ind1.plot_label} ∩ {self.ind2.plot_label}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        s1 = self.ind1.compute(df)
        s2 = self.ind2.compute(df)

        valid = s1.notna() & s2.notna()

        b0 = s1.map(lambda v: int(self.ind1.confirms_bullish(v)) if pd.notna(v) else 0)
        b1 = s2.map(lambda v: int(self.ind2.confirms_bullish(v)) if pd.notna(v) else 0)
        b2 = s1.map(lambda v: int(self.ind1.confirms_bearish(v)) if pd.notna(v) else 0)
        b3 = s2.map(lambda v: int(self.ind2.confirms_bearish(v)) if pd.notna(v) else 0)

        encoded = (b0 + b1 * 2 + b2 * 4 + b3 * 8).astype(float)
        encoded[~valid] = np.nan
        return encoded

    def confirms_bullish(self, value: float) -> bool:
        if np.isnan(value):
            return True   # нет данных → не блокируем
        return (int(value) & 3) == 3

    def confirms_bearish(self, value: float) -> bool:
        if np.isnan(value):
            return True
        return (int(value) & 12) == 12

    def get_level_lines(self) -> List[dict]:
        return []   # визуализируем sub-панели, не bit-серию
