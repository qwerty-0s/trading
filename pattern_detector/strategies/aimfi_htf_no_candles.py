"""
strategies/aimfi_htf_no_candles.py
===================================
Стратегия разворота импульса на базе Adaptive MFI (K-Means) с фильтром HTF-тренда.
Не использует свечные паттерны (Pure AiMFI + HTF EMA).
"""

from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

# Импорт базовых классов и индикатора из структуры проекта
from indicators.mfi import AdaptiveMFIIndicator
from strategy_test import BaseStrategy, Signal


class AimfiHtfNoCandlesStrategy(BaseStrategy):
    """
    Стратегия входа по адаптивному MFI K-Means без свечных паттернов.

    Логика входа:
      - LONG:  AiMFI пересекает снизу вверх пороговое значение mfi_long_thr 
               (выход из зоны перепроданности) + HTF тренд бычий/нейтральный.
      - SHORT: AiMFI пересекает сверху вниз пороговое значение mfi_short_thr 
               (выход из зоны перегретости) + HTF тренд медвежий/нейтральный.
    """

    def __init__(
        self,
        atr_sl_mult: float = 1.5,
        rr_ratio: float = 2.0,
        mfi_period: int = 14,
        mfi_training_size: int = 300,
        mfi_long_thr: float = 40.0,
        mfi_short_thr: float = 60.0,
        use_htf_filter: bool = True
    ):
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.mfi_period = mfi_period
        self.mfi_training_size = mfi_training_size
        self.mfi_long_thr = mfi_long_thr
        self.mfi_short_thr = mfi_short_thr
        self.use_htf_filter = use_htf_filter

        # Инициализируем K-Means индикатор из indicators/mfi.py
        self.aimfi_indicator = AdaptiveMFIIndicator(
            mfi_len=self.mfi_period,
            training_size=self.mfi_training_size
        )

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка технических индикаторов на рабочем таймфрейме."""
        df = df.copy()

        # 1. Быстрая скользящая EMA10
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()

        # 2. Истинный средний диапазон ATR(14)
        hl = df['high'] - df['low']
        hcp = (df['high'] - df['close'].shift()).abs()
        lcp = (df['low'] - df['close'].shift()).abs()
        df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(14).mean()

        # 3. Адаптивный K-Means MFI из indicators/mfi.py
        df['aimfi'] = self.aimfi_indicator.compute(df)

        return df

    def on_bar(
        self,
        df: pd.DataFrame,
        i: int,
        htf_df: Optional[pd.DataFrame] = None
    ) -> Optional[Signal]:
        """Обработка каждого бара и генерация сигнала."""
        
        # Пропускаем прогревочные бары (K-Means требует историю)
        min_warmup = max(50, self.mfi_training_size)
        if i < min_warmup:
            return None

        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Проверка наличия всех индикаторов
        if (
            pd.isna(row.get('atr', np.nan)) or
            pd.isna(row.get('aimfi', np.nan)) or
            pd.isna(prev_row.get('aimfi', np.nan))
        ):
            return None

        # Определение тренда на старшем таймфрейме (HTF)
        htf_trend = self._get_htf_trend(htf_df, row['time']) if (self.use_htf_filter and htf_df is not None) else 0

        curr_mfi = row['aimfi']
        prev_mfi = prev_row['aimfi']

        # Триггеры пересечения уровней
        is_long_trigger = (prev_mfi < self.mfi_long_thr) and (curr_mfi >= self.mfi_long_thr)
        is_short_trigger = (prev_mfi > self.mfi_short_thr) and (curr_mfi <= self.mfi_short_thr)

        stop_dist = row['atr'] * self.atr_sl_mult
        tp_dist = stop_dist * self.rr_ratio

        # --- ВХОД В LONG ---
        if is_long_trigger:
            if not self.use_htf_filter or htf_trend >= 0:
                return Signal(
                    type="LONG",
                    entry_price=row['close'],
                    stop_loss=row['close'] - stop_dist,
                    take_profit=row['close'] + tp_dist,
                    pattern_name="AiMFI_Cross_Up",
                    bar_index=i,
                    bar_time=row['time'],
                )

        # --- ВХОД В SHORT ---
        if is_short_trigger:
            if not self.use_htf_filter or htf_trend <= 0:
                return Signal(
                    type="SHORT",
                    entry_price=row['close'],
                    stop_loss=row['close'] + stop_dist,
                    take_profit=row['close'] - tp_dist,
                    pattern_name="AiMFI_Cross_Down",
                    bar_index=i,
                    bar_time=row['time'],
                )

        return None

    @staticmethod
    def _get_htf_trend(htf_df: pd.DataFrame, current_time: datetime) -> int:
        """Возвращает направление тренда HTF без look-ahead bias (1: Up, -1: Down, 0: Neutral)."""
        time_col = 'close_time' if 'close_time' in htf_df.columns else 'time'
        subset = htf_df[htf_df[time_col] <= current_time]
        if subset.empty:
            return 0
        last = subset.iloc[-1]
        
        if 'ema10_htf' in last and not pd.isna(last['ema10_htf']):
            if last['close'] > last['ema10_htf']:
                return 1
            elif last['close'] < last['ema10_htf']:
                return -1
        return 0