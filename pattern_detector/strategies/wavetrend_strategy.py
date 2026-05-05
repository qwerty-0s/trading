"""
wavetrend_strategy.py
=====================
Стратегия на основе WaveTrend осциллятора (порт TV Community Algo).

Логика входа:
  - WT1 пересекает WT2 снизу → BUY
  - WT1 пересекает WT2 сверху → SELL
  - ADX > adx_threshold (фильтр пилы)
  - +DI / -DI подтверждает направление (опционально)
  - Опциональный фильтр зон: вход только из перекупленности/перепроданности

Выход:
  - По противоположному сигналу (основной) + жёсткий ATR-стоп (защита)
  - Или только по ATR TP/SL (режим fixed_exit=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# Импортируй свой BaseStrategy из движка
from strategy_test import BaseStrategy, Signal, Position


# ==============================================================================
# ИНДИКАТОРЫ
# ==============================================================================

def wavetrend(df: pd.DataFrame, n1: int = 14, n2: int = 21) -> tuple[pd.Series, pd.Series]:
    """
    WaveTrend осциллятор. Точный порт из Pine Script.
    Возвращает (wt1, wt2).
    """
    ap  = (df['high'] + df['low'] + df['close']) / 3          # hlc3
    esa = ap.ewm(span=n1, adjust=False).mean()                 # EMA(ap, n1)
    d   = (ap - esa).abs().ewm(span=n1, adjust=False).mean()  # EMA(|ap - esa|, n1)

    # Защита от деления на ноль
    ci   = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1  = ci.ewm(span=n2, adjust=False).mean()               # EMA(ci, n2)
    wt2  = wt1.rolling(window=4).mean()                       # SMA(wt1, 4)

    return wt1, wt2


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX + DI+ + DI-.
    Возвращает (adx_line, plus_di, minus_di).
    """
    high, low, close = df['high'], df['low'], df['close']

    plus_dm  = high.diff()
    minus_dm = -low.diff()

    # Оставляем только "чистые" движения
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing (EWM с alpha = 1/period)
    atr_w     = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_w
    minus_di  = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_w

    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line  = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx_line, plus_di, minus_di


# ==============================================================================
# СТРАТЕГИЯ
# ==============================================================================

class WaveTrendStrategy(BaseStrategy):
    """
    WaveTrend crossover + ADX тренд-фильтр.

    Параметры:
      wt_n1, wt_n2        — периоды WaveTrend (14, 21 как в оригинале)
      adx_period          — период ADX (14)
      adx_threshold       — минимальный ADX для входа (22 — начало тренда)
      use_di_filter       — требовать совпадения +DI/-DI с направлением сигнала
      zone_filter         — входить только из зон перекупленности/перепроданности
      ob_level            — уровень перекупленности WT1 для шорта (60)
      os_level            — уровень перепроданности WT1 для лонга (-60)
      atr_sl_mult         — множитель ATR для стоп-лосса
      atr_tp_mult         — множитель ATR для тейк-профита (None = выход по сигналу)
      fixed_exit          — True: выходить только по ATR TP/SL
                            False: выходить по противоположному сигналу + ATR стоп
    """

    def __init__(self,
                 wt_n1: int = 14,
                 wt_n2: int = 21,
                 adx_period: int = 14,
                 adx_threshold: float = 22.0,
                 use_di_filter: bool = True,
                 zone_filter: bool = False,
                 ob_level: float = 60.0,
                 os_level: float = -60.0,
                 atr_period: int = 14,
                 atr_sl_mult: float = 1.5,
                 atr_tp_mult: Optional[float] = 3.0,
                 fixed_exit: bool = False):

        self.wt_n1          = wt_n1
        self.wt_n2          = wt_n2
        self.adx_period     = adx_period
        self.adx_threshold  = adx_threshold
        self.use_di_filter  = use_di_filter
        self.zone_filter    = zone_filter
        self.ob_level       = ob_level
        self.os_level       = os_level
        self.atr_period     = atr_period
        self.atr_sl_mult    = atr_sl_mult
        self.atr_tp_mult    = atr_tp_mult
        self.fixed_exit     = fixed_exit

    # ---------------------------------------------------------------- prepare
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # WaveTrend
        df['wt1'], df['wt2'] = wavetrend(df, self.wt_n1, self.wt_n2)

        # Кроссоверы (crossover: wt1 пересёк wt2 снизу)
        df['wt_cross_up']   = (df['wt1'] > df['wt2']) & (df['wt1'].shift() <= df['wt2'].shift())
        df['wt_cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift() >= df['wt2'].shift())

        # ADX + DI
        df['adx'], df['plus_di'], df['minus_di'] = adx(df, self.adx_period)

        # ATR
        hl  = df['high'] - df['low']
        hcp = (df['high'] - df['close'].shift()).abs()
        lcp = (df['low']  - df['close'].shift()).abs()
        df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(self.atr_period).mean()

        # Флаг дублирования сигналов (как в оригинале: не генерим BUY после BUY)
        df['_last_signal'] = self._mark_last_signal(df)

        return df

    @staticmethod
    def _mark_last_signal(df: pd.DataFrame) -> pd.Series:
        """
        Воспроизводит логику var sell/buy из Pine Script.
        Возвращает серию: 'buy', 'sell', или None для каждой свечи.
        """
        signals = pd.Series([None] * len(df), dtype=object)
        last = None
        for i in range(len(df)):
            cross_up   = df['wt_cross_up'].iloc[i]
            cross_down = df['wt_cross_down'].iloc[i]

            if cross_up and last != 'buy':
                signals.iloc[i] = 'buy'
                last = 'buy'
            elif cross_down and last != 'sell':
                signals.iloc[i] = 'sell'
                last = 'sell'
        return signals

    # ---------------------------------------------------------------- on_bar
    def on_bar(self,
               df: pd.DataFrame,
               i: int,
               htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:

        if i < max(self.adx_period, self.wt_n1 + self.wt_n2) + 5:
            return None

        row = df.iloc[i]

        # Базовые NaN-проверки
        if any(pd.isna(row.get(c, np.nan)) for c in ['wt1', 'wt2', 'adx', 'atr']):
            return None

        signal_dir = row['_last_signal']   # 'buy', 'sell', или None
        if signal_dir is None:
            return None

        # --- ADX фильтр ---
        if row['adx'] < self.adx_threshold:
            return None                    # рынок в боковике — не торгуем

        # --- DI фильтр (опционально) ---
        if self.use_di_filter:
            if signal_dir == 'buy'  and row['plus_di']  <= row['minus_di']:
                return None
            if signal_dir == 'sell' and row['minus_di'] <= row['plus_di']:
                return None

        # --- Зональный фильтр (опционально) ---
        # Входить только когда WT1 выходит из экстремальной зоны
        if self.zone_filter:
            prev_wt1 = df['wt1'].iloc[i - 1]
            if signal_dir == 'buy'  and prev_wt1 > self.os_level:
                return None   # не было перепроданности
            if signal_dir == 'sell' and prev_wt1 < self.ob_level:
                return None   # не было перекупленности

        # --- Формируем сигнал ---
        entry     = row['close']
        stop_dist = row['atr'] * self.atr_sl_mult
        tp_dist   = row['atr'] * self.atr_tp_mult if self.atr_tp_mult else stop_dist * 2

        if signal_dir == 'buy':
            return Signal(
                type='LONG',
                entry_price=entry,
                stop_loss=entry - stop_dist,
                take_profit=entry + tp_dist,
                pattern_name=f"WT Cross Up (ADX={row['adx']:.1f})",
                bar_index=i,
                bar_time=row['time'],
            )
        else:
            return Signal(
                type='SHORT',
                entry_price=entry,
                stop_loss=entry + stop_dist,
                take_profit=entry - tp_dist,
                pattern_name=f"WT Cross Down (ADX={row['adx']:.1f})",
                bar_index=i,
                bar_time=row['time'],
            )

    # ---------------------------------------------------------------- should_exit
    def should_exit(self, position: Position, row: pd.Series) -> bool:
        """
        fixed_exit=False: выход по ATR стоп-лоссу ИЛИ по противоположному сигналу.
        fixed_exit=True:  выход только по ATR SL/TP.
        """
        # Всегда проверяем ATR стоп
        sl_hit = (position.type == 'LONG'  and row['low']  <= position.sl) or \
                 (position.type == 'SHORT' and row['high'] >= position.sl)
        if sl_hit:
            return True

        if self.fixed_exit:
            # TP по ATR
            tp_hit = (position.type == 'LONG'  and row['high'] >= position.tp) or \
                     (position.type == 'SHORT' and row['low']  <= position.tp)
            return tp_hit
        else:
            # Выход по противоположному сигналу
            opposite = row.get('_last_signal')
            if position.type == 'LONG'  and opposite == 'sell':
                return True
            if position.type == 'SHORT' and opposite == 'buy':
                return True
            return False


# ==============================================================================
# БЫСТРЫЙ ТЕСТ (без Tinkoff API — на синтетических данных)
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Генерируем синтетический тренд для smoke-теста
    np.random.seed(42)
    n = 500
    price = 80000 + np.cumsum(np.random.randn(n) * 100)
    df_test = pd.DataFrame({
        'time':   pd.date_range("2026-01-01", periods=n, freq="15min"),
        'open':   price,
        'high':   price + np.abs(np.random.randn(n) * 50),
        'low':    price - np.abs(np.random.randn(n) * 50),
        'close':  price + np.random.randn(n) * 30,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    strategy = WaveTrendStrategy(
        adx_threshold=18.0,    # мягче для синтетики
        use_di_filter=True,
        zone_filter=False,
        atr_sl_mult=1.5,
        atr_tp_mult=None,      # выход по противоположному сигналу
        fixed_exit=False,
    )

    df_prep = strategy.prepare(df_test)

    signals_found = df_prep['_last_signal'].dropna()
    print(f"Smoke test: {len(signals_found)} сигналов на {n} свечах")
    print(f"  Buy:  {(signals_found == 'buy').sum()}")
    print(f"  Sell: {(signals_found == 'sell').sum()}")
    print(f"  ADX > 18: {(df_prep['adx'] > 18).sum()} свечей")
    print("✅ WaveTrendStrategy инициализирована корректно")
    print()
    print("Для полного теста используй:")
    print("  engine = BacktestEngine(strategy, initial_balance=100_000)")
    print("  portfolio = engine.run(df_real)")
    print("  engine.report()")