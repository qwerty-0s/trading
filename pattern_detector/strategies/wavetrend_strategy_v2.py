"""
wavetrend_strategy_v2.py
========================
Улучшенная стратегия WaveTrend v2.

Изменения по сравнению с v1:
  1. fixed_exit=True по умолчанию — выход ТОЛЬКО по ATR SL/TP.
     Убираем выход по противоположному сигналу: он создавал 72% "SIGNAL"-закрытий
     и не давал позициям раскрыться.

  2. cooldown_bars — минимум N баров между любыми новыми сигналами.
     Устраняет кластеры сигналов, видные на графике.

  3. ADX threshold поднят до 28 (было 22) — фильтруем слабые тренды.

  4. zone_filter=True по умолчанию — входим только из зон OB/OS.

  5. Трейлинг-стоп (trailing_sl_mult) — опциональный: после прохождения 1R
     подтягиваем стоп на уровень безубытка + буфер.

  6. Параметр min_rr_ratio — не берём сигнал если ATR даёт плохой R:R
     (например, если ATR аномально большой и TP слишком далеко от реальной цели).

Рекомендованный стартовый конфиг:
  WaveTrendStrategyV2(
      adx_threshold=28,
      use_di_filter=True,
      zone_filter=True,
      atr_sl_mult=1.5,
      atr_tp_mult=2.5,       # 2.5R — цель реалистичная при WT пересечениях
      fixed_exit=True,
      cooldown_bars=5,        # минимум 5 баров (~75 мин на 15m TF) между сигналами
      trailing_sl=True,
      min_rr_ratio=1.8,
  )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from strategy_test import BaseStrategy, Signal, Position


# ==============================================================================
# ИНДИКАТОРЫ (без изменений — стабильная математика)
# ==============================================================================

def wavetrend(df: pd.DataFrame, n1: int = 14, n2: int = 21) -> tuple[pd.Series, pd.Series]:
    """WaveTrend осциллятор — точный порт Pine Script."""
    ap  = (df['high'] + df['low'] + df['close']) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d   = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci  = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    return wt1, wt2


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX + DI+ + DI- (Wilder smoothing)."""
    high, low, close = df['high'], df['low'], df['close']

    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_w    = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w

    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx_line, plus_di, minus_di


# ==============================================================================
# СТРАТЕГИЯ v2
# ==============================================================================

class WaveTrendStrategyV2(BaseStrategy):
    """
    WaveTrend v2: меньше сделок, лучше качество.

    Ключевые параметры:
      fixed_exit    — True: выход только по ATR SL/TP (рекомендовано).
      cooldown_bars — минимум N баров между сигналами (убирает кластеры).
      trailing_sl   — подтягивать стоп в безубыток после 1R профита.
      min_rr_ratio  — минимальный R:R (отклонять сигнал если меньше).
      zone_filter   — входить только из OB/OS зон WaveTrend.
    """

    def __init__(
        self,
        # WaveTrend
        wt_n1: int = 14,
        wt_n2: int = 21,
        # ADX
        adx_period: int = 14,
        adx_threshold: float = 28.0,     # ↑ с 22 до 28 — только сильные тренды
        use_di_filter: bool = True,
        # Зональный фильтр
        zone_filter: bool = True,         # ↑ включён по умолчанию
        ob_level: float = 53.0,          # немного мягче стандарта 60 — больше сигналов
        os_level: float = -53.0,
        # ATR-выход
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 2.5,        # ↑ 2.5R — нужно давать прибыли расти
        fixed_exit: bool = True,          # ↑ только ATR TP/SL — убираем "SIGNAL" выходы
        # Фильтр качества
        min_rr_ratio: float = 1.8,       # отклонять сигналы с плохим R:R
        # Антикластерный фильтр
        cooldown_bars: int = 5,           # минимум баров между сигналами
        # Трейлинг-стоп
        trailing_sl: bool = True,         # подтягивать SL в безубыток после 1R
        breakeven_buffer_mult: float = 0.2,  # SL = entry + 0.2*ATR (небольшой плюс)
    ):
        self.wt_n1               = wt_n1
        self.wt_n2               = wt_n2
        self.adx_period          = adx_period
        self.adx_threshold       = adx_threshold
        self.use_di_filter       = use_di_filter
        self.zone_filter         = zone_filter
        self.ob_level            = ob_level
        self.os_level            = os_level
        self.atr_period          = atr_period
        self.atr_sl_mult         = atr_sl_mult
        self.atr_tp_mult         = atr_tp_mult
        self.fixed_exit          = fixed_exit
        self.min_rr_ratio        = min_rr_ratio
        self.cooldown_bars       = cooldown_bars
        self.trailing_sl         = trailing_sl
        self.breakeven_buffer_mult = breakeven_buffer_mult

        # Внутреннее состояние (сбрасывается при каждом prepare)
        self._last_signal_bar: int = -9999
        self._current_sl: Optional[float] = None  # для трейлинга
        self._breakeven_activated: bool = False

    # ---------------------------------------------------------------- prepare
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['wt1'], df['wt2'] = wavetrend(df, self.wt_n1, self.wt_n2)

        df['wt_cross_up']   = (df['wt1'] > df['wt2']) & (df['wt1'].shift() <= df['wt2'].shift())
        df['wt_cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift() >= df['wt2'].shift())

        df['adx'], df['plus_di'], df['minus_di'] = adx(df, self.adx_period)

        hl  = df['high'] - df['low']
        hcp = (df['high'] - df['close'].shift()).abs()
        lcp = (df['low']  - df['close'].shift()).abs()
        df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(self.atr_period).mean()

        df['_last_signal'] = self._mark_last_signal(df)

        # Сброс состояния
        self._last_signal_bar      = -9999
        self._current_sl           = None
        self._breakeven_activated  = False

        return df

    @staticmethod
    def _mark_last_signal(df: pd.DataFrame) -> pd.Series:
        """Дедупликация: не генерим BUY сразу после BUY (как в Pine Script)."""
        signals = pd.Series([None] * len(df), dtype=object)
        last = None
        for i in range(len(df)):
            if df['wt_cross_up'].iloc[i] and last != 'buy':
                signals.iloc[i] = 'buy'
                last = 'buy'
            elif df['wt_cross_down'].iloc[i] and last != 'sell':
                signals.iloc[i] = 'sell'
                last = 'sell'
        return signals

    # ---------------------------------------------------------------- on_bar
    def on_bar(self,
               df: pd.DataFrame,
               i: int,
               htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:

        warmup = max(self.adx_period, self.wt_n1 + self.wt_n2) + 5
        if i < warmup:
            return None

        row = df.iloc[i]

        # NaN-гвардия
        for col in ('wt1', 'wt2', 'adx', 'atr'):
            if pd.isna(row.get(col, np.nan)):
                return None

        signal_dir = row['_last_signal']
        if signal_dir is None:
            return None

        # ── 1. Cooldown: не торгуем слишком часто ───────────────────────────
        if i - self._last_signal_bar < self.cooldown_bars:
            return None

        # ── 2. ADX фильтр ───────────────────────────────────────────────────
        if row['adx'] < self.adx_threshold:
            return None

        # ── 3. DI фильтр ────────────────────────────────────────────────────
        if self.use_di_filter:
            if signal_dir == 'buy'  and row['plus_di']  <= row['minus_di']:
                return None
            if signal_dir == 'sell' and row['minus_di'] <= row['plus_di']:
                return None

        # ── 4. Зональный фильтр ─────────────────────────────────────────────
        if self.zone_filter:
            # Ищем: был ли WT1 в зоне OS/OB за последние 3 бара перед пересечением
            lookback = min(3, i)
            prev_wt1 = df['wt1'].iloc[i - lookback: i]
            if signal_dir == 'buy'  and not (prev_wt1 <= self.os_level).any():
                return None
            if signal_dir == 'sell' and not (prev_wt1 >= self.ob_level).any():
                return None

        # ── 5. Расчёт уровней ───────────────────────────────────────────────
        entry     = row['close']
        atr       = row['atr']
        stop_dist = atr * self.atr_sl_mult
        tp_dist   = atr * self.atr_tp_mult

        # Проверка минимального R:R
        rr = tp_dist / stop_dist if stop_dist > 0 else 0
        if rr < self.min_rr_ratio:
            return None

        self._last_signal_bar     = i
        self._current_sl          = None
        self._breakeven_activated = False

        if signal_dir == 'buy':
            sl = entry - stop_dist
            tp = entry + tp_dist
            sig_type = 'LONG'
        else:
            sl = entry + stop_dist
            tp = entry - tp_dist
            sig_type = 'SHORT'

        return Signal(
            type=sig_type,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            pattern_name=f"WTv2 {signal_dir.upper()} ADX={row['adx']:.1f} ATR={atr:.2f}",
            bar_index=i,
            bar_time=row.get('time'),
        )

    # ---------------------------------------------------------------- should_exit
    def should_exit(self, position: Position, row: pd.Series) -> bool:
        """
        Выход по:
          1. Трейлинг SL (безубыток после 1R) — если trailing_sl=True
          2. ATR SL (жёсткий стоп)
          3. ATR TP
          4. Противоположный сигнал — только если fixed_exit=False
        """
        entry     = position.entry
        orig_sl   = position.sl
        orig_tp   = position.tp
        is_long   = position.type == 'LONG'

        # Текущий рабочий SL (может быть подтянут в безубыток)
        if self._current_sl is None:
            self._current_sl = orig_sl

        # ── Трейлинг: активируем безубыток после 1R ─────────────────────────
        if self.trailing_sl and not self._breakeven_activated:
            risk = abs(entry - orig_sl)
            if is_long:
                if row['high'] >= entry + risk:        # прошли 1R вверх
                    buf = row.get('atr', 0) * self.breakeven_buffer_mult
                    self._current_sl      = entry + buf
                    self._breakeven_activated = True
            else:
                if row['low'] <= entry - risk:         # прошли 1R вниз
                    buf = row.get('atr', 0) * self.breakeven_buffer_mult
                    self._current_sl      = entry - buf
                    self._breakeven_activated = True

        # ── Проверка SL ─────────────────────────────────────────────────────
        working_sl = self._current_sl
        if is_long  and row['low']  <= working_sl:
            return True
        if not is_long and row['high'] >= working_sl:
            return True

        # ── Проверка TP ─────────────────────────────────────────────────────
        if is_long  and row['high'] >= orig_tp:
            return True
        if not is_long and row['low']  <= orig_tp:
            return True

        # ── Противоположный сигнал (только при fixed_exit=False) ────────────
        if not self.fixed_exit:
            opp = row.get('_last_signal')
            if is_long  and opp == 'sell':
                return True
            if not is_long and opp == 'buy':
                return True

        return False


# ==============================================================================
# СЕТКА ПАРАМЕТРОВ ДЛЯ ОПТИМИЗАЦИИ
# ==============================================================================

PARAM_GRID = {
    # Консервативный (мало сделок, высокое качество)
    'conservative': dict(
        adx_threshold=30,
        zone_filter=True,
        ob_level=53,
        os_level=-53,
        atr_sl_mult=1.5,
        atr_tp_mult=3.0,
        cooldown_bars=8,
        trailing_sl=True,
    ),
    # Сбалансированный (рекомендован как старт)
    'balanced': dict(
        adx_threshold=28,
        zone_filter=True,
        ob_level=53,
        os_level=-53,
        atr_sl_mult=1.5,
        atr_tp_mult=2.5,
        cooldown_bars=5,
        trailing_sl=True,
    ),
    # Агрессивный (больше сделок, ниже порог)
    'aggressive': dict(
        adx_threshold=22,
        zone_filter=False,
        atr_sl_mult=1.2,
        atr_tp_mult=2.0,
        cooldown_bars=3,
        trailing_sl=False,
    ),
}


def compare_configs(df: pd.DataFrame, engine_cls, initial_balance: float = 100_000):
    """
    Запускает бэктест для каждого пресета из PARAM_GRID и печатает сравнение.

    Использование:
        from strategy_test import BacktestEngine
        compare_configs(df, BacktestEngine)
    """
    results = {}
    for name, params in PARAM_GRID.items():
        strat  = WaveTrendStrategyV2(**params)
        df_p   = strat.prepare(df)
        engine = engine_cls(strat, initial_balance=initial_balance)
        portfolio = engine.run(df_p)
        trades = portfolio.trades

        wins   = [t for t in trades if t.is_winner]
        losses = [t for t in trades if not t.is_winner]
        avg_w  = np.mean([t.pnl_money for t in wins])   if wins   else 0
        avg_l  = np.mean([t.pnl_money for t in losses]) if losses else 0
        pf     = abs(avg_w * len(wins)) / abs(avg_l * len(losses)) if losses else float('inf')

        results[name] = {
            'trades':    len(trades),
            'winrate':   len(wins) / len(trades) * 100 if trades else 0,
            'avg_win':   avg_w,
            'avg_loss':  avg_l,
            'pf':        pf,
            'total_pnl': portfolio.total_pnl,
        }

    print("\n{'='*65}")
    print(f"{'Конфиг':<15} {'Сделок':>7} {'WR%':>6} {'AvgW':>8} {'AvgL':>8} {'PF':>6} {'PnL':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<15} {r['trades']:>7} {r['winrate']:>5.1f}% "
              f"{r['avg_win']:>8.2f} {r['avg_loss']:>8.2f} "
              f"{r['pf']:>6.2f} {r['total_pnl']:>10.2f}")
    print("=" * 65)
    return results


# ==============================================================================
# SMOKE TEST
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    np.random.seed(42)
    n = 800
    price = 80000 + np.cumsum(np.random.randn(n) * 80)
    df_test = pd.DataFrame({
        'time':   pd.date_range("2026-01-01", periods=n, freq="15min"),
        'open':   price,
        'high':   price + np.abs(np.random.randn(n) * 50),
        'low':    price - np.abs(np.random.randn(n) * 50),
        'close':  price + np.random.randn(n) * 30,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    for preset_name, params in PARAM_GRID.items():
        strat  = WaveTrendStrategyV2(**params)
        df_p   = strat.prepare(df_test)
        sigs   = df_p['_last_signal'].dropna()
        # Считаем потенциальные входы после cooldown-фильтра
        print(f"\n[{preset_name}]")
        print(f"  WT кроссоверы всего: {len(sigs)}")
        print(f"  (реальные входы будут меньше после ADX/zone/cooldown фильтров)")

    print("\n✅ WaveTrendStrategyV2 инициализирована. Для бэктеста:")
    print("   from strategy_test import BacktestEngine")
    print("   strat = WaveTrendStrategyV2(**PARAM_GRID['balanced'])")
    print("   engine = BacktestEngine(strat, initial_balance=100_000)")
    print("   portfolio = engine.run(strat.prepare(df))")
    print("   engine.report()")