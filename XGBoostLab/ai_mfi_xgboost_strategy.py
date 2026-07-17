import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from indicators.mfi import AdaptiveMFIIndicator
from backtest_engine import BaseStrategy, Signal
from prepare_dataset_no_candles import get_htf_context

class AimfiXGBoostStrategy(BaseStrategy):
    """
    Торговая стратегия Meta-Labeling на базе XGBoost.
    
    1. Находит базовые разворотные триггеры AiMFI + HTF.
    2. Извлекает вектор квант-фичей текущего бара.
    3. Запрашивает вероятностный прогноз модели XGBoost.
    4. Генерирует Signal только при P(Win) >= optimal_tau.
    """

    def __init__(
        self,
        model_path: str = "models/xgboost_metalabel.joblib",
        atr_sl_mult: float = 1.5,
        rr_ratio: float = 2.0,
        mfi_long_thr: float = 40.0,
        mfi_short_thr: float = 60.0,
        use_htf_filter: bool = True,
        tau_override: Optional[float] = None
    ):
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.mfi_long_thr = mfi_long_thr
        self.mfi_short_thr = mfi_short_thr
        self.use_htf_filter = use_htf_filter

        # Загрузка артефактов обученной модели XGBoost
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.feature_cols = artifacts['features']
        
        # Порог вероятности (из модели или переопределенный)
        self.tau = tau_override if tau_override is not None else artifacts.get('optimal_tau', 0.50)
        
        print(f"✅ Модель XGBoost загружена из {model_path}")
        print(f"🎯 Исполняемый вероятностный порог (Tau): {self.tau:.2f}")
        print(f"📊 Количество используемых фичей: {len(self.feature_cols)}")

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Стратегия НЕ пересчитывает признаки сама — она требует, чтобы вызывающий код
        (run_lab.py) уже прогнал df через compute_features_and_indicators() из
        prepare_dataset_no_candles.py, ТЕМИ ЖЕ функциями, что использовались при
        обучении модели. Пересчёт "похожих, но других" фичей здесь — прямой путь
        к train/inference skew (модель получает не те данные, на которых училась).
        """
        required = {
            'aimfi', 'aimfi_diff1', 'aimfi_accel', 'bb_z_score', 'atr_norm',
            'atr_regime', 'dist_to_high_20', 'dist_to_low_20', 'vol_ratio',
            'vol_accel', 'vol_5d_ratio', 'dist_ema10', 'body_ratio',
            'upper_shade_ratio', 'lower_shade_ratio', 'hour_sin', 'hour_cos',
            'dayofweek', 'atr', 'ema10',
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"AimfiXGBoostStrategy: в df отсутствуют колонки {sorted(missing)}. "
                f"Прогони df через compute_features_and_indicators() из "
                f"prepare_dataset_no_candles.py перед запуском BacktestEngine — "
                f"именно этот пайплайн использовался при обучении модели."
            )
        return df

    def on_bar(
        self,
        df: pd.DataFrame,
        i: int,
        htf_df: Optional[pd.DataFrame] = None
    ) -> Optional[Signal]:
        
        # Пропуск стартовых баров для корректной истории
        if i < 50:
            return None

        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Проверка целостности ключевых индикаторов
        if (
            pd.isna(row.get('atr', np.nan)) or
            pd.isna(row.get('aimfi', np.nan)) or
            pd.isna(prev_row.get('aimfi', np.nan))
        ):
            return None

        curr_mfi = row['aimfi']
        prev_mfi = prev_row['aimfi']

        # 1. Поиск базового разворотного триггера AiMFI
        is_long_trigger = (prev_mfi < self.mfi_long_thr) and (curr_mfi >= self.mfi_long_thr)
        is_short_trigger = (prev_mfi > self.mfi_short_thr) and (curr_mfi <= self.mfi_short_thr)

        if not (is_long_trigger or is_short_trigger):
            return None

        # 2. Определение тренда, наклона и уровня EMA10 HTF (без look-ahead bias)
        if self.use_htf_filter and htf_df is not None:
            htf_trend, htf_slope, htf_ema10 = get_htf_context(htf_df, row['time'])
        else:
            htf_trend, htf_slope, htf_ema10 = 0, 0.0, np.nan

        # 3. Базовая фильтрация по направлению тренда HTF
        sig_type_val = 0
        if is_long_trigger and (not self.use_htf_filter or htf_trend >= 0):
            sig_type_val = 1   # LONG
        elif is_short_trigger and (not self.use_htf_filter or htf_trend <= 0):
            sig_type_val = -1  # SHORT

        if sig_type_val == 0:
            return None

        # dist_htf_ema10 — точно так же, как считалось в generate_ml_dataset()
        dist_htf_ema10 = (row['close'] - htf_ema10) / htf_ema10 if not pd.isna(htf_ema10) else 0.0

        # 4. Вектор признаков — СТРОГО те же 21 фича и те же формулы,
        # что в FEATURE_COLS / generate_ml_dataset() из train_mfi_htf.py.
        # Все значения берутся из колонок, посчитанных compute_features_and_indicators(),
        # никакого альтернативного пересчёта здесь быть не должно.
        raw_features = {
            'signal_type': sig_type_val,
            'aimfi': curr_mfi,
            'aimfi_diff1': row['aimfi_diff1'],
            'aimfi_accel': row['aimfi_accel'],
            'bb_z_score': row['bb_z_score'],
            'atr_norm': row['atr_norm'],
            'atr_regime': row['atr_regime'],
            'dist_to_high_20': row['dist_to_high_20'],
            'dist_to_low_20': row['dist_to_low_20'],
            'vol_ratio': row['vol_ratio'],
            'vol_accel': row['vol_accel'],
            'vol_5d_ratio': row['vol_5d_ratio'],
            'dist_ema10': row['dist_ema10'],
            'htf_slope': htf_slope,
            'dist_htf_ema10': dist_htf_ema10,
            'body_ratio': row['body_ratio'],
            'upper_shade_ratio': row['upper_shade_ratio'],
            'lower_shade_ratio': row['lower_shade_ratio'],
            'hour_sin': row['hour_sin'],
            'hour_cos': row['hour_cos'],
            'dayofweek': row['dayofweek'],
        }

        # Выравнивание признаков строго под FEATURE_COLS модели
        X_sample = pd.DataFrame([raw_features])[self.feature_cols]

        # 5. Классификация в XGBoost (оценка вероятности положительного исхода)
        prob_win = self.model.predict_proba(X_sample)[0, 1]

        # 6. Фильтрация по оптимизированному порогу (Tau)
        if prob_win < self.tau:
            return None  # XGBoost заблокировал сигнал как низковероятный

        # 7. Расчет Stop-Loss и Take-Profit при прохождении фильтра
        stop_dist = row['atr'] * self.atr_sl_mult
        tp_dist = stop_dist * self.rr_ratio

        if sig_type_val == 1:
            return Signal(
                type="LONG",
                entry_price=row['close'],
                stop_loss=row['close'] - stop_dist,
                take_profit=row['close'] + tp_dist,
                pattern_name=f"AiMFI_XGBoost_LONG (P={prob_win:.2f})",
                bar_index=i,
                bar_time=row['time'],
            )
        else:
            return Signal(
                type="SHORT",
                entry_price=row['close'],
                stop_loss=row['close'] + stop_dist,
                take_profit=row['close'] - tp_dist,
                pattern_name=f"AiMFI_XGBoost_SHORT (P={prob_win:.2f})",
                bar_index=i,
                bar_time=row['time'],
            )