import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

# Импорт базовой стратегии и сигналов из движка бэктеста
from strategy_test import BaseStrategy, Signal

# Импорт функций расчета индикаторов и контекста HTF из готового скрипта
from trading.pattern_detector.XGBoost.htf_mfi_no_candles.prepare_dataset_no_candles import compute_features_and_indicators, get_htf_context, get_htf_context


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
        Предварительный расчет индикаторов на случай, 
        если входной датасет еще не содержал рассчитываемые признаки.
        """
        df = df.copy()
        
        if 'atr' not in df.columns:
            hl = df['high'] - df['low']
            hcp = (df['high'] - df['close'].shift()).abs()
            lcp = (df['low'] - df['close'].shift()).abs()
            df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(14).mean()
            
        if 'ema10' not in df.columns:
            df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()

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

        # 2. Определение тренда и наклона HTF (без look-ahead bias)
        htf_trend, htf_slope = get_htf_context(htf_df, row['time']) if (self.use_htf_filter and htf_df is not None) else (0, 0.0)

        # 3. Базовая фильтрация по направлению тренда HTF
        sig_type_val = 0
        if is_long_trigger and (not self.use_htf_filter or htf_trend >= 0):
            sig_type_val = 1   # LONG
        elif is_short_trigger and (not self.use_htf_filter or htf_trend <= 0):
            sig_type_val = -1  # SHORT

        if sig_type_val == 0:
            return None

        # 4. Формирование вектора признаков (точно в том порядке, как при обучении)
        raw_features = {
            'signal_type': sig_type_val,
            'aimfi': curr_mfi,
            'aimfi_diff1': row.get('aimfi_diff1', curr_mfi - prev_mfi),
            'aimfi_diff3': row.get('aimfi_diff3', curr_mfi - df.iloc[i - 3]['aimfi'] if i >= 3 else 0.0),
            'atr_norm': row.get('atr_norm', row['atr'] / row['close']),
            'vol_ratio': row.get('vol_ratio', 1.0),
            'body_ratio': row.get('body_ratio', 0.5),
            'upper_shade_ratio': row.get('upper_shade_ratio', 0.2),
            'lower_shade_ratio': row.get('lower_shade_ratio', 0.2),
            'dist_ema10': row.get('dist_ema10', (row['close'] - row['ema10']) / row['ema10']),
            'htf_slope': htf_slope,
            'hour': row['time'].hour if hasattr(row['time'], 'hour') else 0,
            'dayofweek': row['time'].dayofweek if hasattr(row['time'], 'dayofweek') else 0,
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