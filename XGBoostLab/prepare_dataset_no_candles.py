import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional

# Импорт вашего адаптивного MFI
from trading.pattern_detector.indicators.mfi import AdaptiveMFIIndicator


def compute_features_and_indicators(
    df_10m: pd.DataFrame, 
    df_1h: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Рассчитывает полную матрицу квант-фичей (18 признаков) и индикаторы.
    
    :param df_10m: DataFrame рабочей серии (10m)
    :param df_1h: DataFrame старшей серии (1h)
    :return: (df_10m с фичами, df_1h с индикаторами HTF)
    """
    df = df_10m.copy()
    htf = df_1h.copy()

    # Убедимся, что колонка времени в формате datetime
    df['time'] = pd.to_datetime(df['time'])
    htf['time'] = pd.to_datetime(htf['time'])

    # ------------------------------------------------------------------
    # 1. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ 10m
    # ------------------------------------------------------------------
    # EMA10
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    # ATR(14)
    hl = df['high'] - df['low']
    hcp = (df['high'] - df['close'].shift()).abs()
    lcp = (df['low'] - df['close'].shift()).abs()
    df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(14).mean()

    # Adaptive MFI из mfi.py
    aimfi_calc = AdaptiveMFIIndicator(mfi_len=14, training_size=300)
    df['aimfi'] = aimfi_calc.compute(df)

    # ------------------------------------------------------------------
    # 2. ИНДИКАТОРЫ STARSHЕГО ТАЙМФРЕЙМА (1h HTF) С ЗАЩИТОЙ ОТ LOOK-AHEAD
    # ------------------------------------------------------------------
    htf['ema10_htf'] = htf['close'].ewm(span=10, adjust=False).mean()
    htf['htf_slope'] = htf['ema10_htf'].diff(2)  # Наклон EMA10 на HTF
    
    # Расчет времени закрытия часовой свечи
    htf_bar_duration = htf['time'].diff().median()
    htf['close_time'] = htf['time'] + htf_bar_duration

    # ------------------------------------------------------------------
    # 3. FEATURE ENGINEERING (18 СТАЦИОНАРНЫХ ФИЧЕЙ)
    # ------------------------------------------------------------------
    # А. Импульс и Динамика AiMFI
    df['aimfi_diff1'] = df['aimfi'].diff(1)
    df['aimfi_accel'] = df['aimfi_diff1'].diff(1)  # Вторая производная (Ускорение)

    # Б. Волатильность и Возврат к среднему (Bollinger & ATR)
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_z_score'] = (df['close'] - sma20) / (std20 + 1e-8)  # Z-score Боллинджера
    
    df['atr_norm'] = df['atr'] / df['close']
    atr_sma100 = df['atr'].rolling(100).mean()
    df['atr_regime'] = df['atr'] / (atr_sma100 + 1e-8)  # Фаза волатильности (Squeeze / Expansion)

    # В. Уровни Поддержки и Сопротивления (S/R Context)
    roll_high_20 = df['high'].rolling(20).max()
    roll_low_20 = df['low'].rolling(20).min()
    df['dist_to_high_20'] = (roll_high_20 - df['close']) / (df['atr'] + 1e-8)
    df['dist_to_low_20'] = (df['close'] - roll_low_20) / (df['atr'] + 1e-8)

    # Г. Объемы и Ликвидность
    vol_sma20 = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (vol_sma20 + 1e-8)
    df['vol_accel'] = df['vol_ratio'].diff(1)
    
    # 5 дней = 5 * 24 * 6 = 720 баров на 10-минутках
    vol_sma_5d = df['volume'].rolling(720).mean()
    df['vol_5d_ratio'] = vol_sma20 / (vol_sma_5d + 1e-8)  # Относительный тренд объема за неделю

    # Д. Локальные Тренды
    df['dist_ema10'] = (df['close'] - df['ema10']) / df['ema10']

    # Е. Геометрия свечи
    candle_range = df['high'] - df['low'] + 1e-8
    body_size = (df['close'] - df['open']).abs()
    df['body_ratio'] = body_size / candle_range
    
    upper_shade = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shade = df[['close', 'open']].min(axis=1) - df['low']
    df['upper_shade_ratio'] = upper_shade / candle_range
    df['lower_shade_ratio'] = lower_shade / candle_range

    # Ж. Циклическое кодирование времени
    hours = df['time'].dt.hour + df['time'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    df['dayofweek'] = df['time'].dt.dayofweek

    return df, htf


def get_htf_context(htf_df: pd.DataFrame, current_time: datetime) -> Tuple[int, float, float]:
    """
    Возвращает HTF-контекст на момент current_time строго без Look-Ahead bias.
    
    :return: (htf_trend, htf_slope, htf_ema10)
    """
    subset = htf_df[htf_df['close_time'] <= current_time]
    if subset.empty:
        return 0, 0.0, np.nan
    
    last = subset.iloc[-1]
    
    ema_val = last.get('ema10_htf', np.nan)
    if pd.isna(ema_val):
        return 0, 0.0, np.nan

    close_val = last['close']
    trend = 1 if close_val > ema_val else (-1 if close_val < ema_val else 0)
    slope = last['htf_slope'] if not pd.isna(last['htf_slope']) else 0.0
    
    return trend, slope, ema_val


def generate_ml_dataset(
    df_10m: pd.DataFrame, 
    df_1h: pd.DataFrame,
    atr_sl_mult: float = 1.5,
    rr_ratio: float = 2.0,
    mfi_long_thr: float = 40.0,
    mfi_short_thr: float = 60.0,
    max_holding_bars: int = 25,
    fee_pct: float = 0.0008  # 0.08% комиссия за круг (вход + выход)
) -> pd.DataFrame:
    """
    Генерирует датасет с фичами и PnL-ориентированным бинарным таргетом (Meta-Labeling).
    """
    # Расчет фичей и индикаторов
    df, htf = compute_features_and_indicators(df_10m, df_1h)
    dataset_rows = []
    
    # Начинаем с 720 бара, чтобы прогрелись все длинные rolling (например, 5d volume)
    start_idx = max(720, 300)
    
    for i in range(start_idx, len(df) - max_holding_bars):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Проверка валидности основных полей
        if pd.isna(row['atr']) or pd.isna(row['aimfi']) or pd.isna(prev_row['aimfi']):
            continue

        curr_mfi, prev_mfi = row['aimfi'], prev_row['aimfi']
        
        # Сигналы разворота AiMFI
        is_long = (prev_mfi < mfi_long_thr) and (curr_mfi >= mfi_long_thr)
        is_short = (prev_mfi > mfi_short_thr) and (curr_mfi <= mfi_short_thr)

        if not (is_long or is_short):
            continue

        # Контекст HTF
        htf_trend, htf_slope, htf_ema10 = get_htf_context(htf, row['time'])
        
        # Фильтрация по тренду HTF
        sig_type = None
        if is_long and htf_trend >= 0:
            sig_type = "LONG"
        elif is_short and htf_trend <= 0:
            sig_type = "SHORT"

        if sig_type is None:
            continue

        # ------------------------------------------------------------------
        # СИМУЛЯЦИЯ ТРОЙНОГО БАРЬЕРА И РАСЧЕТ PNL (TRIPLE BARRIER LABELING)
        # ------------------------------------------------------------------
        entry_price = row['close']
        atr = row['atr']
        
        sl_dist = atr * atr_sl_mult
        tp_dist = sl_dist * rr_ratio

        if sig_type == 'LONG':
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        exit_price = None
        future_bars = df.iloc[i + 1 : i + 1 + max_holding_bars]

        for _, f_row in future_bars.iterrows():
            if sig_type == 'LONG':
                hit_sl = f_row['low'] <= sl_price
                hit_tp = f_row['high'] >= tp_price
                
                if hit_sl and hit_tp:
                    # Пессимистичная обработка интрабарной неоднозначности
                    exit_price = sl_price
                    break
                elif hit_sl:
                    exit_price = sl_price
                    break
                elif hit_tp:
                    exit_price = tp_price
                    break

            else:  # SHORT
                hit_sl = f_row['high'] >= sl_price
                hit_tp = f_row['low'] <= tp_price
                
                if hit_sl and hit_tp:
                    exit_price = sl_price
                    break
                elif hit_sl:
                    exit_price = sl_price
                    break
                elif hit_tp:
                    exit_price = tp_price
                    break

        # Если за max_holding_bars уровни не пробиты — выход по цене закрытия 25-го бара
        if exit_price is None:
            exit_price = future_bars.iloc[-1]['close']

        # Расчет относительного PnL
        if sig_type == 'LONG':
            raw_pnl = (exit_price - entry_price) / entry_price
        else:
            raw_pnl = (entry_price - exit_price) / entry_price

        # Чистый PnL с учетом комиссии
        net_pnl = raw_pnl - fee_pct

        # Бинарный таргет: 1 — прибыльный сигнал для Meta-Model, 0 — убыточный
        target = 1 if net_pnl > 0 else 0

        # Расстояние от цены 10m до EMA10 старшего таймфрейма
        dist_htf_ema10 = (row['close'] - htf_ema10) / htf_ema10 if not pd.isna(htf_ema10) else 0.0

        # ------------------------------------------------------------------
        # ФОРМИРОВАНИЕ СТРОКИ С ФИЧАМИ (18 КВАНТ-ФИЧЕЙ)
        # ------------------------------------------------------------------
        features = {
            'bar_time': row['time'],
            'target': target,
            'signal_type': 1 if sig_type == 'LONG' else -1,
            'net_pnl': net_pnl,  # Сохраняем для аналитики и взвешивания выбытки/прибыли
            
            # 1. Импульс AiMFI
            'aimfi': curr_mfi,
            'aimfi_diff1': row['aimfi_diff1'],
            'aimfi_accel': row['aimfi_accel'],
            
            # 2. Волатильность и Боллинджер
            'bb_z_score': row['bb_z_score'],
            'atr_norm': row['atr_norm'],
            'atr_regime': row['atr_regime'],
            
            # 3. Уровни Поддержки / Сопротивления
            'dist_to_high_20': row['dist_to_high_20'],
            'dist_to_low_20': row['dist_to_low_20'],
            
            # 4. Объемы
            'vol_ratio': row['vol_ratio'],
            'vol_accel': row['vol_accel'],
            'vol_5d_ratio': row['vol_5d_ratio'],
            
            # 5. Тренды и HTF
            'dist_ema10': row['dist_ema10'],
            'htf_slope': htf_slope,
            'dist_htf_ema10': dist_htf_ema10,
            
            # 6. Свечи и Время
            'body_ratio': row['body_ratio'],
            'upper_shade_ratio': row['upper_shade_ratio'],
            'lower_shade_ratio': row['lower_shade_ratio'],
            'hour_sin': row['hour_sin'],
            'hour_cos': row['hour_cos'],
            'dayofweek': row['dayofweek'],
        }
        dataset_rows.append(features)

    return pd.DataFrame(dataset_rows)