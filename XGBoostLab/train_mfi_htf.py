import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import shap
from typing import List, Tuple, Dict, Any

# Импорт подготовщика данных
from prepare_dataset_no_candles import generate_ml_dataset

# ======================================================================
# 1. КОНФИГУРАЦИЯ И СПИСОК ФИЧЕЙ
# ======================================================================
FEATURE_COLS = [
    'signal_type',        # 1 для LONG, -1 для SHORT
    'aimfi', 'aimfi_diff1', 'aimfi_accel',
    'bb_z_score', 'atr_norm', 'atr_regime',
    'dist_to_high_20', 'dist_to_low_20',
    'vol_ratio', 'vol_accel', 'vol_5d_ratio',
    'dist_ema10', 'htf_slope', 'dist_htf_ema10',
    'body_ratio', 'upper_shade_ratio', 'lower_shade_ratio',
    'hour_sin', 'hour_cos', 'dayofweek'
]


# ======================================================================
# 2. РАСЧЕТ ВЕСОВ НАБЛЮДЕНИЙ (SAMPLE WEIGHTS)
# ======================================================================
def calculate_sample_weights(df: pd.DataFrame, max_holding_bars: int = 25) -> np.ndarray:
    """
    Рассчитывает веса для градиентного бустинга:
    1. Вес по величине PnL (|net_pnl|).
    2. Уникальность сигнала (штраф за кучность сделок во времени).
    """
    weights = np.abs(df['net_pnl'].values) + 1e-4  # Базовый вес от доходности

    # Убеждаемся в правильном типе времени
    times = pd.to_datetime(df['bar_time']).values
    n = len(df)
    uniqueness = np.ones(n)

    holding_delta = np.timedelta64(10 * max_holding_bars, 'm')

    for i in range(n):
        t_start = times[i]
        # Считаем количество сигналов, попавших в интервал удержания сделки
        overlap_count = np.sum((times >= t_start) & (times <= t_start + holding_delta))
        if overlap_count > 1:
            uniqueness[i] = 1.0 / overlap_count

    final_weights = weights * uniqueness
    # Нормализуем веса к среднему 1.0
    return final_weights / np.mean(final_weights)


# ======================================================================
# 3. PURGED WALK-FORWARD CROSS-VALIDATION
# ======================================================================
def purged_walk_forward_splits(
    df: pd.DataFrame, 
    n_splits: int = 5, 
    embargo_bars: int = 50
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Разбивает временной ряд на фракции Walk-Forward с соблюдением Purging и Embargo.
    """
    n_samples = len(df)
    fold_size = n_samples // (n_splits + 1)
    splits = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_start = train_end + embargo_bars  # Задержка Embargo против автокорреляции
        test_end = test_start + fold_size

        if test_end > n_samples:
            test_end = n_samples

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(test_idx) > 0 and len(train_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ======================================================================
# 4. ПОИСК ОПТИМАЛЬНОГО ВЕРОЯТНОСТНОГО ПОРОГА (\tau)
# ======================================================================
def optimize_probability_threshold(
    oof_df: pd.DataFrame, 
    min_trades: int = 20
) -> Tuple[float, Dict[str, Any]]:
    """
    Сканирует пороги вероятности Tau на Out-Of-Fold прогнозах
    и вычисляет оптимальный порог, максимизирующий Sharpe Ratio.
    """
    best_tau = 0.50
    best_sharpe = -999.0
    best_metrics = {}

    thresholds = np.arange(0.50, 0.85, 0.01)

    for tau in thresholds:
        selected = oof_df[oof_df['pred_prob'] >= tau]
        
        if len(selected) < min_trades:
            continue

        pnls = selected['net_pnl'].values
        win_rate = np.mean(selected['target'] == 1)
        
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = np.abs(np.sum(pnls[pnls < 0])) + 1e-8
        profit_factor = gross_profit / gross_loss

        # Аннуализированный коэффициент Шарпа (для 10m свечей: 252 дня * 144 бара)
        std_pnl = np.std(pnls) + 1e-8
        sharpe = (np.mean(pnls) / std_pnl) * np.sqrt(252 * 144)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_tau = float(tau)
            best_metrics = {
                'tau': round(best_tau, 2),
                'trades_count': int(len(selected)),
                'win_rate': round(float(win_rate), 4),
                'profit_factor': round(float(profit_factor), 3),
                'sharpe_ratio': round(float(sharpe), 3),
                'total_net_pnl': round(float(np.sum(pnls)), 5)
            }

    # Если ни один порог не подошел, возвращаем дефолтные значения
    if not best_metrics:
        best_metrics = {'tau': 0.50, 'trades_count': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'sharpe_ratio': 0.0, 'total_net_pnl': 0.0}

    return best_tau, best_metrics


# ======================================================================
# 5. ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ======================================================================
def train_metalabel_model(
    df_10m: pd.DataFrame, 
    df_1h: pd.DataFrame,
    save_model_path: str = "models/xgboost_metalabel.joblib"
) -> Tuple[xgb.XGBClassifier, float, pd.DataFrame]:
    
    print("🚀 [1/5] Генерация датасета квант-фичей...")
    dataset = generate_ml_dataset(df_10m, df_1h)
    
    if dataset.empty:
        raise ValueError("❌ Датасет пуст! Убедитесь в корректности входных данных.")

    print(f"📊 Сформировано {len(dataset)} сигналов.")
    print(f"   Распределение классов (Target): {dataset['target'].value_counts().to_dict()}")

    X = dataset[FEATURE_COLS]
    y = dataset['target']
    sample_weights = calculate_sample_weights(dataset)

    # ------------------------------------------------------------------
    # WALK-FORWARD КРОСС-ВАЛИДАЦИЯ
    # ------------------------------------------------------------------
    splits = purged_walk_forward_splits(dataset, n_splits=5, embargo_bars=50)
    
    oof_predictions = np.zeros(len(dataset))
    oof_mask = np.zeros(len(dataset), dtype=bool)

    # Параметры XGBoost (защищенные от оверфиттинга)
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,              # Неглубокие деревья
        'learning_rate': 0.015,       # Низкая скорость обучения
        'subsample': 0.7,            # Дропаут строк
        'colsample_bytree': 0.7,     # Дропаут фичей
        'min_child_weight': 15,      # Фильтрация редких ветвей
        'gamma': 0.1,                # Регуляризация сплита
        'random_state': 42
    }

    print("\n🔄 [2/5] Запуск Purged Walk-Forward Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        w_train = sample_weights[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Балансировка дисбаланса классов
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        scale_pos = num_neg / max(1, num_pos)

        model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos, n_estimators=300)
        
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        preds_prob = model.predict_proba(X_test)[:, 1]
        oof_predictions[test_idx] = preds_prob
        oof_mask[test_idx] = True

        auc = roc_auc_score(y_test, preds_prob) if len(np.unique(y_test)) > 1 else 0.5
        print(f"   • Фолд {fold + 1}/{len(splits)} | OOF Test AUC: {auc:.4f} | Объём Test: {len(test_idx)}")

    # ------------------------------------------------------------------
    # ОПТИМИЗАЦИЯ ПОРОГА (TAU)
    # ------------------------------------------------------------------
    oof_df = dataset[oof_mask].copy()
    oof_df['pred_prob'] = oof_predictions[oof_mask]

    print("\n🎯 [3/5] Оптимизация вероятностного порога (Tau)...")
    best_tau, best_metrics = optimize_probability_threshold(oof_df)
    
    print("=" * 55)
    print("📈 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ПОРОГА (Out-Of-Fold):")
    for k, v in best_metrics.items():
        print(f"  • {k:<15}: {v}")
    print("=" * 55)

    # ------------------------------------------------------------------
    # ФИНАЛЬНОЕ ОБУЧЕНИЕ МОДЕЛИ
    # ------------------------------------------------------------------
    print("\n🧠 [4/5] Финальное обучение XGBoost на всех данных...")
    num_neg = np.sum(y == 0)
    num_pos = np.sum(y == 1)
    final_scale_pos = num_neg / max(1, num_pos)

    final_model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=final_scale_pos, n_estimators=250)
    final_model.fit(X, y, sample_weight=sample_weights)

    # ------------------------------------------------------------------
    # SHAP-АНАЛИЗ ВАЖНОСТИ ФИЧЕЙ
    # ------------------------------------------------------------------
    print("🔍 [5/5] Анализ важности признаков (SHAP Values)...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)

    # Безопасная обработка массива SHAP в зависимости от версии
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\n🏆 ТОП-10 ВАЖНЕЙШИХ КВАНТ-ФИЧЕЙ:")
    print(feature_importance.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # СОХРАНЕНИЕ
    # ------------------------------------------------------------------
    model_dir = os.path.dirname(save_model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    payload = {
        'model': final_model,
        'features': FEATURE_COLS,
        'optimal_tau': best_tau,
        'metrics': best_metrics,
        'feature_importance': feature_importance
    }
    
    joblib.dump(payload, save_model_path)
    print(f"\n💾 Артефакты модели успешно сохранены в: {save_model_path}")

    return final_model, best_tau, feature_importance


# ======================================================================
# ТОЧКА ВХОДА ДЛЯ ЗАПУСКА
# ======================================================================
if __name__ == "__main__":
    # Запуск как отдельный скрипт (в т.ч. через os.system из run_lab.py):
    # берём уже закэшированные parquet-файлы из data/, ничего заново не качаем.
    from load_data import load_data_sync

    MODEL_PATH = os.getenv("XGB_MODEL_PATH", "xgb_model.pkl")
    TAU_PATH = os.getenv("XGB_TAU_PATH", "xgb_threshold.txt")

    TINKOFF_TOKEN = os.getenv("TINKOFF_TOKEN", "")
    FIGI = os.getenv("FIGI", "FUTSI0626000")

    print("📥 Загрузка данных (из кэша data/, если он есть)...")
    df_10m, df_1h = load_data_sync(figi=FIGI, token=TINKOFF_TOKEN, days=180, force_reload=False)

    model, tau, importance = train_metalabel_model(
        df_10m=df_10m,
        df_1h=df_1h,
        save_model_path=MODEL_PATH,
    )

    # Дублируем tau в отдельный txt — так его умеет читать run_lab.py
    with open(TAU_PATH, "w") as f:
        f.write(str(tau))
    print(f"💾 Порог Tau также сохранён отдельно в {TAU_PATH}")