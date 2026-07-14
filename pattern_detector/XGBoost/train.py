"""
train_xgboost.py
================
Обучение XGBoost-классификатора на датасете свечных паттернов.

Запуск:
    python3 XGBoost/train_xgboost.py
    python3 XGBoost/train_xgboost.py --dataset /path/to/si_patterns_dataset.parquet
    python3 XGBoost/train_xgboost.py --no-shap   # без SHAP (быстрее)

Выход:
    xgb_model.pkl            — обученная модель
    xgb_threshold.txt        — оптимальный порог вероятности
    feature_importance.png   — топ-25 важных признаков
    shap_summary.png         — SHAP-анализ (если не --no-shap)
    train_report.txt         — полный текстовый отчёт
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from xml.parsers.expat import model

import numpy as np
import pandas as pd

# Добавляем родительский каталог в путь (для импорта из patterns/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")   # без GUI — рендерим в файл
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

warnings.filterwarnings("ignore")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ОПИСАНИЕ ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════════════════════
#
# ── Блок A: Геометрия текущей свечи ──────────────────────────────────────────
#
#   c_body_ratio        — тело / полный диапазон свечи (0..1).
#                         Высокое значение → мощная направленная свеча,
#                         низкое → дожи или свеча с длинными тенями.
#
#   c_upper_sh_ratio    — верхняя тень / диапазон (0..1).
#                         Длинная верхняя тень у бычьего паттерна — плохой знак:
#                         продавцы давили цену сверху.
#
#   c_lower_sh_ratio    — нижняя тень / диапазон (0..1).
#                         Для Молота ожидаем значение > 0.6.
#                         Длинная нижняя тень = отбой от поддержки.
#
#   c_rel_body          — тело текущей свечи / среднее тело за 10 свечей.
#                         Значение > 1.3 → свеча крупнее среднего → паттерн весомее.
#
#   c_is_white          — 1 если бычья свеча, 0 если медвежья.
#
#   pattern_cleanliness — составная оценка "чистоты" паттерна (0..1).
#                         Для бычьих: 0.5 * body_ratio + 0.5 * lower_sh_ratio + бонус за размер.
#                         Для медвежьих: body_ratio + upper_sh_ratio.
#                         Чем чище паттерн — тем выше ожидаемая надёжность.
#
# ── Блок B: Контекст (предыдущие 2 свечи) ────────────────────────────────────
#
#   p_body_ratio        — тело предыдущей свечи / диапазон текущей.
#                         Для Engulfing ожидаем, что текущая свеча значительно
#                         крупнее предыдущей → p_body_ratio < 0.8.
#
#   pp_body_ratio       — тело свечи N-2 / диапазон текущей.
#
#   p_is_white          — направление предыдущей свечи.
#                         Для Bullish Engulfing p_is_white=0 (нужна медвежья перед бычьей).
#
#   pp_is_white         — направление свечи N-2.
#
#   same_color_streak   — 1 если все три свечи (текущая + 2 предыдущих) одного цвета.
#                         Три белых солдата / три чёрных вороны.
#
# ── Блок C: Полосы Боллинджера ───────────────────────────────────────────────
#
#   bb_pct_b            — позиция цены внутри полос: 0 = у нижней, 1 = у верхней.
#                         Бычий паттерн у нижней полосы (bb_pct_b < 0.2) —
#                         статистически надёжнее, чем в середине.
#                         Медвежий у верхней (bb_pct_b > 0.8) — аналогично.
#
#   bb_width            — ширина полос / средняя цена (нормированная волатильность).
#                         Узкие полосы (bb_width < 0.01) = рынок в сжатии,
#                         паттерны в сжатии часто дают ложные сигналы.
#
#   dist_to_upper       — расстояние до верхней полосы / цена.
#                         Чем меньше — тем ближе к сопротивлению.
#
#   dist_to_lower       — расстояние до нижней полосы / цена.
#                         Малое значение = цена у поддержки → усиливает бычий сигнал.
#
# ── Блок D: Объём ────────────────────────────────────────────────────────────
#
#   vol_ratio           — объём текущей свечи / средний объём за 20 свечей.
#                         vol_ratio > 1.5 на паттерне разворота = подтверждение
#                         "умных денег". vol_ratio < 0.5 = тихий рынок, ненадёжно.
#
#   vol_trend           — средний объём 3 последних свечей /
#                           средний объём 7 свечей перед ними.
#                         > 1.0 = объём нарастает (импульс),
#                         < 1.0 = объём угасает (истощение).
#
# ── Блок E: Тренд и позиция цены ─────────────────────────────────────────────
#
#   price_vs_ema        — (close − EMA10) / EMA10.
#                         Для бычьего паттерна ожидаем < 0 (цена под EMA = нисходящий тренд).
#                         Патерны у EMA (≈0) часто ложные — нет чёткого тренда.
#
#   ema_slope           — наклон EMA10 за 5 свечей = (ema[now] − ema[−5]) / ema[−5].
#                         Крутой нисходящий склон → рынок перепродан → бычий разворот вероятнее.
#
#   bars_in_trend       — сколько свечей подряд цена была по одну сторону от EMA.
#                         bars_in_trend = 15 → тренд зашёлся → статистически
#                         вероятность разворота выше.
#
# ── Блок F: Волатильность ─────────────────────────────────────────────────────
#
#   atr_pct             — ATR(14) / close. Нормированная волатильность.
#                         При высоком atr_pct цена "дышит" широко — цели 1.5 ATR
#                         достигаются быстрее. При низком — рынок вялый.
#
#   atr_ratio_5_20      — ATR(5) / ATR(20). Отношение краткосрочной к долгосрочной.
#                         < 1.0 = сужение, рынок готовится к движению.
#                         > 1.2 = расширение, рынок уже в движении.
#
# ── Блок G: Старший таймфрейм (1h) ───────────────────────────────────────────
#
#   htf_trend           — направление тренда на 1h: +1 (бычий) или −1 (медвежий).
#                         Ключевая фича: паттерн согласованный со старшим ТФ
#                         статистически работает лучше.
#
#   htf_ema_slope       — наклон EMA10 на 1h. Мера силы тренда на старшем ТФ.
#
#   htf_bb_pct_b        — %B на 1h. Позиция цены в полосах Боллинджера старшего ТФ.
#                         Бычий паттерн при htf_bb_pct_b < 0.3 = двойное подтверждение.
#
#   htf_bb_width        — ширина полос на 1h. Мера волатильности на старшем ТФ.
#
#   htf_vol_ratio       — объём на 1h / средний объём. Институциональный интерес.
#
#   htf_agreement       — 1 если направление паттерна совпадает с htf_trend.
#                         Самая бинарная и интерпретируемая фича из HTF-блока.
#
# ── Исключено из признаков ────────────────────────────────────────────────────
#
#   hour, day_of_week   — ИСКЛЮЧЕНЫ. На SHAP давали экстремальный разброс по
#                         отдельным часам при небольшой тестовой выборке (~424
#                         строк) — похоже на переобучение на шум конкретного
#                         периода, а не устойчивую сезонность. load_data.py их
#                         больше не считает как фичи; здесь на всякий случай
#                         тоже фильтруются через EXCLUDED_FEATURES, если
#                         используется старый parquet.
#
#   pattern             — тип паттерна (Hammer/Engulfing/...) БОЛЬШЕ НЕ подаётся
#                         модели как признак. По feature importance и SHAP
#                         идентичность паттерна была внизу обеих таблиц —
#                         контекст (объём, BB, согласованность со старшим ТФ)
#                         значит куда больше, чем название формации. Строка
#                         остаётся в датафрейме только как метаданные для
#                         разбивки метрик по типам (plot_pattern_analysis).
#
#   is_bullish          — ОСТАЁТСЯ. Это направление паттерна (1 бычий / 0
#                         медвежий), а не его идентичность — контекстный, а не
#                         категориальный признак.
#
# ══════════════════════════════════════════════════════════════════════════════


# ── Конфигурация ──────────────────────────────────────────────────────────────

DATASET_PATH    = "si_patterns_dataset.parquet"
MODEL_PATH      = "XGBoost/xgb_model.pkl"
THRESHOLD_PATH  = "XGBoost/xgb_threshold.txt"
OUTPUT_DIR      = "XGBoost"

# Столбцы которые НЕ являются признаками
NON_FEATURE_COLS = {"label", "datetime", "bar_idx", "pattern"}

# Признаки, которые исключаем из обучения сознательно, даже если они есть в parquet
# (например, датасет собирался старой версией load_data.py):
#   - hour / day_of_week — на SHAP давали экстремальный разброс по отдельным часам
#     при небольшой выборке, похоже на переобучение на шум, а не на устойчивую сезонность.
#   - pattern_encoded — идентичность типа паттерна (Hammer/Engulfing/...) почти не
#     влияла на предсказание (низ обеих таблиц важности), а её кодирование ранее
#     утекало в фичи в обход комментария в коде. Контекстные признаки вроде
#     is_bullish (направление) и pattern_cleanliness (чистота формы) — оставляем,
#     это не идентичность конкретного паттерна.
EXCLUDED_FEATURES = {"hour", "day_of_week", "pattern_encoded"}

# ── Экономика сделки (должна совпадать с load_data.py!) ────────────────────────
# Разметка: движение ≥ ATR_TARGET*ATR → успех (label=1), ≤ -ATR_STOP*ATR → стоп (label=0)
ATR_TARGET = 1.5
ATR_STOP   = 1.0
BREAKEVEN_PRECISION = ATR_STOP / (ATR_TARGET + ATR_STOP)   # = 0.40 при текущих значениях
# Запас над брейкивеном на комиссии/проскальзывание/неточность разметки
PRECISION_MARGIN     = 0.10
MIN_TARGET_PRECISION = BREAKEVEN_PRECISION + PRECISION_MARGIN   # = 0.50

# Параметры XGBoost
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        4,         # неглубокие деревья — меньше переобучение
    "learning_rate":    0.03,      # медленное обучение + много деревьев
    "subsample":        0.8,       # 80% строк на каждое дерево
    "colsample_bytree": 0.8,       # 80% признаков на каждое дерево
    "min_child_weight": 5,         # минимум 5 образцов в листе
    "gamma":            0.1,       # минимальный прирост для сплита
    "reg_alpha":        0.1,       # L1-регуляризация
    "reg_lambda":       1.0,       # L2-регуляризация
    "eval_metric":      "logloss",
    "use_label_encoder": False,
    "random_state":     42,
    "n_jobs":           -1,
}

# Кросс-валидация
CV_FOLDS = 5

# Доля данных в тесте (по времени, не рандомно!)
TEST_SIZE = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    log.info("Загружаю датасет: %s", path)
    df = pd.read_parquet(path)
    log.info("Загружено: %d строк, %d колонок", *df.shape)

    # Проверяем наличие обязательных колонок
    required = {"label", "datetime", "pattern"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"В датасете нет колонок: {missing}")

    log.info(
        "Метки: разворот(1)=%d (%.1f%%) | ложный(0)=%d (%.1f%%)",
        df["label"].sum(),         100 * df["label"].mean(),
        (df["label"] == 0).sum(),  100 * (1 - df["label"].mean()),
    )
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Готовит матрицу признаков X и вектор меток y.

    Тип паттерна (Hammer/Engulfing/...) СОЗНАТЕЛЬНО не кодируется и не подаётся
    модели: по feature importance и SHAP идентичность паттерна почти не влияла
    на предсказание — контекст (объём, положение в BB, согласованность со
    старшим ТФ) значит куда больше, чем название формации. Раньше здесь было
    ordinal-кодирование паттерна (LabelEncoder), но оно фактически утекало в
    фичи в обход намерения его исключить — теперь просто не создаём его.
    "pattern" остаётся в датафрейме только как метаданные для анализа
    (см. plot_pattern_analysis), но не идёт в X.

    hour/day_of_week тоже исключаются, если вдруг присутствуют в parquet
    (старый датасет) — см. EXCLUDED_FEATURES.
    """
    df = df.copy()

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c not in EXCLUDED_FEATURES
    ]

    X = df[feature_cols].copy()
    y = df["label"].astype(int)

    log.info("Признаков для обучения: %d", X.shape[1])
    log.info("Список признаков:\n  %s", "\n  ".join(X.columns.tolist()))

    dropped = (set(df.columns) & EXCLUDED_FEATURES) - NON_FEATURE_COLS
    if dropped:
        log.info("Явно исключены из признаков: %s", sorted(dropped))

    # Проверка на NaN
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        log.warning("NaN в колонках: %s — заполняю медианой", nan_cols)
        X[nan_cols] = X[nan_cols].fillna(X[nan_cols].median())

    return X, y


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> Tuple:
    """
    Разбивка на train/test ПО ВРЕМЕНИ — не рандомно!

    ПОЧЕМУ это критично:
        Рандомный split (train_test_split) перемешивает временные ряды.
        Модель "видит" будущее при обучении → метрики завышены в 1.5-2x.
        В реальной торговле такой эффект называется look-ahead bias и
        приводит к тому что бэктест показывает прибыль, а реальный счёт убыток.

    Мы берём последние test_size% данных (по времени) как тест.
    Модель обучается только на прошлом и тестируется на будущем.
    """
    n        = len(X)
    split_at = int(n * (1 - test_size))

    # Сортируем по времени (на случай если датасет не отсортирован)
    time_order = df["datetime"].argsort()
    X_sorted   = X.iloc[time_order].reset_index(drop=True)
    y_sorted   = y.iloc[time_order].reset_index(drop=True)

    X_train = X_sorted.iloc[:split_at]
    X_test  = X_sorted.iloc[split_at:]
    y_train = y_sorted.iloc[:split_at]
    y_test  = y_sorted.iloc[split_at:]

    log.info(
        "Train: %d строк (%s → %s) | Test: %d строк (%s → %s)",
        len(X_train),
        df["datetime"].iloc[time_order.iloc[0]].strftime("%Y-%m-%d"),
        df["datetime"].iloc[time_order.iloc[split_at - 1]].strftime("%Y-%m-%d"),
        len(X_test),
        df["datetime"].iloc[time_order.iloc[split_at]].strftime("%Y-%m-%d"),
        df["datetime"].iloc[time_order.iloc[-1]].strftime("%Y-%m-%d"),
    )

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# ОБУЧЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Компенсация дисбаланса классов через scale_pos_weight.

    XGBoost умножает градиент положительного класса на этот коэффициент,
    что эквивалентно oversampling класса 1.
    Формула: count(0) / count(1) — если разворотов 30%, получим 70/30 ≈ 2.3.

    ВАЖНО: НЕ используем SMOTE или random oversampling для временных рядов —
    синтетические образцы нарушили бы временную структуру данных.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ratio = n_neg / max(n_pos, 1)
    log.info(
        "Дисбаланс классов: neg=%d, pos=%d → scale_pos_weight=%.2f",
        n_neg, n_pos, ratio,
    )
    return ratio


def cross_validate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict,
) -> np.ndarray:
    """
    Временная кросс-валидация (TimeSeriesSplit в StratifiedKFold).

    Используем StratifiedKFold чтобы каждый fold имел похожее распределение
    меток. При очень сильном дисбалансе это критично для стабильной оценки.

    Метрика — ROC-AUC:
        - Не зависит от порога вероятности
        - Показывает способность модели ранжировать примеры
        - AUC=0.5 → случайная модель, AUC=0.7+ → хороший сигнал
    """
    model = xgb.XGBClassifier(**params)
    cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)   # shuffle=False = хронологический порядок

    scores = cross_val_score(
        model, X_train, y_train,
        cv      = cv,
        scoring = "roc_auc",
        n_jobs  = -1,
    )

    log.info(
        "CV ROC-AUC: %.3f ± %.3f  (folds: %s)",
        scores.mean(), scores.std(),
        " | ".join(f"{s:.3f}" for s in scores),
    )
    return scores


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    params:  Dict,
) -> xgb.XGBClassifier:
    """
    Финальное обучение с early stopping.

    Early stopping останавливает добавление деревьев когда logloss
    на валидационном наборе перестаёт улучшаться N раундов подряд.
    Это защищает от переобучения лучше чем фиксированное n_estimators.

    eval_set = X_test используем ТОЛЬКО для early stopping,
    не для подбора гиперпараметров — иначе тест "протечёт" в обучение.
    """
    log.info("Обучаю финальную модель...")

    params_with_es = {**params, "early_stopping_rounds": 30}
    model = xgb.XGBClassifier(**params_with_es)
    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = 50,
    )

    log.info(
        "Модель обучена: %d деревьев (best_iteration=%d)",
        model.n_estimators, model.best_iteration,
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ОЦЕНКА И ПОДБОР ПОРОГА
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(
    model:            xgb.XGBClassifier,
    X_test:           pd.DataFrame,
    y_test:           pd.Series,
    min_precision:    float = MIN_TARGET_PRECISION,
) -> float:
    """
    Подбор порога вероятности по кривой Precision-Recall — теперь по Precision,
    а не по F1.

    ПОЧЕМУ НЕ F1:
    F1 одинаково взвешивает Precision и Recall. Для торговли это неверная
    целевая функция: цена ложного срабатывания (FP) — реальные деньги на
    убыточной сделке, а цена пропущенного сигнала (FN) — просто упущенная
    возможность. Ошибки асимметричны, значит и оптимизировать нужно
    асимметрично.

    ЭКОНОМИКА СДЕЛКИ:
    Разметка использует target=ATR_TARGET*ATR, stop=ATR_STOP*ATR (см. константы
    выше и в load_data.py). Точка безубыточности:
        breakeven = ATR_STOP / (ATR_TARGET + ATR_STOP)
    При ATR_TARGET=1.5, ATR_STOP=1.0 → breakeven = 40%.
    Torговать при precision ниже этого числа — в среднем убыточно ещё до
    комиссий и проскальзывания. Поэтому просим MIN_TARGET_PRECISION с запасом
    (по умолчанию breakeven + 10 п.п. = 50%).

    АЛГОРИТМ:
    Среди порогов, где precision >= min_precision, берём тот, что даёт
    максимальный recall (чем ниже такой порог, тем больше сигналов при
    сохранении нужной точности). Если ни один порог не достигает
    min_precision — берём порог с максимально возможным precision и громко
    предупреждаем: модель в текущем виде не готова для прибыльной торговли.
    """
    proba  = model.predict_proba(X_test)[:, 1]
    precs, recs, thresholds = precision_recall_curve(y_test, proba)
    # precision_recall_curve возвращает на 1 элемент больше, чем thresholds —
    # обрезаем последнюю точку (соответствует threshold=+inf, recall=0)
    precs, recs = precs[:-1], recs[:-1]

    ok = precs >= min_precision

    if ok.any():
        # Среди подходящих по precision порогов — максимальный recall
        candidates = np.where(ok)[0]
        best_idx   = candidates[np.argmax(recs[candidates])]
        best_thr   = thresholds[best_idx]
        log.info(
            "Порог по target precision (>=%.2f): %.3f  →  Precision=%.3f | Recall=%.3f",
            min_precision, best_thr, precs[best_idx], recs[best_idx],
        )
    else:
        # Ни один порог не даёт нужный precision — берём максимум из доступного
        best_idx = precs.argmax()
        best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        log.warning(
            "НИ ОДИН порог не достиг target precision=%.2f (breakeven=%.2f + запас). "
            "Лучшее достижимое: Precision=%.3f | Recall=%.3f при пороге=%.3f. "
            "Модель в текущем виде НЕ готова к прибыльной торговле по всем "
            "паттернам сразу — см. per-pattern анализ (plot_pattern_analysis).",
            min_precision, BREAKEVEN_PRECISION, precs[best_idx], recs[best_idx], best_thr,
        )

    return float(best_thr)


def evaluate_model(
    model:     xgb.XGBClassifier,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    threshold: float,
) -> Dict:
    """Полная оценка модели на тестовой выборке."""
    proba  = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1  = f1_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_test, y_pred,
                                    target_names=["Ложный(0)", "Разворот(1)"])

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    margin_pp = (precision - BREAKEVEN_PRECISION) * 100

    log.info("\n%s", "═" * 55)
    log.info("  РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    log.info("═" * 55)
    log.info("  ROC-AUC  : %.4f", auc)
    log.info("  F1-score : %.4f  (порог=%.3f)", f1, threshold)
    log.info("  Confusion matrix:")
    log.info("             Pred 0  Pred 1")
    log.info("  Actual 0:  %5d   %5d   (FP=ложные сигналы)", tn, fp)
    log.info("  Actual 1:  %5d   %5d   (TP=пойманные развороты)", fn, tp)
    log.info(
        "  Precision: %.3f  |  Breakeven: %.3f  |  Запас: %+.1f п.п. %s",
        precision, BREAKEVEN_PRECISION, margin_pp,
        "(прибыльно в среднем)" if margin_pp > 0 else "(УБЫТОЧНО в среднем)",
    )
    log.info("\n%s", report)

    return {
        "roc_auc":   auc,
        "f1":        f1,
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "breakeven": BREAKEVEN_PRECISION,
        "margin_pp": margin_pp,
        "report":    report,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model:    xgb.XGBClassifier,
    features: List[str],
    out_path: str,
    top_n:    int = 25,
) -> None:
    """
    График важности признаков (gain).

    gain = суммарный прирост качества от всех сплитов по этому признаку.
    Интерпретация: признак с gain=100 в 10x важнее признака с gain=10.
    """
    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [features[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color="#26a69a",
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top-{top_n} важных признаков")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("График важности сохранён: %s", out_path)


def plot_shap(
    model:    xgb.XGBClassifier,
    X_test:   pd.DataFrame,
    out_path: str,
) -> None:
    """
    SHAP summary plot.

    SHAP (SHapley Additive exPlanations) показывает не просто "какой признак важен",
    но и В КАКУЮ СТОРОНУ и ПРИ КАКИХ ЗНАЧЕНИЯХ он влияет на предсказание.

    Читать график:
        - Каждая точка = один образец из теста
        - Ось X = SHAP value (вклад признака в log-odds предсказания)
        - Цвет = значение признака (красный=высокое, синий=низкое)

    Пример интерпретации:
        htf_agreement: красные точки справа → когда HTF согласован (=1),
        модель сильно повышает вероятность разворота. Синие слева → несогласованность
        снижает вероятность. Это подтверждает нашу гипотезу.
    """
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test)

        plt.figure(figsize=(10, 10))
        shap.summary_plot(shap_vals, X_test, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("SHAP summary сохранён: %s", out_path)
    except ImportError:
        log.warning("SHAP не установлен: pip install shap")
    except Exception as e:
        log.warning("SHAP ошибка: %s", e)


def plot_pattern_analysis(
    model:     xgb.XGBClassifier,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    df_test:   pd.DataFrame,
    threshold: float,
    out_path:  str,
) -> None:
    """
    Точность по каждому типу паттерна на тесте.
    Показывает какие паттерны модель научилась фильтровать лучше всего.
    """
    proba  = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    results = df_test.copy()
    results["y_pred"] = y_pred
    results["y_true"] = y_test.values
    results["correct"] = (results["y_pred"] == results["y_true"]).astype(int)

    summary = results.groupby("pattern").agg(
        total    = ("y_true", "count"),
        reversal = ("y_true", "sum"),
        correct  = ("correct", "sum"),
    )
    summary["base_rate"]    = summary["reversal"] / summary["total"]
    summary["model_acc"]    = summary["correct"]  / summary["total"]
    summary = summary.sort_values("total", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x  = np.arange(len(summary))
    w  = 0.35
    ax.bar(x - w/2, summary["base_rate"],  w, label="Base rate (% разворотов)", color="#ef5350", alpha=0.8)
    ax.bar(x + w/2, summary["model_acc"],  w, label="Accuracy модели",          color="#26a69a", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Доля")
    ax.set_title("Base rate vs точность модели по типу паттерна")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Добавляем количество образцов
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(i, 0.02, f"n={int(row['total'])}", ha="center", fontsize=7, color="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Анализ паттернов сохранён: %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# СОХРАНЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(
    model:     xgb.XGBClassifier,
    threshold: float,
    metrics:   Dict,
    features:  List[str],
    cv_scores: np.ndarray,
    model_path:    str,
    threshold_path: str,
    report_path:   str,
) -> None:
    """Сохраняет модель, порог и текстовый отчёт."""

    # Модель + список признаков в одном pickle.
    # Энкодера паттернов больше нет — тип паттерна не подаётся модели как фича.
    artifact = {
        "model":     model,
        "threshold": threshold,
        "features":  features,
        "trained_at": datetime.now().isoformat(),
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    log.info("Модель сохранена: %s", model_path)

    # Порог отдельным файлом — удобно читать без pickle
    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    log.info("Порог сохранён: %s  (%.4f)", threshold_path, threshold)

    # Текстовый отчёт
    with open(report_path, "w") as f:
        f.write("XGBoost Pattern Classifier — Training Report\n")
        f.write(f"Trained: {datetime.now()}\n\n")
        f.write(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write(f"Test ROC-AUC:  {metrics['roc_auc']:.4f}\n")
        f.write(f"Test F1:       {metrics['f1']:.4f}\n")
        f.write(f"Threshold:     {metrics['threshold']:.4f}\n")
        f.write(f"Precision:     {metrics['precision']:.4f}\n")
        f.write(f"Recall:        {metrics['recall']:.4f}\n")
        f.write(f"Breakeven:     {metrics['breakeven']:.4f}\n")
        f.write(f"Margin:        {metrics['margin_pp']:+.1f} pp "
                f"({'profitable on average' if metrics['margin_pp'] > 0 else 'UNPROFITABLE on average'})\n\n")
        f.write(f"TP={metrics['tp']} | FP={metrics['fp']} | "
                f"FN={metrics['fn']} | TN={metrics['tn']}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics["report"])
    log.info("Отчёт сохранён: %s", report_path)


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def main(dataset_path: str, run_shap: bool = True,
         min_precision: float = MIN_TARGET_PRECISION) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Загрузка и подготовка ──────────────────────────────────────────────
    df   = load_dataset(dataset_path)
    X, y = prepare_features(df)

    # ── 2. Временной split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = temporal_split(X, y, df)

    # Сохраняем строки теста с метаданными для анализа паттернов
    test_indices = y_test.index
    df_test_meta = df.iloc[test_indices].reset_index(drop=True)

    # ── 3. Кросс-валидация ────────────────────────────────────────────────────
    log.info("Кросс-валидация (%d folds)...", CV_FOLDS)

    spw    = compute_scale_pos_weight(y_train)
    params = {**XGB_PARAMS, "scale_pos_weight": spw}

    cv_scores = cross_validate_model(X_train, y_train, params)

    if cv_scores.mean() < 0.52:
        log.warning(
            "CV AUC=%.3f очень низкий. Возможные причины:\n"
            "  • Мало данных (< 200 паттернов)\n"
            "  • Рынок случаен на 15min горизонте\n"
            "  • Нужно расширить датасет (больше тикеров или дней)",
            cv_scores.mean()
        )

    # ── 4. Финальное обучение с early stopping ────────────────────────────────
    model = train_model(X_train, y_train, X_test, y_test, params)

    # ── 5. Подбор порога ──────────────────────────────────────────────────────
    threshold = find_optimal_threshold(model, X_test, y_test, min_precision=min_precision)

    # ── 6. Полная оценка ──────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test, threshold)

    # ── 7. Визуализация ───────────────────────────────────────────────────────
    feature_names = X_train.columns.tolist()

    plot_feature_importance(
        model, feature_names,
        out_path = os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )
    plot_pattern_analysis(
        model, X_test, y_test, df_test_meta, threshold,
        out_path = os.path.join(OUTPUT_DIR, "pattern_analysis.png"),
    )
    if run_shap:
        plot_shap(
            model, X_test,
            out_path = os.path.join(OUTPUT_DIR, "shap_summary.png"),
        )

    # ── 8. Сохранение артефактов ──────────────────────────────────────────────
    save_artifacts(
        model     = model,
        threshold = threshold,
        metrics   = metrics,
        features  = feature_names,
        cv_scores = cv_scores,
        model_path     = MODEL_PATH,
        threshold_path = THRESHOLD_PATH,
        report_path    = os.path.join(OUTPUT_DIR, "train_report.txt"),
    )

    log.info("══ Готово ══")


# ══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение XGBoost на паттернах Si")
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH,
        help=f"Путь к датасету (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Пропустить SHAP-анализ (быстрее, нужен: pip install shap)",
    )
    parser.add_argument(
        "--min-precision", type=float, default=MIN_TARGET_PRECISION,
        help=(
            f"Минимальный требуемый precision при подборе порога "
            f"(default: {MIN_TARGET_PRECISION:.2f} = breakeven "
            f"{BREAKEVEN_PRECISION:.2f} + запас {PRECISION_MARGIN:.2f})"
        ),
    )
    args = parser.parse_args()

    main(dataset_path=args.dataset, run_shap=not args.no_shap,
         min_precision=args.min_precision)