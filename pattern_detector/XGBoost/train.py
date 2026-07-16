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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb

warnings.filterwarnings("ignore")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH    = "si_patterns_dataset.parquet"
MODEL_PATH      = "XGBoost/xgb_model.pkl"
THRESHOLD_PATH  = "XGBoost/xgb_threshold.txt"
OUTPUT_DIR      = "XGBoost"

# Столбцы, которые НЕ являются признаками
NON_FEATURE_COLS = {"label", "datetime", "bar_idx", "pattern"}

# Признаки, которые исключаем из обучения сознательно
EXCLUDED_FEATURES = {"hour", "day_of_week", "pattern_encoded"}

# Экономика сделки
ATR_TARGET = 1.5
ATR_STOP   = 1.0
BREAKEVEN_PRECISION = ATR_STOP / (ATR_TARGET + ATR_STOP)   # = 0.40
PRECISION_MARGIN     = 0.10
MIN_TARGET_PRECISION = BREAKEVEN_PRECISION + PRECISION_MARGIN   # = 0.50

# Защита от выбора порога с "красивым" precision на 2-3 сэмплах: требуем,
# чтобы под порогом было хотя бы столько сигналов на Val. Иначе argmax(precs)
# на хвосте P-R кривой почти всегда указывает на статистический шум, а не на
# реальную закономерность.
MIN_SIGNALS_SUPPORT = 30

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

# Доли данных для разделения (по времени)
TEST_SIZE = 0.20  # 20% на итоговый изолированный тест
VAL_SIZE  = 0.15  # 15% на валидацию (early stopping + выбор порога)


# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    log.info("Загружаю датасет: %s", path)
    df = pd.read_parquet(path)
    log.info("Загружено: %d строк, %d колонок", *df.shape)

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


def temporal_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Трехкомпонентная хронологическая разбивка: Train -> Val -> Test.

    - Train (прошлое): Обучение деревьев XGBoost
    - Val (ближайшее прошлое): Early stopping + Оптимизация порога вероятности
    - Test (настоящее/будущее): Итоговая изолированная оценка качества
    """
    n = len(X)
    time_order = df["datetime"].argsort()
    X_sorted   = X.iloc[time_order].reset_index(drop=True)
    y_sorted   = y.iloc[time_order].reset_index(drop=True)
    df_sorted  = df.iloc[time_order].reset_index(drop=True)

    n_test  = int(n * test_size)
    n_val   = int(n * val_size)
    n_train = n - n_test - n_val

    X_train = X_sorted.iloc[:n_train]
    y_train = y_sorted.iloc[:n_train]

    X_val   = X_sorted.iloc[n_train : n_train + n_val]
    y_val   = y_sorted.iloc[n_train : n_train + n_val]

    X_test  = X_sorted.iloc[n_train + n_val :]
    y_test  = y_sorted.iloc[n_train + n_val :]

    log.info(
        "Train: %d строк (%s → %s)",
        len(X_train),
        df_sorted["datetime"].iloc[0].strftime("%Y-%m-%d"),
        df_sorted["datetime"].iloc[n_train - 1].strftime("%Y-%m-%d"),
    )
    log.info(
        "Val:   %d строк (%s → %s)",
        len(X_val),
        df_sorted["datetime"].iloc[n_train].strftime("%Y-%m-%d"),
        df_sorted["datetime"].iloc[n_train + n_val - 1].strftime("%Y-%m-%d"),
    )
    log.info(
        "Test:  %d строк (%s → %s)",
        len(X_test),
        df_sorted["datetime"].iloc[n_train + n_val].strftime("%Y-%m-%d"),
        df_sorted["datetime"].iloc[-1].strftime("%Y-%m-%d"),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# ОБУЧЕНИЕ И ВАЛИДАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def compute_scale_pos_weight(y_train: pd.Series) -> float:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ratio = n_neg / max(n_pos, 1)
    log.info(
        "Дисбаланс классов (Train): neg=%d, pos=%d → scale_pos_weight=%.2f",
        n_neg, n_pos, ratio,
    )
    return ratio


def cross_validate_model(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    params: Dict,
) -> np.ndarray:
    """
    Временная кросс-валидация через TimeSeriesSplit.
    Гарантирует отсутствие утечек данных из будущего в прошлое.
    """
    model = xgb.XGBClassifier(**params)
    cv    = TimeSeriesSplit(n_splits=CV_FOLDS)

    scores = cross_val_score(
        model, X_train_full, y_train_full,
        cv      = cv,
        scoring = "roc_auc",
        n_jobs  = -1,
    )

    log.info(
        "CV TimeSeries ROC-AUC: %.3f ± %.3f  (folds: %s)",
        scores.mean(), scores.std(),
        " | ".join(f"{s:.3f}" for s in scores),
    )
    return scores


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    params:  Dict,
) -> xgb.XGBClassifier:
    """Обучение модели с early stopping на ХРАНИМОМ ВАЛИДАЦИОННОМ множестве (X_val)."""
    log.info("Обучаю модель с ранней остановкой по Validation сету...")

    params_with_es = {**params, "early_stopping_rounds": 30}
    model = xgb.XGBClassifier(**params_with_es)
    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
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
    model:         xgb.XGBClassifier,
    X_val:         pd.DataFrame,
    y_val:         pd.Series,
    min_precision: float = MIN_TARGET_PRECISION,
    min_support:   int   = MIN_SIGNALS_SUPPORT,
) -> float:
    """
    Подбор оптимального порога вероятности исключительно на ВАЛИДАЦИОННОЙ выборке.

    Критерий: precision >= min_precision И количество сигналов под порогом
    (support = TP+FP) >= min_support. Без ограничения на support легко
    выбрать точку в хвосте P-R кривой, где precision=100%, потому что там
    буквально 2-3 сэмпла — статистический шум, а не сигнал.
    """
    proba  = model.predict_proba(X_val)[:, 1]
    precs, recs, thresholds = precision_recall_curve(y_val, proba)
    precs, recs = precs[:-1], recs[:-1]

    # support = TP + FP = TP / precision, а TP = recall * P (P = всего позитивов в Val)
    n_pos   = int(y_val.sum())
    tp      = recs * n_pos
    support = np.divide(tp, precs, out=np.zeros_like(tp), where=precs > 0)

    ok = (precs >= min_precision) & (support >= min_support)

    if ok.any():
        candidates = np.where(ok)[0]
        best_idx   = candidates[np.argmax(recs[candidates])]
        best_thr   = thresholds[best_idx]
        log.info(
            "Порог по target precision (Val >= %.2f, support >= %d): %.3f  →  "
            "Val Precision=%.3f | Val Recall=%.3f | Support=%d",
            min_precision, min_support, best_thr,
            precs[best_idx], recs[best_idx], int(support[best_idx]),
        )
    else:
        # Ищем максимум precision СРЕДИ порогов с достаточной поддержкой.
        # Если и такой нет — это явный сигнал, что данных мало и в живую
        # торговать по этому порогу рискованно (см. предупреждение ниже).
        supported = support >= min_support
        if supported.any():
            cand_idx = np.where(supported)[0]
            best_idx = cand_idx[np.argmax(precs[cand_idx])]
        else:
            best_idx = precs.argmax()
        best_thr = thresholds[best_idx]
        log.warning(
            "На Val ни один порог не достиг target precision=%.2f при support>=%d. "
            "Лучший доступный: Precision=%.3f | Recall=%.3f | Support=%d при пороге=%.3f. "
            "Это значит, что либо признаков не хватает, либо данных мало — "
            "перед реальной торговлей стоит перепроверить на большем датасете.",
            min_precision, min_support,
            precs[best_idx], recs[best_idx], int(support[best_idx]), best_thr,
        )

    return float(best_thr)


def evaluate_model(
    model:     xgb.XGBClassifier,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    threshold: float,
) -> Dict:
    """Чистая оценка модели на изолированном ТЕСТОВОМ наборе (X_test)."""
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
    log.info("  РЕЗУЛЬТАТЫ НА ИЗОЛИРОВАННОМ ТЕСТЕ (OUT-OF-SAMPLE)")
    log.info("═" * 55)
    log.info("  ROC-AUC  : %.4f", auc)
    log.info("  F1-score : %.4f  (применён порог=%.3f)", f1, threshold)
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
# ВИЗУАЛИЗАЦИЯ И СОХРАНЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model:    xgb.XGBClassifier,
    features: List[str],
    out_path: str,
    top_n:    int = 25,
) -> None:
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
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)

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
    proba  = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    results = df_test.copy()
    results["y_pred"]  = y_pred
    results["y_true"]  = y_test.values
    results["correct"] = (results["y_pred"] == results["y_true"]).astype(int)

    summary = results.groupby("pattern").agg(
        total    = ("y_true", "count"),
        reversal = ("y_true", "sum"),
        correct  = ("correct", "sum"),
    )
    summary["base_rate"] = summary["reversal"] / summary["total"]
    summary["model_acc"] = summary["correct"]  / summary["total"]
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

    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(i, 0.02, f"n={int(row['total'])}", ha="center", fontsize=7, color="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Анализ паттернов сохранён: %s", out_path)


def save_artifacts(
    model:         xgb.XGBClassifier,
    threshold:     float,
    metrics:       Dict,
    features:      List[str],
    cv_scores:     np.ndarray,
    model_path:    str,
    threshold_path: str,
    report_path:   str,
) -> None:
    artifact = {
        "model":      model,
        "threshold":  threshold,
        "features":   features,
        "trained_at": datetime.now().isoformat(),
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    log.info("Модель сохранена: %s", model_path)

    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    log.info("Порог сохранён: %s  (%.4f)", threshold_path, threshold)

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

    # 1. Загрузка данных
    df   = load_dataset(dataset_path)
    X, y = prepare_features(df)

    # 2. Хронологический сплит (Train / Val / Test)
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
        X, y, df, test_size=TEST_SIZE, val_size=VAL_SIZE
    )

    df_test_meta = df.iloc[y_test.index].reset_index(drop=True)

    # 3. Временная кросс-валидация на обучающем блоке (Train + Val)
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    log.info("Запуск временной кросс-валидации (%d folds)...", CV_FOLDS)
    spw    = compute_scale_pos_weight(y_train)
    params = {**XGB_PARAMS, "scale_pos_weight": spw}

    cv_scores = cross_validate_model(X_train_full, y_train_full, params)

    if cv_scores.mean() < 0.52:
        log.warning(
            "CV AUC=%.3f очень низкий. Возможные причины:\n"
            "  • Мало данных (< 200 паттернов)\n"
            "  • Рынок случаен на данном горизонте\n"
            "  • Нужно расширить датасет (больше тикеров, дней или таймфреймов)",
            cv_scores.mean()
        )

    # 4. Обучение модели с ранней остановкой по Validation сету
    model = train_model(X_train, y_train, X_val, y_val, params)

    # 5. Подбор порога на Validation сете
    threshold = find_optimal_threshold(model, X_val, y_val, min_precision=min_precision)

    # 6. Оценка на полностью заблокированном тестовом сете (Test)
    metrics = evaluate_model(model, X_test, y_test, threshold)

    # 7. Визуализация
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

    # 8. Сохранение артефактов
    save_artifacts(
        model          = model,
        threshold      = threshold,
        metrics        = metrics,
        features       = feature_names,
        cv_scores      = cv_scores,
        model_path     = MODEL_PATH,
        threshold_path = THRESHOLD_PATH,
        report_path    = os.path.join(OUTPUT_DIR, "train_report.txt"),
    )

    log.info("══ Готово ══")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение XGBoost на паттернах Si")
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH,
        help=f"Путь к датасету (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Пропустить SHAP-анализ (быстрее)",
    )
    parser.add_argument(
        "--min-precision", type=float, default=MIN_TARGET_PRECISION,
        help=f"Минимальный требуемый precision (default: {MIN_TARGET_PRECISION:.2f})",
    )
    args = parser.parse_args()

    main(dataset_path=args.dataset, run_shap=not args.no_shap,
         min_precision=args.min_precision)