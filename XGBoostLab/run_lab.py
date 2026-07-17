import os
import pandas as pd

# Локальные модули XGBoostLab
from load_data import load_data_sync
from prepare_dataset_no_candles import compute_features_and_indicators
from ai_mfi_xgboost_strategy import AimfiXGBoostStrategy
from backtest_engine import BacktestEngine

# Конфигурация Т-Инвестиций (токен ТОЛЬКО из .env — не хардкодим боевые токены в коде!)
TINKOFF_TOKEN = os.getenv("TINKOFF_TOKEN", "")
FIGI = os.getenv("FIGI", "FUTSI0626000")  # Пример: SBER / Si / Ваш целевой инструмент

if not TINKOFF_TOKEN:
    raise RuntimeError("TINKOFF_TOKEN не задан в .env — обучение/бэктест невозможны без него.")

# Доля истории на обучение; остаток (более поздние бары) — чистый out-of-sample бэктест,
# который модель никогда не видела ни на этапе обучения, ни на этапе выбора tau.
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.75"))


def _split_time(df_10m: pd.DataFrame, train_fraction: float) -> pd.Timestamp:
    idx = int(len(df_10m) * train_fraction)
    idx = min(max(idx, 1), len(df_10m) - 1)
    return df_10m['time'].iloc[idx]


def run_pipeline():
    # --------------------------------------------------------------------------
    # 1. ЗАГРУЗКА ДАННЫХ ИЗ Т-ИНВЕСТИЦИЙ
    # --------------------------------------------------------------------------
    print("📥 1. Получение котировок из Т-Инвестиций...")
    df_10m, df_1h = load_data_sync(
        figi=FIGI,
        token=TINKOFF_TOKEN,
        days=180,
        force_reload=False  # Поставьте True, если нужно принудительно обновить с биржи
    )

    split_time = _split_time(df_10m, TRAIN_FRACTION)
    print(f"\n✂️  Хронологический сплит: train < {split_time}  |  OOS-бэктест >= {split_time}")

    # Модель увидит ТОЛЬКО эту часть истории — ни один бар из бэктеста не попадёт в обучение
    df_10m_train = df_10m[df_10m['time'] < split_time].reset_index(drop=True)
    df_1h_train = df_1h[df_1h['time'] < split_time].reset_index(drop=True)

    # --------------------------------------------------------------------------
    # 2. ОБУЧЕНИЕ МОДЕЛИ СТРАТЕГИИ (СТРОГО НА TRAIN-СРЕЗЕ)
    # --------------------------------------------------------------------------
    model_path = "xgb_model.pkl"

    # Любая ранее сохранённая модель могла быть обучена на 100% истории (старая версия
    # пайплайна) — то есть уже "видела" тестовый период. Считаем её контаминированной
    # и переобучаем на честном train-срезе.
    force_retrain = os.getenv("FORCE_RETRAIN", "1") == "1"
    if force_retrain and os.path.exists(model_path):
        print(f"⚠️  Удаляю потенциально контаминированную модель {model_path} — переобучаем на train-срезе.")
        os.remove(model_path)

    if not os.path.exists(model_path):
        print("\n🚀 3. Обучение XGBoost строго на train-срезе (без утечки в тестовый период)...")
        from train_mfi_htf import train_metalabel_model
        _, optimal_tau, _ = train_metalabel_model(
            df_10m=df_10m_train,
            df_1h=df_1h_train,
            save_model_path=model_path,
        )
    else:
        print(f"\n📦 3. Используем сохранённую модель: {model_path}")
        import joblib
        optimal_tau = joblib.load(model_path).get("optimal_tau", 0.55)

    print(f"🎯 Порог вероятности входа (Tau): {optimal_tau:.2f}")

    # --------------------------------------------------------------------------
    # 3. ИНДИКАТОРЫ НА ПОЛНОЙ ИСТОРИИ, БЭКТЕСТ — ТОЛЬКО НА OOS-ОТРЕЗКЕ
    # --------------------------------------------------------------------------
    # compute_features_and_indicators честно каузальна (rolling-окна смотрят только назад),
    # поэтому считать её на полной истории безопасно — просто у тестовых баров будет
    # корректный "разогрев" индикаторов, а не NaN в начале среза.
    print("\n⚙️ Генерация индикаторов (AiMFI, HTF, ATR) на полной истории...")
    df_features_full, df_htf_full = compute_features_and_indicators(df_10m, df_1h)

    df_features_test = df_features_full[df_features_full['time'] >= split_time].reset_index(drop=True)
    print(f"   OOS-отрезок для бэктеста: {len(df_features_test)} баров "
          f"({df_features_test['time'].min()} → {df_features_test['time'].max()})")

    # --------------------------------------------------------------------------
    # 4. ИСПОЛНЕНИЕ СТРАТЕГИИ И БЭКТЕСТ (ТОЛЬКО НА НЕВИДАННЫХ ДАННЫХ)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 4. СИМУЛЯЦИЯ ТОРГОВЛИ В BACKTEST ENGINE (OUT-OF-SAMPLE)")
    print("=" * 60)

    strategy = AimfiXGBoostStrategy(
        model_path=model_path,
        tau_override=optimal_tau,
        atr_sl_mult=1.5,
        rr_ratio=2.0,
    )

    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=100_000.0,
        position_size=10
    )

    # htf_df передаём ПОЛНЫЙ (не обрезанный) — HTF-контекст на начало теста должен
    # видеть предшествующую историю, это не утечка: get_htf_context фильтрует по
    # close_time <= текущий бар, т.е. остаётся строго каузальным.
    engine.run(df=df_features_test, htf_df=df_htf_full)
    df_trades = engine.report()

    return engine, df_trades


if __name__ == "__main__":
    run_pipeline()