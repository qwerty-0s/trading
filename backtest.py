"""
backtest.py — бэктест MorrisBot за последние N дней (по умолчанию 2).

Запуск:
    python backtest.py

Что делает:
    1. Загружает данные за указанный период для всех пар ticker tf.
    2. Прогоняет каждую закрытую свечу через PatternDetector.
    3. Собирает статистику: всего сигналов, бычьих / медвежьих, по тикерам.
    4. Отправляет итоговый отчёт в Telegram (текст).
    5. Отправляет скриншот графика для каждого найденного сигнала.
"""

from trading_test import (
    MorrisBot, ScannerConfig, PatternDetector,
    ChartVisualizer, NoIndicator,
)

import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Структура одного сигнала
# ---------------------------------------------------------------------------

@dataclass
class BacktestSignal:
    ticker:    str
    tf:        str
    pattern:   str
    candle_dt: datetime
    price:     float
    is_bullish: bool
    ind_label: str = ""
    ind_value: float = float("nan")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

BULLISH_KEYWORDS = ("bull", "hammer", "morning", "soldier", "piercing")


def _is_bullish(pattern: str) -> bool:
    return any(k in pattern.lower() for k in BULLISH_KEYWORDS)


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%d.%m %H:%M")


# ---------------------------------------------------------------------------
# Основная функция бэктеста
# ---------------------------------------------------------------------------

def run_backtest(
    config: Dict[str, List[str]],
    days: int = 2,
    send_screenshots: bool = True,
    send_per_signal: bool = False,   # True → отдельное фото на каждый сигнал
) -> List[BacktestSignal]:
    """
    Запускает бэктест для всех пар ticker tf за последние `days` дней.

    Параметры
    ----------
    config           : словарь {"TICKER": ["tf1", "tf2"], ...}
    days             : глубина данных в днях (по умолчанию 2)
    send_screenshots : прикладывать ли скриншот к итоговому сообщению
                       (финальный сводный скрин первого найденного сигнала)
    send_per_signal  : если True — отправляет отдельный фото-сигнал на
                       каждый паттерн (как живой бот), иначе только сводку

    Возвращает список BacktestSignal.
    """

    bot = MorrisBot()
    all_signals: List[BacktestSignal] = []
    df_cache: Dict[Tuple[str, str], "pd.DataFrame"] = {}

    print(f"\n{'='*60}")
    print(f"  BACKTEST  |  глубина: {days} дн.  |  {datetime.now():%d.%m.%Y %H:%M}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Сбор сигналов
    # ------------------------------------------------------------------
    for ticker, timeframes in config.items():
        for tf in timeframes:
            print(f"\n▶ {ticker} {tf} — загрузка данных...")

            # Берём на 1 день больше, чтобы хватило на прогрев EMA/RSI
            df_raw = bot.fetch_data(ticker, tf, days=days + 1)
            if df_raw.empty:
                print(f"  ⚠ Нет данных для {ticker} {tf}")
                continue

            detector = bot.get_detector(ticker)

            # --------------------------------------------------------
            # Бэктест без фильтрации индикатором:
            # get_pattern_at_index вызывает bullish_confirmed() /
            # bearish_confirmed() внутри себя. Чтобы они всегда
            # возвращали True — подменяем индикатор на NoIndicator.
            # Остальные параметры конфига (long_body_coeff и др.)
            # сохраняются — поведение паттернов идентично боту.
            # --------------------------------------------------------
            detector.config.indicator = NoIndicator()

            df = bot._prepare_df(df_raw, detector)
            df_cache[(ticker, tf)] = df  # сохраняем для _send_summary

            # _prepare_df добавляет колонку индикатора; для NoIndicator
            # это "_no_indicator" с NaN — get_pattern_at_index это
            # обрабатывает и не блокирует сигналы.

            ind = detector.config.indicator   # теперь всегда NoIndicator
            ind_col = ind.column_name

            # Фильтруем только свечи нужного периода.
            # df после reset_index(drop=True) имеет индекс 0,1,...,N-1,
            # поэтому df_period.index содержит позиционные номера — iloc
            # внутри get_pattern_at_index работает корректно.
            cutoff = datetime.now() - timedelta(days=days)
            df_period = df[df["datetime"] >= cutoff]

            signals_for_pair: List[BacktestSignal] = []

            # Итерируем по всем свечам периода (кроме последней — открытая).
            # [:-1] по позиционному срезу, поэтому берём список индексов явно.
            period_indices = df_period.index.tolist()[:-1]

            for pos in period_indices:
                patterns = detector.get_pattern_at_index(df, pos)
                if not patterns:
                    continue

                row = df.loc[pos]
                for pattern in patterns:
                    ind_val = row[ind_col] if ind_col in df.columns else float("nan")
                    sig = BacktestSignal(
                        ticker=ticker,
                        tf=tf,
                        pattern=pattern,
                        candle_dt=row["datetime"],
                        price=row["close"],
                        is_bullish=_is_bullish(pattern),
                        ind_label=ind.plot_label if not isinstance(ind, NoIndicator) else "",
                        ind_value=float(ind_val) if ind_col in df.columns else float("nan"),
                    )
                    signals_for_pair.append(sig)
                    all_signals.append(sig)

                    # --- Опционально: отправить скрин на каждый сигнал ---
                    if send_per_signal:
                        direction = "🟢" if sig.is_bullish else "🔴"
                        ind_str = (
                            f"\n📈 {sig.ind_label}: `{sig.ind_value:.1f}`"
                            if sig.ind_label and not _is_nan(sig.ind_value)
                            else ""
                        )
                        msg = (
                            f"{direction} *[BACKTEST]* *{pattern}*\n"
                            f"📊 `{ticker}` | `{tf}`\n"
                            f"🕐 `{_format_dt(sig.candle_dt)}`\n"
                            f"💰 Цена: `{sig.price:.2f}`"
                            f"{ind_str}"
                        )
                        img_path = ChartVisualizer.create_screenshot(
                            df, ticker, tf, pattern, sig.candle_dt,
                            ind, output_dir=bot.output_dir,
                        )
                        if img_path:
                            bot.router.send_photo(ticker, tf, msg, img_path)
                            os.remove(img_path)
                        else:
                            bot.router.send_message(ticker, tf, msg)

            print(f"  ✅ Найдено сигналов: {len(signals_for_pair)}")
            for s in signals_for_pair:
                icon = "🟢" if s.is_bullish else "🔴"
                print(f"     {icon} {_format_dt(s.candle_dt)}  {s.pattern}  @ {s.price:.2f}")

    # ------------------------------------------------------------------
    # 2. Итоговый отчёт → Telegram
    # ------------------------------------------------------------------
    _send_summary(bot, all_signals, days, send_screenshots, df_cache=df_cache)

    print(f"\n{'='*60}")
    print(f"  Итого сигналов: {len(all_signals)}")
    print(f"{'='*60}\n")

    return all_signals


# ---------------------------------------------------------------------------
# Формирование и отправка сводного сообщения
# ---------------------------------------------------------------------------

def _is_nan(v: float) -> bool:
    import math
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


def _send_summary(
    bot: MorrisBot,
    signals: List[BacktestSignal],
    days: int,
    send_screenshots: bool,
    # df_cache хранит уже загруженные df чтобы не делать повторный fetch
    df_cache: Dict[Tuple[str, str], "pd.DataFrame"] = None,
):
    """Формирует и отправляет итоговый отчёт в Telegram.

    Текст всегда идёт через sendMessage — у него нет лимита на длину
    (Telegram принимает до 4096 символов).
    Скриншот отправляется отдельным sendPhoto с короткой подписью (≤1024 символа).
    Это разделение обязательно: sendPhoto caption ограничен 1024 символами,
    и при превышении сервер обрывает соединение.
    """

    total     = len(signals)
    bullish   = sum(1 for s in signals if s.is_bullish)
    bearish   = total - bullish

    date_from = (datetime.now() - timedelta(days=days)).strftime("%d.%m.%Y")
    date_to   = datetime.now().strftime("%d.%m.%Y")

    lines = [
        f"📋 *Бэктест за {days} дн.* ({date_from} – {date_to})",
        "",
        f"🔢 Всего сигналов:  *{total}*",
        f"🟢 Бычьих:  *{bullish}*   🔴 Медвежьих:  *{bearish}*",
        "",
    ]

    if total == 0:
        lines.append("_Паттерны не обнаружены._")
    else:
        by_ticker: Dict[str, List[BacktestSignal]] = {}
        for s in signals:
            by_ticker.setdefault(s.ticker, []).append(s)

        for ticker, sigs in by_ticker.items():
            b = sum(1 for s in sigs if s.is_bullish)
            r = len(sigs) - b
            lines.append(f"*{ticker}* — {len(sigs)} сигн. (🟢{b} / 🔴{r})")
            for s in sigs:
                icon = "🟢" if s.is_bullish else "🔴"
                ind_str = (
                    f"  _{s.ind_label}: {s.ind_value:.1f}_"
                    if s.ind_label and not _is_nan(s.ind_value)
                    else ""
                )
                lines.append(
                    f"  {icon} `{_format_dt(s.candle_dt)}` `{s.tf}` — "
                    f"{s.pattern} @ `{s.price:.2f}`{ind_str}"
                )
            lines.append("")

    summary_text = "\n".join(lines)

    # Определяем «главную» пару для роутинга (первый сигнал, или fallback)
    route_ticker = signals[0].ticker if signals else "SBER"
    route_tf     = signals[0].tf     if signals else "15min"

    # 1. Сводное сообщение — всегда текстом (sendMessage, лимит 4096 символов)
    bot.router.send_message(route_ticker, route_tf, summary_text)

    # 2. Скриншот первого сигнала — отдельным sendPhoto с короткой подписью
    if send_screenshots and signals:
        first = signals[0]

        # Используем уже загруженный df если передан, иначе fetch
        df = None
        if df_cache and (first.ticker, first.tf) in df_cache:
            df = df_cache[(first.ticker, first.tf)]
        else:
            df_raw = bot.fetch_data(first.ticker, first.tf, days=days + 1)
            if not df_raw.empty:
                detector = bot.get_detector(first.ticker)
                detector.config.indicator = NoIndicator()
                df = bot._prepare_df(df_raw, detector)

        if df is not None:
            img_path = ChartVisualizer.create_screenshot(
                df, first.ticker, first.tf,
                first.pattern, first.candle_dt,
                NoIndicator(),          # индикатор отключён и в бэктесте
                output_dir=bot.output_dir,
            )
            if img_path:
                # Короткая подпись — caption не должен превышать 1024 символа
                short_caption = (
                    f"📊 *{first.ticker}* `{first.tf}` — {first.pattern}\n"
                    f"🕐 `{_format_dt(first.candle_dt)}` @ `{first.price:.2f}`"
                )
                bot.router.send_photo(route_ticker, route_tf, short_caption, img_path)
                os.remove(img_path)

    print("\n[Backtest] Итоговый отчёт отправлен в Telegram.")


# ---------------------------------------------------------------------------
# Минимальный smoke-тест: найти первый сигнал и отправить один скриншот
# ---------------------------------------------------------------------------

def send_first_signal(ticker: str = "SiM6", tf: str = "15min", days: int = 2):
    """
    Загружает данные, находит первый паттерн за последние `days` дней
    и отправляет один sendPhoto в нужную тему.
    Используй это чтобы убедиться что доставка работает, прежде чем
    запускать полный run_backtest.
    """
    bot = MorrisBot()

    print(f"▶ Загрузка {ticker} {tf} за {days} дн...")
    df_raw = bot.fetch_data(ticker, tf, days=days + 1)
    if df_raw.empty:
        print("  ⚠ Нет данных")
        return

    detector = bot.get_detector(ticker)
    detector.config.indicator = NoIndicator()
    df = bot._prepare_df(df_raw, detector)

    cutoff = datetime.now() - timedelta(days=days)
    period_indices = df[df["datetime"] >= cutoff].index.tolist()[:-1]

    # Ищем первый индекс с паттерном
    first_sig = None
    for pos in period_indices:
        patterns = detector.get_pattern_at_index(df, pos)
        if patterns:
            row = df.loc[pos]
            first_sig = (pos, patterns[0], row)
            break

    if first_sig is None:
        print("  ⚠ Паттерны не найдены — попробуй увеличить days")
        return

    pos, pattern, row = first_sig
    candle_dt = row["datetime"]
    price     = row["close"]
    is_bull   = _is_bullish(pattern)
    icon      = "🟢" if is_bull else "🔴"

    print(f"  {icon} Сигнал: {pattern} @ {_format_dt(candle_dt)}, цена {price:.2f}")
    print(f"  Генерирую скриншот...")

    img_path = ChartVisualizer.create_screenshot(
        df, ticker, tf, pattern, candle_dt,
        NoIndicator(), output_dir=bot.output_dir,
    )

    if img_path:
        print(f"  Скриншот: {img_path}  ({os.path.getsize(img_path) // 1024} КБ)")
    else:
        print("  ⚠ Скриншот не создан — проверь kaleido/plotly")
        return

    caption = (
        f"{icon} *{pattern}*\n"
        f"📊 `{ticker}` | `{tf}`\n"
        f"🕐 `{_format_dt(candle_dt)}` @ `{price:.2f}`"
    )

    print(f"  Отправляю sendPhoto в тему {ticker}_{tf.upper()}...")
    bot.router.send_photo(ticker, tf, caption, img_path)

    if os.path.exists(img_path):
        os.remove(img_path)

    print("  Готово.")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Шаг 1: проверить доставку одного скриншота ---
    #send_first_signal(ticker="SiM6", tf="15min", days=2)

    #--- Шаг 2: когда доставка подтверждена — раскомментировать полный бэктест ---
    TICKERS_CONFIG = {
        "SiM6": ["15min"],
         "BRJ6": ["15min"],
         "CCH6": ["15min"],
         "NGH6": ["15min"],
         "KCJ6": ["15min"],
    }
    run_backtest(
         config=TICKERS_CONFIG,
         days=2,
         send_screenshots=True,
         send_per_signal=True,
    )