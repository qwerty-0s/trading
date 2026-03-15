"""
backtest.py — бэктест MorrisBot за последние N дней (по умолчанию 2).

Запуск:
    python backtest.py

Что делает:
    1. Загружает данные за указанный период для всех пар ticker × tf.
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
    Запускает бэктест для всех пар ticker × tf за последние `days` дней.

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
    _send_summary(bot, all_signals, days, send_screenshots)

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
):
    """Формирует и отправляет итоговый отчёт в Telegram."""

    total     = len(signals)
    bullish   = sum(1 for s in signals if s.is_bullish)
    bearish   = total - bullish

    date_from = (datetime.now() - timedelta(days=days)).strftime("%d.%m.%Y")
    date_to   = datetime.now().strftime("%d.%m.%Y")

    lines = [
        f"📋 *Бэктест за {days} дн.* ({date_from} – {date_to})",
        f"",
        f"🔢 Всего сигналов:  *{total}*",
        f"🟢 Бычьих:  *{bullish}*   🔴 Медвежьих:  *{bearish}*",
        f"",
    ]

    if total == 0:
        lines.append("_Паттерны не обнаружены._")
    else:
        # Группируем по тикеру
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

    # Скриншот первого сигнала как «обложка» отчёта
    img_path = None
    if send_screenshots and signals:
        first = signals[0]
        bot_tmp = MorrisBot()
        detector = bot_tmp.get_detector(first.ticker)
        df_raw = bot_tmp.fetch_data(first.ticker, first.tf, days=10)
        if not df_raw.empty:
            df = bot_tmp._prepare_df(df_raw, detector)
            img_path = ChartVisualizer.create_screenshot(
                df, first.ticker, first.tf,
                first.pattern, first.candle_dt,
                detector.config.indicator,
                output_dir=bot_tmp.output_dir,
            )

    if img_path:
        bot.router.send_photo(route_ticker, route_tf, summary_text, img_path)
        os.remove(img_path)
    else:
        bot.router.send_message(route_ticker, route_tf, summary_text)

    print("\n[Backtest] Итоговый отчёт отправлен в Telegram.")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":

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
        send_screenshots=True,   # прикрепить скрин к итоговому сообщению
        send_per_signal=True,   # True = отдельный алерт на каждый паттерн
    )