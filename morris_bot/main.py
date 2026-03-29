from morris_bot.indicators import (
    NoIndicator, RSIIndicator, MACDIndicator, StochasticIndicator,
    BollingerPercentBIndicator, AdaptiveMFIIndicator, DualConfirmIndicator,
)
from morris_bot.bot.morris_bot import MorrisBot
from morris_bot.backtest import (
    visual_backtest, visual_backtest_dual, run_backtest,
    test_telegram, test_all_topics,
)


if __name__ == "__main__":

    #── Быстрый визуальный просмотр (без статистики) ──────────────────────
    #visual_backtest('NGJ6', '15min', 2, indicator=NoIndicator(), use_confirmation=True)

    #── Полный бэктест: Dual BB%B + AI MFI (по умолчанию) ─────────────────
    visual_backtest_dual(
        ticker='NGJ6',
        tf='15min',
        days=12,
        forward_candles=10,
        min_move_pct=0.4,
        cooldown_candles=3,
        show_all=False,
        indicator=AdaptiveMFIIndicator()
    )

    # ── Бэктест с одним индикатором ────────────────────────────────────────
    # visual_backtest_dual('SBER', '15min', days=20,
    #                      indicator=RSIIndicator(14, oversold=30, overbought=70))

    # ── Только статистика (без графика) ───────────────────────────────────
    # signals_df, stats_df = run_backtest(
    #     ticker='SBER', tf='1h', days=60,
    #     indicator=DualConfirmIndicator(
    #         BollingerPercentBIndicator(),
    #         AdaptiveMFIIndicator(),
    #     ),
    #     forward_candles=10,
    #     min_move_pct=0.5,
    #     cooldown_candles=5,
    # )

    # ── Сравнение нескольких индикаторов ──────────────────────────────────
    # for label, ind in [
    #     ('No filter', NoIndicator()),
    #     ('RSI',       RSIIndicator()),
    #     ('BB%B',      BollingerPercentBIndicator()),
    #     ('AI MFI',    AdaptiveMFIIndicator()),
    #     ('Dual',      DualConfirmIndicator(BollingerPercentBIndicator(),
    #                                        AdaptiveMFIIndicator())),
    # ]:
    #     print(f'\n>>> {label}')
    #     run_backtest('NGJ6', '15min', 30, ind, forward_candles=10)

    # ── Тест Telegram ──────────────────────────────────────────────────────
    # test_telegram('BRH6', '15min')
    # test_all_topics()

    # ── Запуск бота ────────────────────────────────────────────────────────
    # bot = MorrisBot()
    # bot.run({
    #     'SiM6':  ['15min'],
    #     'BRJ6':  ['15min'],
    #     'CCJ6':  ['15min'],
    #     'NGJ6':  ['15min'],
    #     'KCJ6':  ['15min'],
    # })
