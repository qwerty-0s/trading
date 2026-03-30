from morris_bot.indicators import (
    NoIndicator, RSIIndicator, MACDIndicator, StochasticIndicator,
    BollingerPercentBIndicator, AdaptiveMFIIndicator, DualConfirmIndicator,
)
from morris_bot.bot.morris_bot import MorrisBot
from morris_bot.backtest import (
    visual_backtest, visual_backtest_dual, run_backtest,
    test_telegram, test_all_topics,
)
from morris_bot.strategy_test import (
    StrategyParams, run_strategy_backtest,
    visual_strategy_backtest, compare_strategies,
)



if __name__ == "__main__":

    # ── Быстрый визуальный просмотр (без статистики) ──────────────────────
    # visual_backtest('BRJ6', '15min', 2, indicator=NoIndicator())

    # ── Статистический бэктест (с подтверждением паттернов) ───────────────
    # run_backtest('SBER', '15min', days=20,
    #              indicator=RSIIndicator(14),
    #              use_pattern_confirmation=True,
    #              forward_candles=10,
    #              min_move_pct=0.2)

    # ── Стратегический бэктест: SL на экстремуме, TP на EMA10 ─────────────
    # ── 1. Базовый запуск: фикс. 1.5R, без фильтров ──────────────────────
    """params_base = StrategyParams(
        tp_mode='atr',
        rr_multiplier=1.5,
        sl_mode='lookback',
        sl_lookback=5,
        max_candles=20,
        use_indicator=True,
        use_pattern_confirmation=True,
        min_rr=0.0,             # без фильтра R:R — видим все сделки
        trailing_stop=True,
        partial_take=True,
    )
 
    visual_strategy_backtest(
        ticker='NGJ6',
        tf='15min',
        days=30,
        indicator=AdaptiveMFIIndicator(),
        params=params_base,
    )"""

    # ── Dual-индикатор ─────────────────────────────────────────────────────
    # dual = DualConfirmIndicator(BollingerPercentBIndicator(), AdaptiveMFIIndicator())
    # visual_strategy_backtest('NGJ6', '15min', days=40,
    #                           indicator=dual, params=params)

    # ── Сравнение конфигураций ─────────────────────────────────────────────
    # compare_strategies('SBER', '15min', days=30, configs={
    #     'NoFilter': (NoIndicator(),   StrategyParams(sl_lookback=5)),
    #     'RSI':      (RSIIndicator(),  StrategyParams(sl_lookback=5)),
    #     'BB%B':     (BollingerPercentBIndicator(), StrategyParams(sl_lookback=7)),
    #     'Dual':     (DualConfirmIndicator(BollingerPercentBIndicator(),
    #                                       AdaptiveMFIIndicator()),
    #                  StrategyParams(sl_lookback=7, max_candles=25)),
    # })

    # ── Запуск бота ────────────────────────────────────────────────────────
    # bot = MorrisBot()
    # bot.run(
    #     config={'SiM6': ['15min'], 'BRJ6': ['15min'], 'NGJ6': ['15min']},
    #     indicators={'SiM6': RSIIndicator(14), 'BRJ6': RSIIndicator(14)},
    # )
    
    #test_all_topics()
    
    
    bot = MorrisBot()
    bot.run(config={
    'SiM6':  ['15min','30min'],
    'BRJ6':  ['15min','30min'],
    'CCJ6':  ['15min','30min'],
    'NGJ6':  ['15min','30min'],
    'KCJ6':  ['15min','30min']
    }            
    )
    
