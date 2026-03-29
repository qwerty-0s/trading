import os
from morris_bot.indicators.base import BaseIndicator, NoIndicator


class ScannerConfig:
    def __init__(self,
                 long_body_coeff: float = None,
                 short_body_coeff: float = None,
                 shadow_limit: float = None,
                 indicator: BaseIndicator = None):
        self.long_body_coeff  = long_body_coeff  or float(os.getenv("DEFAULT_LONG_BODY_COEFF", 1.3))
        self.short_body_coeff = short_body_coeff or float(os.getenv("DEFAULT_SHORT_BODY_COEFF", 0.5))
        self.shadow_limit     = shadow_limit     or float(os.getenv("DEFAULT_SHADOW_LIMIT", 0.1))
        self.indicator: BaseIndicator = indicator or NoIndicator()
