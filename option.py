from __future__ import annotations

import datetime as dt
from enum import Enum

import numpy as np


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class OptionExerciceType(Enum):
    EUROPEEN = "europeen"
    AMERICAINE = "americaine"


class Option:
    def __init__(self, strike: float, maturity: dt.date, option_type: OptionType | str,
                 option_exercise_type: OptionExerciceType | str) -> None:
        if strike <= 0.0:
            raise ValueError("Le strike doit être strictement positif.")
        self.strike: float = float(strike)
        self.maturity = maturity
        self.call_put: str = OptionType(option_type).value        # "call" ou "put"
        self.eur_am = OptionExerciceType(option_exercise_type)

    def is_american(self) -> bool:
        return self.eur_am is OptionExerciceType.AMERICAINE

    def payoff(self, und_price: float) -> float:
        if self.call_put == "call":
            return max(und_price - self.strike, 0.0)
        return max(self.strike - und_price, 0.0)

    def payoff_array(self, und_price: np.ndarray) -> np.ndarray:
        """Version vectorisée du gain à l’exercice."""
        spot_prices = np.asarray(und_price, dtype=float)
        if self.call_put == "call":
            return np.maximum(spot_prices - self.strike, 0.0)
        return np.maximum(self.strike - spot_prices, 0.0)
