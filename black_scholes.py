from __future__ import annotations

import math

from scipy.stats import norm

from market import Market
from option import Option
from price_parameters import PriceParameters
from sensibilities import GREEKS, TRADING_DAYS


class BlackScholes:
    def __init__(self, pricer_parameters: PriceParameters, market_data: Market, option_data: Option) -> None:
        if not BlackScholes.is_available(market_data, option_data):
            raise TypeError("Black-Scholes : l'option doit être européenne et sans dividende.")
        self.pricer_parameters, self.market_data, self.option_data = pricer_parameters, market_data, option_data
        self.maturity_years: float = PriceParameters.year_fraction(pricer_parameters.pricing_date, option_data.maturity)
        s0, k, r, vol = market_data.s0, option_data.strike, market_data.rate, market_data.vol
        self._degenerate = vol <= 1e-10 or self.maturity_years <= 1e-10 or s0 <= 1e-10   # Volatilité ou durée nulles : d1 et d2 non calculés
        if self._degenerate:
            self.d1 = self.d2 = 0.0
        else:
            self.d1 = (math.log(s0 / k) + (r + 0.5 * vol ** 2) * self.maturity_years) / (vol * math.sqrt(self.maturity_years))
            self.d2 = self.d1 - vol * math.sqrt(self.maturity_years)

    def pricing_bs(self) -> float:
        m, o = self.market_data, self.option_data
        if self._degenerate:
            fwd = m.s0 * math.exp(m.rate * self.maturity_years)
            intrinsic = max(fwd - o.strike, 0.0) if o.call_put == "call" else max(o.strike - fwd, 0.0)
            return intrinsic * math.exp(-m.rate * self.maturity_years)
        disc_k = o.strike * math.exp(-m.rate * self.maturity_years)
        if o.call_put == "call":
            return m.s0 * norm.cdf(self.d1) - disc_k * norm.cdf(self.d2)
        return disc_k * norm.cdf(-self.d2) - m.s0 * norm.cdf(-self.d1)

    @staticmethod
    def price_from(s0: float, k: float, maturity_years: float, r: float, vol: float, kind: str) -> float:
        """Prix Black-Scholes pour un strike donné, sans construire d'objets Market/Option."""
        if vol <= 1e-10 or maturity_years <= 1e-10 or s0 <= 1e-10:
            fwd = s0 * math.exp(r * max(maturity_years, 0.0))
            intrinsic = max(fwd - k, 0.0) if kind == "call" else max(k - fwd, 0.0)
            return intrinsic * math.exp(-r * max(maturity_years, 0.0))
        d1 = (math.log(s0 / k) + (r + 0.5 * vol ** 2) * maturity_years) / (vol * math.sqrt(maturity_years))
        d2 = d1 - vol * math.sqrt(maturity_years)
        if kind == "call":
            return s0 * norm.cdf(d1) - k * math.exp(-r * maturity_years) * norm.cdf(d2)
        return k * math.exp(-r * maturity_years) * norm.cdf(-d2) - s0 * norm.cdf(-d1)

    @staticmethod
    def is_available(market_data: Market, option_data: Option) -> bool:
        return not option_data.is_american() and market_data.dividend == 0.0 and market_data.vol > 0.0


class GreeksBS:
    def __init__(self, pricer_parameters: PriceParameters, market_data: Market, option_data: Option) -> None:
        self.bs = BlackScholes(pricer_parameters, market_data, option_data)

    def compute(self, greeks: tuple[str, ...] = GREEKS) -> dict:
        bs = self.bs
        if bs._degenerate:
            return {g: math.nan for g in greeks}
        s, k, r, v = bs.market_data.s0, bs.option_data.strike, bs.market_data.rate, bs.market_data.vol
        maturity_years = bs.maturity_years
        d1, d2, call = bs.d1, bs.d2, bs.option_data.call_put == "call"
        sign = 1.0 if call else -1.0
        phi, disc = norm.pdf(d1), k * math.exp(-r * maturity_years)
        vega = s * phi * math.sqrt(maturity_years)
        # Les unités des grecques suivent les conventions utilisées dans les analyses.
        values = {"Delta": norm.cdf(sign * d1) * sign, "Gamma": phi / (s * v * math.sqrt(maturity_years)),
            "Theta": (-s * phi * v / (2 * math.sqrt(maturity_years)) - sign * r * disc * norm.cdf(sign * d2)) / TRADING_DAYS,
            "Vega": vega / 100,  "Rho": sign * disc * maturity_years * norm.cdf(sign * d2) / 100,
            "Vanna": -phi * d2 / v / 100, "Vomma": vega * d1 * d2 / v / 100 ** 2}
        return {g: float(values[g]) for g in greeks}
