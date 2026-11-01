from __future__ import annotations

import datetime as dt
import gc

import streamlit as st

from market import Market
from option import Option
from orchestrator import Orchestrator
from price_parameters import PriceParameters, TreeMethod
from study import EtudeComparative, EtudePerformances, EtudePricing


class PricingApp:
    PRUNING_THRESHOLD = 1e-9
    MAX_STEPS = 10_000
    ENGINES = {"Vectorisé numpy": ("vector", "backward"), "Objets Node - backward": ("objects", "backward"),
               "Objets Node - récursif": ("objects", "recursive")}

    def __init__(self) -> None:
        st.set_page_config(page_title="Option Pricer", page_icon="📈", layout="wide")
        st.title("Option Pricer - arbre binomial/trinomial")
        st.caption("1. Saisissez les paramètres dans le menu de gauche · 2. Cliquez sur « Lancer le pricer » · "
                   "3. Explorez les trois onglets (les onglets 2 et 3 se lancent avec leur propre bouton).")

    def parameter_form(self) -> tuple[Market, Option, PriceParameters, bool]:
        """Recueille les paramètres dans le même ordre que le formulaire latéral."""
        today = dt.date.today()
        st.sidebar.header("Paramètres")
        spot, rate, vol, dividend, ex_div_date = self._market_form(today)
        strike, kind, exercise, maturity = self._option_form(today)
        pricing_date, method, nb_steps, engine, pricing_method, pruning, submitted = (
            self._pricer_form(today, maturity, spot, vol)
        )
        # Un dividende hors de la période de valorisation est sans effet.
        if not (pricing_date < ex_div_date <= maturity):
            dividend = 0.0
        market = Market(spot, rate, vol, dividend, ex_div_date if dividend > 0 else None)
        option = Option(strike, maturity, kind, exercise)
        pricer = PriceParameters(
            pricing_date, int(nb_steps), method, pruning,
            self.PRUNING_THRESHOLD, engine, pricing_method
        )
        return market, option, pricer, submitted

    @staticmethod
    def _market_form(today: dt.date) -> tuple:
        """Affiche et lit les champs de marché."""
        st.sidebar.subheader("Marché")
        spot = st.sidebar.number_input("Spot S0", min_value=0.01, value=100.0, step=1.0)
        rate = st.sidebar.number_input("Taux sans risque (%)", value=3.0, step=0.25, format="%.2f") / 100.0
        vol = st.sidebar.number_input("Volatilité (%)", min_value=0.0, value=20.0, step=1.0, format="%.2f") / 100.0
        dividend = st.sidebar.number_input(
            "Dividende (montant)", min_value=0.0, value=0.0, step=0.5,
            help="Dividende discret versé une fois ; 0 = pas de dividende."
        )
        ex_div_date = st.sidebar.date_input(
            "Date ex-dividende", value=today + dt.timedelta(days=182), format="DD/MM/YYYY"
        )
        return spot, rate, vol, dividend, ex_div_date

    @staticmethod
    def _option_form(today: dt.date) -> tuple:
        """Affiche et lit le type d'option, son strike et sa maturité."""
        st.sidebar.subheader("Option")
        strike = st.sidebar.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
        c1, c2 = st.sidebar.columns(2)
        kind = c1.selectbox("Type", ["call", "put"], format_func=str.capitalize)
        exercise = c2.selectbox(
            "Exercice", ["europeen", "americaine"],
            format_func=lambda e: "Européenne" if e == "europeen" else "Américaine"
        )
        maturity = st.sidebar.date_input(
            "Maturité", value=today + dt.timedelta(days=365), format="DD/MM/YYYY"
        )
        return strike, kind, exercise, maturity

    # Les réglages du pricer sont indépendants des données de marché.
    def _pricer_form(self, today: dt.date, maturity: dt.date,
                     spot: float, vol: float) -> tuple:
        """Affiche les choix du moteur, du nombre de pas et de l'élagage."""
        st.sidebar.subheader("Pricer")
        pricing_date = st.sidebar.date_input("Date de pricing", value=today, format="DD/MM/YYYY")
        method = st.sidebar.selectbox(
            "Méthode d'arbre", [TreeMethod.BINOMIAL, TreeMethod.TRINOMIAL],
            format_func=lambda m: m.label
        )
        optimal_mode = st.sidebar.radio(
            "Choix du nombre de pas", ["Nombre de pas N", "Nombre de pas optimal"],
            horizontal=True, key="steps_mode"
        ) == "Nombre de pas optimal"
        nb_steps = self._number_of_steps(optimal_mode, pricing_date, maturity, spot, vol)
        engine, pricing_method = self.ENGINES[
            st.sidebar.selectbox("Moteur de calcul", list(self.ENGINES))
        ]
        pruning = st.sidebar.checkbox(
            "Activer le pruning", value=True,
            help=f"Les nœuds de probabilité < {self.PRUNING_THRESHOLD:.0e} ne sont plus ramifiés."
        )
        submitted = st.sidebar.button("Lancer le pricer", type="primary")
        return pricing_date, method, nb_steps, engine, pricing_method, pruning, submitted

    # Les saisies restent disponibles quand le choix du mode change.
    def _number_of_steps(self, optimal: bool, pricing_date: dt.date,
                         maturity: dt.date, spot: float, vol: float) -> int:
        """Conserve en session les deux modes de saisie du nombre de pas."""
        st.session_state.setdefault("saved_nb_steps", 100)
        st.session_state.setdefault("saved_max_error", 0.01)
        if optimal:
            max_error = st.sidebar.number_input(
                "Erreur maximale", min_value=1e-6,
                value=st.session_state.saved_max_error,
                step=0.001, format="%.4f"
            )
            st.session_state.saved_max_error = max_error
            maturity_years = PriceParameters.year_fraction(pricing_date, maturity)
            nb_steps = min(
                PriceParameters.nb_steps_from_error(max_error, spot, vol, max(maturity_years, 1e-12)),
                self.MAX_STEPS
            )
            st.sidebar.caption(f"Nombre de pas optimal retenu : N = {nb_steps:,}")
        else:
            nb_steps = st.sidebar.number_input(
                "Nombre de pas N", min_value=1, max_value=self.MAX_STEPS,
                value=st.session_state.saved_nb_steps, step=10
            )
            st.session_state.saved_nb_steps = nb_steps
        return nb_steps

    # Vérification des limites de dates et de mémoire avant le calcul.
    @staticmethod
    def check(option: Option, pricer: PriceParameters) -> None:
        if option.maturity <= pricer.pricing_date:
            raise ValueError("La maturité doit être postérieure à la date de pricing.")
        if not pricer.pruning and pricer.nb_steps > pricer.max_unpruned_steps():
            raise ValueError(f"Sans pruning, l'arbre est limité à {pricer.max_unpruned_steps():,} pas avec ce moteur "
                             "(mémoire) : activez le pruning.")

    def run(self) -> None:
        """Lance le calcul demandé puis affiche les trois onglets."""
        market, option, pricer, submitted = self.parameter_form()
        if submitted:
            self._submit_pricing(market, option, pricer)
        self._show_tabs()

    # L’ancien arbre est libéré avant de lancer une nouvelle valorisation.
    def _submit_pricing(self, market: Market, option: Option,
                        pricer: PriceParameters) -> None:
        """Remplace les données du calcul précédent après validation."""
        try:
            self.check(option, pricer)
            old = st.session_state.pop("pricing", None)
            if old is not None:
                old["tree"].release()
            for key in ("res_greeks", "res_dg_values", "res_checks", "res_ex"):
                st.session_state.pop(key, None)
            gc.collect()
            with st.spinner("Construction et valorisation de l'arbre…"):
                st.session_state["pricing"] = Orchestrator.pricing_report(market, option, pricer)
            st.session_state["inputs"] = (market, option, pricer)
        except Exception as exc:
            st.error(f"Le pricer n'a pas pu terminer : {exc}")

    @staticmethod
    def _show_tabs() -> None:
        """Réutilise les derniers résultats pour chacun des onglets."""
        tabs = st.tabs(["Pricer et analyse d'un arbre", "Analyse comparative", "Analyse des performances"])
        if "pricing" not in st.session_state:
            for tab in tabs:
                tab.info("Choisissez les paramètres dans le menu latéral puis cliquez sur « Lancer le pricer ».")
            return
        market, option, pricer = st.session_state["inputs"]
        key = repr((vars(market), vars(option), vars(pricer)))
        with tabs[0]:
            EtudePricing.render(market, option, pricer, st.session_state["pricing"])
        with tabs[1]:
            EtudeComparative.render(market, option, pricer, key)
        with tabs[2]:
            EtudePerformances.render(market, option, pricer, key)


def main() -> None:
    PricingApp().run()


if __name__ == "__main__":
    main()
