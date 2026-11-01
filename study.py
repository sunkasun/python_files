from __future__ import annotations

import datetime as dt
import gc
import math
import re
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from black_scholes import BlackScholes
from market import Market
from option import Option
from orchestrator import Orchestrator
from price_parameters import PriceParameters, TreeMethod
from sensibilities import GREEKS
from tree_visualizer import TreeVisualizer


# ====================================================================================
# Mise en forme commune aux trois onglets
# ====================================================================================


class Affichage:
    """Mise en forme commune : graphiques, tableaux, blocs, colonnes, messages d'erreur, résumé des paramètres."""

    NEW_WIDTH_API = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2]) >= (1, 50)
    COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    METHOD_STYLE = {TreeMethod.BINOMIAL: (COLORS[0], "solid"), TreeMethod.TRINOMIAL: (COLORS[1], "dash")}

    @staticmethod
    def show_chart(fig: go.Figure, key: str) -> None:
        width = {"width": "stretch"} if Affichage.NEW_WIDTH_API else {"use_container_width": True}
        st.plotly_chart(fig, key=key, config={"displaylogo": False, "scrollZoom": True}, **width)

    @staticmethod
    def show_df(df: pd.DataFrame, **kw: object) -> None:
        width = {"width": "stretch"} if Affichage.NEW_WIDTH_API else {"use_container_width": True}
        st.dataframe(df, hide_index=True, **width, **kw)

    @staticmethod
    def new_fig(title: str, x_title: str, y_title: str, height: int = 430) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)), template="plotly_white",
                          height=height, margin=dict(l=70, r=30, t=60, b=40), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
        fig.update_xaxes(title_text=x_title, showline=True, linecolor="#bbb")
        fig.update_yaxes(title_text=y_title, showline=True, linecolor="#bbb", tickformat="~g")
        return fig

    @staticmethod
    def lines(series: dict, title: str, y_title: str, key: str, log_x: bool = False, log_y: bool = False) -> None:
        """Courbes « N → valeur » : series = {libellé: (DataFrame, colonne, style de trait)}."""
        fig = Affichage.new_fig(title, "Nombre de pas N", y_title)
        for i, (label, (frame, col, dash)) in enumerate(series.items()):
            if not frame.empty:
                fig.add_trace(go.Scatter(x=frame["N"], y=frame[col], mode="lines+markers", name=label,
                                         line=dict(color=Affichage.COLORS[i % 7], dash=dash, width=2)))
        fig.update_xaxes(type="log" if log_x else "linear")
        fig.update_yaxes(type="log" if log_y else "linear")
        Affichage.show_chart(fig, key)

    @staticmethod
    def error_heatmap(df: pd.DataFrame, value: str, title: str, key: str, log: bool = True) -> None:
        """Heatmap p_min x N d'une erreur (couleur en log10) ou d'un pourcentage (log=False, couleur linéaire)."""
        pivot = df.pivot(index="p_min", columns="N", values=value).sort_index(ascending=False)
        values = pivot.to_numpy(dtype=float)
        fig = Affichage.new_fig(title, "Nombre de pas N", "Seuil de pruning p_min")
        fig.add_trace(go.Heatmap(x=[str(c) for c in pivot.columns], y=[f"{p:.0e}" for p in pivot.index],
                                 z=np.log10(np.maximum(values, 1e-16)) if log else values,
                                 colorscale="Viridis" if log else "Magma",
                                 text=[[f"{v:.1e}" if log else f"{v:.0%}" for v in row] for row in values],
                                 texttemplate="%{text}",
                                 colorbar=dict(title="log10(erreur)" if log else "part", thickness=14)))
        Affichage.show_chart(fig, key)

    @staticmethod
    def verdict_table(rows: list[dict], verdict_col: str = "Verdict") -> None:
        """Tableau de contrôles : ligne verte si « Validé », orange si « Attention », rouge si « Échec »."""
        colors = {"Validé": "background-color: #d4edda; color: #155724",
                  "Attention": "background-color: #fff3cd; color: #856404",
                  "Échec": "background-color: #f8d7da; color: #721c24"}
        df = pd.DataFrame(rows)
        styled = df.style.apply(lambda row: [colors.get(row[verdict_col], "")] * len(row), axis=1)
        Affichage.show_df(styled)

    @staticmethod
    @contextmanager
    def section(title: str, caption: str | None = None):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if caption:
                st.caption(caption)
            yield

    @staticmethod
    @contextmanager
    def safe(label: str):
        """Affiche une erreur lisible au lieu d'interrompre toute la page."""
        try:
            yield
        except Exception as exc:          # noqa: BLE001 - message utilisateur
            st.error(f"{label} : {exc}")

    @staticmethod
    def params_expander(market: Market, option: Option, pricer: PriceParameters, extra: dict | None = None) -> None:
        """Liste déroulante des paramètres utilisés dans l'analyse."""
        day = lambda d: d.strftime("%d/%m/%Y") if d is not None else "—"
        maturity_years = PriceParameters.year_fraction(pricer.pricing_date, option.maturity)
        rows = {"Spot S0": f"{market.s0:,.2f}", "Taux sans risque r": f"{market.rate:.2%}",
                "Volatilité": f"{market.vol:.2%}", "Dividende": f"{market.dividend:,.2f}",
                "Date ex-dividende": day(market.ex_div_date) if market.dividend > 0 else "—",
                "Strike K": f"{option.strike:,.2f}",
                "Option": f"{option.call_put.capitalize()} {'américain' if option.is_american() else 'européen'}",
                "Date de pricing": day(pricer.pricing_date),
                "Maturité": f"{day(option.maturity)} (T = {maturity_years:.4f} an)",
                "Méthode d'arbre": pricer.tree_method.label, "Nombre de pas N": f"{pricer.nb_steps:,}",
                "Moteur": "Vectorisé (numpy)" if pricer.engine == "vector" else f"Objets Node ({pricer.pricing_method})",
                "Pruning": f"Activé (p_min = {pricer.p_min:.0e})" if pricer.pruning else "Désactivé", **(extra or {})}
        with st.expander("Paramètres utilisés dans l'analyse", expanded=False):
            Affichage.show_df(pd.DataFrame({"Paramètre": list(rows), "Valeur": [str(v) for v in rows.values()]}))


# ====================================================================================
# Onglet 1 : pricer et analyse d'un arbre
# ====================================================================================


class EtudePricing:
    """Résultats, arbre, distribution, grecques."""

    @classmethod
    def render(cls, market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        Affichage.params_expander(market, option, pricer, {
            "Temps de construction (s)": f"{state['build_s']:.3f}",
            "Temps de valorisation (s)": f"{state['price_s']:.3f}",
        })
        with Affichage.safe("Résultat du pricing"):
            cls.resultats(market, option, pricer, state)
        with Affichage.safe("Grecques"):
            cls.grecques(market, option, pricer, state)
        with Affichage.safe("Arbre"):
            cls.arbre(pricer, state)
        with Affichage.safe("Exercice anticipé par date"):
            cls.exercice_par_date(market, option, pricer, state)
        with Affichage.safe("Distribution"):
            cls.distribution(market, option, pricer, state)
        with Affichage.safe("Débogage et contrôles de cohérence"):
            cls.controles(market, option, pricer, state)

    @staticmethod
    def resultats(market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        """Cartes : prix, valeur intrinsèque, valeur temps, Black-Scholes, taille, probabilités, temps."""
        price, bs = state["price"], state["bs"]
        intrinsic = option.payoff(market.s0)
        tree = state["tree"]                                   # forward du tronc à maturité (dividende inclus)
        fwd_intrinsic = math.exp(-market.rate * tree.dt_ * tree.n_steps) * option.payoff(tree._steps[-1]["F"])
        time_ref = intrinsic if option.is_american() else fwd_intrinsic   # européenne : pas d'exercice avant T
        last = state["tree"].levels[-1]
        spot_prices, probabilities = last["S"], last["P"]
        tol = 1e-9 * option.strike
        label = f"{option.call_put.capitalize()} {'américain' if option.is_american() else 'européen'}"
        with Affichage.section(f"Résultat du pricing - {label}, {pricer.tree_method.label}, {pricer.nb_steps:,} pas"):
            c: object

            def _afficher_prix() -> None:
                """Affiche les prix et la taille du résultat."""
                nonlocal c
                c = st.columns(5)
                c[0].metric("Prix de l'option", f"{price:,.6f}")
                c[1].metric("Valeur intrinsèque", f"{intrinsic:,.2f}")
                c[2].metric("Valeur temps", f"{price - time_ref:,.6f}",
                            help="Prix − valeur intrinsèque immédiate (américaine) ou forward (européenne, qui ne peut "
                                 "pas être exercée avant la maturité). Pour une européenne, prix − valeur intrinsèque "
                                 "spot peut être négatif (put très dans la monnaie, call avec dividende).")
                c[3].metric("Black-Scholes", "N/A" if bs is None else f"{bs:,.6f}",
                            delta=None if bs is None else f"arbre - BS : {price - bs:+.5f}", delta_color="off",
                            help="Disponible seulement pour une option européenne sans dividende.")
                c[4].metric("Taille de l'arbre", f"{state['nodes']:,} nœuds")
            _afficher_prix()
            c: object

            def _afficher_probabilites() -> None:
                """Affiche les masses terminales et les temps de calcul."""
                nonlocal c
                c = st.columns(5)
                c[0].metric("P(S_T > K)", f"{probabilities[spot_prices > option.strike + tol].sum():.2%}")
                c[1].metric("P(S_T < K)", f"{probabilities[spot_prices < option.strike - tol].sum():.2%}")
                c[2].metric("P(S_T = K)", f"{probabilities[np.abs(spot_prices - option.strike) <= tol].sum():.2%}")
                c[3].metric("Temps de calcul", f"{state['build_s'] + state['price_s']:.2f} s",
                            help=f"Construction {state['build_s']:.2f} s + valorisation {state['price_s']:.2f} s.")
                c[4].metric("Valeur intrinsèque forward", f"{fwd_intrinsic:,.4f}",
                            help="e^(−rT)·payoff(F), F = forward à maturité (dividende inclus) : borne basse du prix "
                                 "d'une européenne. Valeur temps d'une européenne = prix − cette valeur, toujours ≥ 0.")
            _afficher_probabilites()

    @staticmethod
    def arbre(pricer: PriceParameters, state: dict) -> None:
        """Arbre de valorisation : ordonnée, couleur des nœuds, fenêtre de pas, frontière d'exercice."""
        n = pricer.nb_steps
        with Affichage.section(f"Arbre de valorisation"):
            c_y, c_color, c_log = st.columns([1, 1, 0.6], vertical_alignment="bottom")
            y_label = c_y.selectbox("Ordonnée", list(TreeVisualizer.Y_OPTIONS), key="tree_y")
            color_label = c_color.selectbox("Couleur des nœuds", list(TreeVisualizer.COLOR_OPTIONS), key="tree_color")
            log_y = c_log.checkbox("Échelle log", key="tree_log")
            t_range = st.slider("Fenêtre de pas affichée", 0, n, (0, min(n, 60)), key="tree_window") if n > 1 else (0, n)
            cache_key = (id(state["tree"]), y_label, color_label, tuple(t_range), log_y)
            if st.session_state.get("tree_view_key") != cache_key:
                st.session_state["tree_view"] = TreeVisualizer.build(
                    state["tree"], TreeVisualizer.Y_OPTIONS[y_label],
                    TreeVisualizer.COLOR_OPTIONS[color_label], tuple(t_range), log_y)
                st.session_state["tree_view_key"] = cache_key
            view = st.session_state["tree_view"]
            Affichage.show_chart(view["fig"], "tree_plot")
            st.caption(view["caption"])
            if view.get("svg") is not None:
                st.download_button(
                    "Exporter l'arbre en SVG", data=view["svg"].encode("utf-8"),
                    file_name="arbre_valorisation.svg", mime="image/svg+xml", key="tree_svg_download"
                )

    @staticmethod
    def exercice_par_date(market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        """Option américaine : probabilité cumulée d'exercice anticipé et valeur moyenne actualisée par date."""
        with Affichage.section("Exercice anticipé : probabilité d'exercer et valeur moyenne actualisée par date",
                               "Probabilité d'exercer pour la première fois avant t (passe avant absorbante). "
                               "Valeur moyenne e^(−rt)·Σ P·V : constante pour l'européenne (martingale), décroissante "
                               "pour l'américaine (surmartingale) ; sa baisse totale est la prime américaine."):
            if not option.is_american():
                st.info("Section réservée aux options américaines.")
                return
            saved = st.session_state.get("res_ex")                      # vidé à chaque nouveau pricing
            if saved is None:
                with st.spinner("Probabilités d'exercice…"):
                    saved = Orchestrator.exercise_profile(market, option, pricer)
                st.session_state["res_ex"] = saved
            def _statistiques_exercice() -> tuple:
                """Évalue la probabilité d'exercice et la prime américaine."""
                times, first = saved["times"], saved["first"]
                early = np.cumsum(first[:-1])
                total = float(early[-1]) if len(early) else 0.0
                median = float(times[np.searchsorted(early, total / 2)]) if total > 0 else float("nan")
                premium = float(saved["americaine"][0] - saved["europeen"][0])
                return times, early, total, median, premium

            times, early, total, median, premium = _statistiques_exercice()
            c = st.columns(3)
            c[0].metric("Probabilité d'exercice anticipé", f"{total:.1%}")
            c[1].metric("Date médiane d'exercice", f"{median:.3f} an" if total > 0 else "—")
            c[2].metric("Prime américaine", f"{premium:,.4f}")
            c1, c2 = st.columns(2)
            fig: object

            def _courbe_exercice() -> None:
                """Trace la probabilité cumulée d’exercice anticipé."""
                nonlocal fig
                with c1:
                    fig = Affichage.new_fig("Probabilité d'avoir exercé avant t", "Temps t (années)", "Probabilité cumulée")
                    fig.add_trace(go.Scatter(x=times[:-1], y=early, mode="lines", name="Exercice anticipé",
                                             line=dict(color=Affichage.COLORS[1], width=2)))
                    if saved["div_time"] > 0:
                        fig.add_vline(x=saved["div_time"], line_dash="dash", line_color="grey",
                                      annotation_text="détachement")
                    Affichage.show_chart(fig, "ex_cum")
            _courbe_exercice()
            fig: object

            def _courbe_valeur() -> None:
                """Trace la valeur moyenne actualisée des deux options."""
                nonlocal fig
                with c2:
                    fig = Affichage.new_fig("Valeur moyenne actualisée par date", "Date t (années)", "E[e^(−rt)·V_t]")
                    for col, label, color in (("americaine", "Américaine", Affichage.COLORS[1]),
                                              ("europeen", "Européenne", Affichage.COLORS[0])):
                        fig.add_trace(go.Scatter(x=times, y=saved[col], mode="lines", name=label,
                                                 line=dict(color=color, width=2)))
                    if saved["div_time"] > 0:
                        fig.add_vline(x=saved["div_time"], line_dash="dash", line_color="grey")
                    Affichage.show_chart(fig, "ex_avg")
            _courbe_valeur()

    @staticmethod
    def distribution(market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        """Histogramme et fonction de répartition du sous-jacent à un pas choisi (axe cadré sur 99,9 % de la masse)."""
        levels = state["tree"].levels
        n = len(levels) - 1
        with Affichage.section(f"Distribution du sous-jacent",
                           "Loi risque-neutre du sous-jacent lue sur l'arbre au pas choisi."):
            t = st.slider("Pas de temps t étudié", 0, n, n, key="dist_t")
            cache_key = (id(state["tree"]), t)
            spot_prices = probabilities = cdf = edges = hi = lo = mass = mean = None

            def _calculer_distribution() -> None:
                """Met en cache les données de distribution du niveau."""
                nonlocal probabilities, spot_prices, cdf, edges, hi, lo, mass, mean
                if st.session_state.get("dist_key") != cache_key:
                    order = np.argsort(levels[t]["S"])
                    spot_prices, probabilities = levels[t]["S"][order], levels[t]["P"][order] / levels[t]["P"].sum()
                    cdf = np.cumsum(probabilities)
                    mean = float((spot_prices * probabilities).sum())
                    lo = spot_prices[min(int(np.searchsorted(cdf, 5e-4)), len(spot_prices) - 1)]
                    hi = spot_prices[min(int(np.searchsorted(cdf, 1 - 5e-4)), len(spot_prices) - 1)]
                    lo, hi = (lo - 1.0, hi + 1.0) if hi <= lo else (lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
                    inside = (spot_prices >= lo) & (spot_prices <= hi)
                    edges = np.linspace(lo, hi, 61)
                    mass = np.bincount(np.clip(np.searchsorted(edges, spot_prices[inside], side="right") - 1, 0, 59),
                                       weights=probabilities[inside], minlength=60)
                    st.session_state["dist"] = (spot_prices, probabilities, cdf, mean, lo, hi, edges, mass)
                    st.session_state["dist_key"] = cache_key
            _calculer_distribution()
            spot_prices, probabilities, cdf, mean, lo, hi, edges, mass = st.session_state["dist"]
            std_dev = math.sqrt(max((probabilities * (spot_prices - mean) ** 2).sum(), 0))
            st.caption(f"Espérance de l'arbre {mean:,.4f} · écart-type {std_dev:,.4f}"
                       f" · {len(spot_prices):,} nœuds")
            left, right = st.columns(2)
            fig: object

            def _histogramme() -> None:
                """Trace l’histogramme de la distribution."""
                nonlocal fig
                with left:
                    fig = Affichage.new_fig(f"Distribution au pas {t}", "Sous-jacent S", "Probabilité par intervalle")
                    fig.add_trace(go.Bar(x=0.5 * (edges[:-1] + edges[1:]), y=mass, width=0.92 * (edges[1] - edges[0]),
                                         marker_color=Affichage.COLORS[0]))
                    fig.add_vline(x=option.strike, line_dash="dash", line_color="grey", annotation_text="K")
                    fig.update_layout(hovermode="closest", showlegend=False)
                    fig.update_xaxes(range=[lo, hi])
                    Affichage.show_chart(fig, "dist_plot")
            _histogramme()
            fig: object

            def _repartition() -> None:
                """Trace la fonction de répartition du sous-jacent."""
                nonlocal fig
                with right:
                    fig = Affichage.new_fig(f"Fonction de répartition au pas {t}", "Sous-jacent S", "P(S ≤ x)")
                    fig.add_trace(go.Scatter(x=spot_prices, y=cdf, mode="lines", line_shape="hv",
                                             line=dict(color=Affichage.COLORS[1], width=2)))
                    fig.add_vline(x=option.strike, line_dash="dash", line_color="grey", annotation_text="K")
                    fig.update_layout(hovermode="closest", showlegend=False)
                    fig.update_xaxes(range=[lo, hi])
                    fig.update_yaxes(range=[0, 1.03], tickformat=".0%")
                    Affichage.show_chart(fig, "cdf_plot")
            _repartition()

    @staticmethod
    def grecques(market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        """Grecques de l'arbre (différences finies, GreeksTree) et Black-Scholes ; Delta / Gamma par 3 méthodes."""
        with Affichage.section("Grecques",
                               "Theta par jour de bourse (/252) ; Vega, Rho, Vanna pour 1 % ; Vomma pour 1 %². Gamma par choc "
                               "du spot : bruité à faible N (la grille se déplace avec S0)"):
            c_sel, c_meth = st.columns([3, 2], vertical_alignment="bottom")
            selected = c_sel.multiselect("Grecques", GREEKS, default=["Delta", "Gamma", "Theta"], key="greeks_sel")
            dg_method = c_meth.selectbox("Méthode pour Delta / Gamma", Orchestrator.DELTA_METHODS, key="greeks_dg")
            if not selected:
                return
            store = st.session_state.setdefault("res_greeks", {})
            missing = [g for g in selected if g not in store]
            if missing:
                with st.spinner("Calcul des grecques…"):
                    store.update(Orchestrator.greeks_tree(market, option, pricer, missing))
            values = {g: store[g] for g in selected}

            def _delta_gamma_alternatifs() -> None:
                """Récupère les sensibilités calculées avec la méthode sélectionnée."""
                if dg_method != Orchestrator.DELTA_METHODS[0] and {"Delta", "Gamma"} & set(selected):
                    dg_store = st.session_state.setdefault("res_dg_values", {})
                    if dg_method not in dg_store:
                        with st.spinner(f"Delta et Gamma via « {dg_method} »…"):
                            dg_store[dg_method] = Orchestrator.delta_gamma(market, option, pricer, dg_method)
                    values["Delta"], values["Gamma"] = dg_store[dg_method]
            _delta_gamma_alternatifs()

            def _tableau_grecques() -> None:
                """Présente les sensibilités et leur référence analytique."""
                bs = Orchestrator.greeks_bs(market, option, pricer, selected) or {}
                df = pd.DataFrame([{"Grecque": g, "Arbre": values[g], "Black-Scholes": bs.get(g, np.nan),
                                    "Écart (arbre - BS)": values[g] - bs.get(g, np.nan)} for g in selected])
                Affichage.show_df(df if bs else df[["Grecque", "Arbre"]],
                                  column_config={c: st.column_config.NumberColumn(format="%.6f") for c in df.columns[1:]})
            _tableau_grecques()

    @staticmethod
    def controles(market: Market, option: Option, pricer: PriceParameters, state: dict) -> None:
        """Rapport de débogage de l'arbre et contrôles de cohérence (parité call-put, Black-Scholes, forward,
        martingale, bornes d'arbitrage, américaine / européenne, moteurs), dans deux listes déroulantes."""
        with Affichage.section("Débogage et contrôles de cohérence",
                               "Ligne verte : contrôle validé · orange : écart faible et expliqué · rouge : échec · "
                               "N/A : contrôle non applicable à cette option."):
            saved = st.session_state.get("res_checks")                  # vidé à chaque nouveau pricing
            saved: object

            def _calculer_controles() -> None:
                """Conserve les contrôles déjà calculés pour cet arbre."""
                nonlocal saved
                if saved is None:
                    with st.spinner("Contrôles de cohérence…"):
                        saved = (Orchestrator.debug_report(market, option, pricer, state["tree"], state["build_s"],
                                                           state["price_s"]),
                                 Orchestrator.sanity_checks(market, option, pricer, state["tree"]))
                    st.session_state["res_checks"] = saved
            _calculer_controles()
            debug, checks = saved
            judged = [c for c in debug + checks if c["Verdict"] in ("Validé", "Attention", "Échec")]
            n_ok = sum(c["Verdict"] == "Validé" for c in judged)
            n_ko = sum(c["Verdict"] == "Échec" for c in judged)
            message = f"{n_ok} / {len(judged)} contrôles validés"

            def _message_controles() -> None:
                """Présente la synthèse des contrôles de cohérence."""
                if n_ko:
                    st.error(f"{message} · {n_ko} échec(s) : voir les lignes rouges.")
                elif n_ok < len(judged):
                    st.warning(f"{message} · les lignes orange signalent un écart faible et expliqué.")
                else:
                    st.success(message)
            _message_controles()
            with st.expander("Rapport de débogage de l'arbre", expanded=False):
                Affichage.verdict_table(debug)
            with st.expander("Contrôles de cohérence (sanity checks)", expanded=False):
                Affichage.verdict_table(checks)


# ====================================================================================
# Onglet 2 : analyse comparative
# ====================================================================================


class EtudeComparative:
    """Convergence (prix, grecques), prix selon le strike, 4 types d'options, grecques selon un paramètre."""

    METHODS = (TreeMethod.BINOMIAL, TreeMethod.TRINOMIAL)
    CONV_VIEWS = ("Erreur (arbre - référence)", "Erreur x N")

    @classmethod
    def render(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        Affichage.params_expander(market, option, pricer)
        if st.button("Lancer l'analyse comparative", type="primary", key="run_cmp"):
            st.session_state["cmp_run"] = key
        if st.session_state.get("cmp_run") != key:
            st.info(
                "Cliquez sur « Lancer l'analyse comparative » pour générer les graphiques "
                "(résultats gardés en mémoire ensuite)."
            )
            return
        with Affichage.safe("Convergence du prix"):
            cls.convergence_prix(market, option, pricer, key)
        with Affichage.safe("Convergence des grecques"):
            cls.convergence_grecques(market, option, pricer, key)
        with Affichage.safe("Prix et pente selon le strike"):
            cls.prix_selon_strike(market, option, pricer, key)
        with Affichage.safe("Types d'options"):
            cls.types_options(market, option, pricer, key)
        with Affichage.safe("Frontières d'exercice"):
            cls.frontieres_exercice(market, option, pricer, key)
        with Affichage.safe("Grecques selon un paramètre"):
            cls.grecques_selon_parametre(market, option, pricer, key)
        # Libérer la mémoire utilisée par les graphiques de comparaison.
        gc.collect()

    @classmethod
    def convergence_prix(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Prix binomial et trinomial selon N (1 à 100, puis 40 valeurs jusqu'au N maximal) : niveau, erreur, x N."""
        n_max = max(500, pricer.nb_steps)
        with Affichage.section(f"Convergence du prix : binomial vs trinomial – 1 à {n_max:,} pas",
                           "Trait plein : binomial ; tirets : "
                           "trinomial. Référence : Black-Scholes, sinon trinomial à 2 x N."):
            view = st.radio("Graphique", cls.CONV_VIEWS, horizontal=True, key=f"cmp_conv_view_{key}")
            saved = st.session_state.get("res_conv_price")
            df: object; ref: object; ref_label: object

            def _calculer_convergence_prix() -> None:
                """Calcule les références et met en cache les prix."""
                nonlocal df, ref, ref_label
                if saved is None or saved[0] != key:
                    with st.spinner("Convergence du prix…"):
                        grid = sorted(set(range(1, 101)) | {int(round(x)) for x in np.geomspace(100, n_max, 41)})
                        df = pd.DataFrame([{"N": n, **{m.label: Orchestrator.price(market, option, PriceParameters(
                            **{**vars(pricer), "nb_steps": n, "tree_method": m, "engine": "vector"}))
                            for m in cls.METHODS}} for n in grid])
                        ref, ref_label = Orchestrator.black_scholes(market, option, pricer), "Black-Scholes"
                        if ref is None:
                            n_ref = min(10_000, 2 * n_max)
                            ref = Orchestrator.price(market, option, PriceParameters(
                                **{**vars(pricer), "nb_steps": n_ref, "tree_method": TreeMethod.TRINOMIAL,
                                   "engine": "vector"}))
                            ref_label = f"trinomial N = {n_ref:,}"
                    st.session_state["res_conv_price"] = (key, df, ref, ref_label)
            _calculer_convergence_prix()
            _, df, ref, ref_label = st.session_state["res_conv_price"]
            scale = df["N"] if view == "Erreur x N" else 1
            y_title = "Erreur × N (arbre - référence)" if view == "Erreur x N" else "Erreur (arbre - référence)"
            fig = Affichage.new_fig(f"Convergence du prix vs {ref_label}", "Nombre de pas N", y_title, height=470)
            m: object

            def _tracer_convergence_prix() -> None:
                """Trace les erreurs de convergence des deux maillages."""
                nonlocal m
                for m in cls.METHODS:
                    color, dash = Affichage.METHOD_STYLE[m]
                    fig.add_trace(go.Scatter(x=df["N"], y=(df[m.label] - ref) * scale, mode="lines+markers",
                                             name=m.label, marker=dict(size=3),
                                             line=dict(color=color, width=1.6, dash=dash)))
                fig.add_hline(y=0, line_dash="dot", line_color="black")
                fig.update_xaxes(type="log")
                Affichage.show_chart(fig, f"cmp_conv_{key}")
            _tracer_convergence_prix()

    @classmethod
    def convergence_grecques(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Une grecque (GreeksTree) selon N, binomial et trinomial ; référence Black-Scholes si disponible."""
        bs = Orchestrator.greeks_bs(market, option, pricer)
        n_max = min(max(100, pricer.nb_steps), 10000 if pricer.pruning else pricer.max_unpruned_steps("vector"))
        with Affichage.section(f"Convergence des grecques : binomial vs trinomial – 3 à {n_max:,} pas",
                               "Erreur vs Black-Scholes ; « Erreur x N » pour l'ordre de convergence. "
                               "Si BS n'est pas disponible (américaine, dividende) : valeurs brutes."):
            c_g, c_v = st.columns([2, 3], vertical_alignment="bottom")
            greek = c_g.selectbox("Grecque", GREEKS, key=f"cmp_greek_conv_{key}")
            if bs:
                view = c_v.radio("Graphique", cls.CONV_VIEWS, horizontal=True, key=f"cmp_gconv_view_{key}")
            else:
                view = None
                c_v.caption("Black-Scholes indisponible : valeurs brutes affichées (pas de bascule erreur).")
            store = st.session_state.setdefault("res_conv_greeks", {})
            if store.get("key") != key:
                store.clear()
                store["key"] = key

            def _calculer_convergence_grecques() -> None:
                """Met en cache les grecques sur différentes grilles."""
                if greek not in store:
                    with st.spinner(f"Convergence de {greek}…"):
                        grid = sorted(set(range(3, min(100, n_max) + 1)) |
                                      ({int(round(x)) for x in np.geomspace(100, n_max, 31)} if n_max > 100 else set()))
                        store[greek] = {m: pd.DataFrame([{"N": n, greek: Orchestrator.greeks_tree(
                            market, option, PriceParameters(**{**vars(pricer), "nb_steps": n, "tree_method": m}),
                            (greek,))[greek]} for n in grid]) for m in cls.METHODS}
            _calculer_convergence_grecques()
            fig: object; m: object

            def _tracer_convergence_grecques() -> None:
                """Trace les différences avec Black-Scholes ou les valeurs brutes."""
                nonlocal fig, m
                if bs:
                    ref = bs[greek]
                    scale = next(iter(store[greek].values()))["N"] if view == "Erreur x N" else 1
                    y_title = f"{greek} : erreur × N vs BS" if view == "Erreur x N" else f"{greek} : erreur vs BS"
                    fig = Affichage.new_fig(f"Convergence de {greek} vs Black-Scholes", "Nombre de pas N", y_title, height=470)
                    for m, df in store[greek].items():
                        color, dash = Affichage.METHOD_STYLE[m]
                        fig.add_trace(go.Scatter(x=df["N"], y=(df[greek] - ref) * scale, mode="lines+markers",
                                                 name=m.label, marker=dict(size=3),
                                                 line=dict(color=color, width=1.6, dash=dash)))
                    fig.add_hline(y=0, line_dash="dot", line_color="black")
                else:
                    fig = Affichage.new_fig(f"Convergence de {greek} (valeurs brutes)", "Nombre de pas N", greek, height=470)
                    for m, df in store[greek].items():
                        color, dash = Affichage.METHOD_STYLE[m]
                        fig.add_trace(go.Scatter(x=df["N"], y=df[greek], mode="lines+markers",
                                                 name=m.label, marker=dict(size=3),
                                                 line=dict(color=color, width=1.6, dash=dash)))
            # Les courbes des sensibilités sont comparées sur une échelle logarithmique.
            _tracer_convergence_grecques()
            fig.update_xaxes(type="log")
            Affichage.show_chart(fig, f"cmp_gconv_{key}")

    @classmethod
    def prix_selon_strike(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Prix, pente dV/dK et écart à Black-Scholes pour 101 strikes valorisés sur la même grille."""
        bs_ok = BlackScholes.is_available(market, option)
        n_used = min(pricer.nb_steps, 10000)
        method = pricer.tree_method
        with Affichage.section(f"Prix et pente selon le strike – {n_used:,} pas ({method.label.lower()})",
                               "Pente d'un call : -e^(-rT)·P(S_T > K) ; d'un put : "
                               "e^(-rT)·P(S_T < K). Axe de droite : pente et écart arbre - BS."):
            strikes = np.linspace(0.5 * option.strike, 1.5 * option.strike, 101)
            saved = st.session_state.get("res_strike")
            curve: object

            def _calculer_strikes() -> None:
                """Calcule les prix de plusieurs strikes sur la même grille."""
                nonlocal curve
                if saved is None or saved[0] != key:
                    with st.spinner("Prix selon le strike…"):
                        prices = Orchestrator.price_many(market, option, PriceParameters(
                                    **{**vars(pricer), "nb_steps": n_used, "tree_method": method}), strikes,
                                    [option.call_put == "call"] * 101, [option.is_american()] * 101)
                        curve = pd.DataFrame({"K": strikes, "Prix": prices, "Pente": np.gradient(prices, strikes)})
                    st.session_state["res_strike"] = (key, curve)
            _calculer_strikes()
            curve = st.session_state["res_strike"][1]
            fig = Affichage.new_fig(
                f"Prix et pente selon le strike – {method.label}",
                "Strike K", "Prix de l'option", height=480,
            )
            color, _ = Affichage.METHOD_STYLE[method]
            fig.add_trace(go.Scatter(x=curve["K"], y=curve["Prix"], name=f"Prix - {method.label}", line=dict(color=color)))
            fig.add_trace(go.Scatter(x=curve["K"], y=curve["Pente"], name=f"Pente - {method.label}", yaxis="y2",
                                     line=dict(color=color, dash="dash")))

            def _ajouter_courbe_bs() -> None:
                """Ajoute les prix et dérivées analytiques à la figure."""
                if bs_ok:                                     # BS : prix fermé et pente analytique
                    maturity_years = PriceParameters.year_fraction(pricer.pricing_date, option.maturity)
                    bs_price = np.array([
                        BlackScholes.price_from(market.s0, k, maturity_years, market.rate, market.vol, option.call_put)
                        for k in strikes
                    ])
                    d2 = ((np.log(market.s0 / strikes)
                           + (market.rate - 0.5 * market.vol ** 2) * maturity_years)
                          / (market.vol * math.sqrt(maturity_years)))
                    disc = math.exp(-market.rate * maturity_years)
                    slope = -disc * norm.cdf(d2) if option.call_put == "call" else disc * norm.cdf(-d2)
                    fig.add_trace(go.Scatter(x=strikes, y=bs_price, name="Prix - BS", line=dict(color="black", width=1)))
                    fig.add_trace(go.Scatter(x=strikes, y=slope, name="Pente - BS", yaxis="y2",
                                             line=dict(color="black", width=1, dash="dash")))
                    fig.add_trace(go.Scatter(x=strikes, y=curve["Prix"] - bs_price, name=f"Écart arbre - BS - {method.label}",
                                             yaxis="y2", line=dict(color=color, dash="dot")))
            _ajouter_courbe_bs()
            fig.update_layout(yaxis2=dict(title="Pente et écart", overlaying="y", side="right", showgrid=False))
            fig.add_vline(x=market.s0, line_dash="dash", line_color="grey", annotation_text="S0")
            Affichage.show_chart(fig, f"cmp_strike_{key}")

    @staticmethod
    def types_options(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Call / put x européen / américain quand le dividende, sa date, la maturité ou le taux varie."""
        n_used = min(pricer.nb_steps, 10000)
        with Affichage.section(f"Call / put, américain / européen – {n_used:,} pas",
                               "Les quatre options sur la même grille (axe de gauche) ; l'écart américaine − "
                               "européenne (axe de droite) mesure la valeur de l'exercice anticipé."):
            c1, c2 = st.columns([3, 1.4], vertical_alignment="bottom")
            which = c1.radio("Paramètre qui varie", ["Dividende", "Date ex-dividende", "Maturité", "Taux"],
                             horizontal=True, key=f"cmp_types_{key}")
            zero_rate = c2.checkbox("Taux r = 0", value=which != "Taux", disabled=which == "Taux",
                                    key=f"cmp_types_r0_{key}_{which}",
                                    help="Isole l'effet du dividende de celui du taux.")
            m = Market(**{**vars(market), "rate": 0.0}) if zero_rate and which != "Taux" else market
            saved = st.session_state.get("res_types")

            def _mettre_a_jour_types() -> None:
                """Réutilise les quatre grilles de prix calculées pour ce scénario."""
                if saved is None or saved[0] != (key, which, zero_rate):
                    with st.spinner("Calcul des 4 types d'options…"):
                        days = (option.maturity - pricer.pricing_date).days
                        div_date = (m.ex_div_date if m.ex_div_date and 0 < (m.ex_div_date - pricer.pricing_date).days <= days
                                    else pricer.pricing_date + dt.timedelta(days=days // 2))
                        div = m.dividend if m.dividend > 0 else round(0.03 * m.s0, 2)
                        cases: object; xs: object

                        def _choisir_scenarios() -> None:
                            """Construit les scénarios de comparaison selon le paramètre choisi."""
                            nonlocal cases, xs
                            if which == "Dividende":
                                xs = list(np.linspace(0.0, 0.15 * m.s0, 13))
                                cases = [(Market(m.s0, m.rate, m.vol, x, div_date), option) for x in xs]
                            elif which == "Date ex-dividende":
                                xs = [pricer.pricing_date + dt.timedelta(days=int(d)) for d in np.linspace(1, days, 13)]
                                cases = [(Market(m.s0, m.rate, m.vol, div, x), option) for x in xs]
                            elif which == "Maturité":
                                xs = [pricer.pricing_date + dt.timedelta(days=max(1, int(d)))
                                      for d in np.linspace(0.1 * days, 2 * days, 13)]
                                cases = [(m, Option(option.strike, x, option.call_put, option.eur_am)) for x in xs]
                            else:
                                xs = list(np.linspace(-0.01, 0.10, 12))
                                cases = [(Market(m.s0, x, m.vol, m.dividend, m.ex_div_date), option) for x in xs]
                        _choisir_scenarios()
                        p = PriceParameters(**{**vars(pricer), "nb_steps": n_used})
                        # Un même scénario sert aux quatre modalités d'exercice.
                        labels = ["Call européen", "Call américain", "Put européen", "Put américain"]
                        df = pd.DataFrame([{"x": x, **dict(zip(labels, Orchestrator.price_many(
                            mk, op, p, [op.strike] * 4, [True, True, False, False], [False, True, False, True])))}
                            for x, (mk, op) in zip(xs, cases)])
                    st.session_state["res_types"] = ((key, which, zero_rate), df)
            _mettre_a_jour_types()
            df = st.session_state["res_types"][1]
            title = f"Prix et écart américain − européen selon : {which.lower()}"
            if m.rate == 0 and which != "Taux":
                title += " (r = 0)"
            fig = Affichage.new_fig(title, which, "Prix")

            def _tracer_types() -> None:
                """Trace les prix et leurs écarts américain-européen."""
                for col in df.columns[1:]:
                    fig.add_trace(go.Scatter(x=df["x"], y=df[col], mode="lines+markers", name=col,
                                             line=dict(color=Affichage.COLORS[0 if col.startswith("Call") else 1],
                                                       dash="solid" if "américain" in col else "dot")))
                for color_idx, kind in ((2, "Call"), (4, "Put")):
                    fig.add_trace(go.Scatter(x=df["x"], y=df[f"{kind} américain"] - df[f"{kind} européen"],
                                             mode="lines+markers", name=f"{kind} : US − EU", yaxis="y2",
                                             line=dict(color=Affichage.COLORS[color_idx], dash="dash")))
            _tracer_types()
            fig.update_layout(yaxis2=dict(title="Écart américain − européen", overlaying="y", side="right",
                                          showgrid=False, tickformat="~g"))
            Affichage.show_chart(fig, f"cmp_types_plot_{key}")

    @staticmethod
    def frontieres_exercice(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Frontières d'exercice du put et du call américains pour des scénarios de taux et de dividende fixés."""
        n_used = min(pricer.nb_steps, 10000)
        # date ex-dividende effective : celle de l'utilisateur, sinon une date de repli (pricing + 6 mois)
        div_date = market.ex_div_date or (pricer.pricing_date + dt.timedelta(days=182))
        div = f"D = 3 au {div_date:%d/%m/%Y}"
        # nom : (taux, dividende, couleur, tirets)
        cases = {"put": {"Put, r = 1 %, sans dividende": (0.01, 0.0, "#9ecae1", "solid"),
                         "Put, r = 3 %, sans dividende": (0.03, 0.0, "#4292c6", "solid"),
                         "Put, r = 5 %, sans dividende": (0.05, 0.0, "#08519c", "solid"),
                         f"Put, r = 3 %, {div}": (0.03, 3.0, "#d62728", "dash")},
                 "call": {"Call, r = −2 %, sans dividende": (-0.02, 0.0, "#2ca02c", "solid"),
                          f"Call, r = −2 %, {div}": (-0.02, 3.0, "#98df8a", "solid"),
                          f"Call, r = 3 %, {div}": (0.03, 3.0, "#ff7f0e", "solid")}}
        # marché passé à l'orchestrateur : on garantit une ex_div_date pour que les scénarios "avec dividende" en aient une
        mkt = market if market.ex_div_date is not None else Market(
            market.s0, market.rate, market.vol, market.dividend, div_date)
        with Affichage.section(f"Frontières d'exercice – {n_used:,} pas",
                               "Put : exercer si S ≤ S* ; call : exercer si S ≥ S*. Taux et dividendes fixés, "
                               "autres paramètres de marché saisis dans la barre latérale."):
            saved = st.session_state.get("res_frontieres")
            results: object

            def _calculer_frontieres() -> None:
                """Met en cache les frontières pour chaque scénario."""
                nonlocal results
                if saved is None or saved[0] != key:
                    with st.spinner("Frontières d'exercice…"):
                        params = PriceParameters(**{**vars(pricer), "nb_steps": n_used})
                        results = {kind: Orchestrator.exercise_frontier_scenarios(
                                       mkt, Option(option.strike, option.maturity, kind, "americaine"),
                                       params,
                                       {name: (r, d) for name, (r, d, _, _) in scenarios.items()})
                                   for kind, scenarios in cases.items()}
                    st.session_state["res_frontieres"] = (key, results)
            _calculer_frontieres()
            results = st.session_state["res_frontieres"][1]
            d: object; kind: object; name: object; r: object; scenarios: object

            def _tracer_frontieres() -> None:
                """Trace les frontières de l’exercice anticipé."""
                nonlocal d, kind, name, r, scenarios
                for col, (kind, scenarios) in zip(st.columns(2), cases.items()):
                    frontiers, div_time = results[kind]
                    fig = Affichage.new_fig(f"Frontière d'exercice du {kind}", "t (années)",
                                            f"S* (exercer si S {'≤' if kind == 'put' else '≥'} S*)")
                    for name, (r, d, color, dash) in scenarios.items():
                        points = frontiers[name]
                        if points:
                            fig.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], name=name,
                                                     mode="lines+markers" if len(points) < 5 else "lines",
                                                     line=dict(color=color, width=2, dash=dash, shape="hv"),
                                                     marker=dict(size=9)))
                    fig.add_hline(y=option.strike, line_dash="dot", line_color="black")
                    if kind == "call" and div_time:
                        fig.add_vline(x=div_time, line_dash="dash", line_color="grey")
                    with col:
                        Affichage.show_chart(fig, f"frontiere_{kind}_{key}")
                        empty = [name for name, points in frontiers.items() if not points]
                        if empty:
                            st.caption("Pas d'exercice anticipé : " + " ; ".join(empty))
            _tracer_frontieres()

    @staticmethod
    def grecques_selon_parametre(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Grecques de l'arbre (trait plein) et Black-Scholes (pointillés) quand S0, r, T ou σ varie."""
        n_used = min(pricer.nb_steps, 1000)
        with Affichage.section(f"Grecques selon un paramètre, comparées à Black-Scholes – {n_used:,} pas", ""):
            c_p, c_g, c_n = st.columns([2, 3, 1.3], vertical_alignment="bottom")
            which = c_p.radio("Paramètre qui varie", ["S0", "r", "T", "σ"], horizontal=True, key=f"cmp_gparam_{key}")
            greeks = c_g.multiselect("Grecques", GREEKS, default=["Delta", "Gamma", "Vega"], key=f"cmp_gparam_sel_{key}")
            normalize = c_n.checkbox("Normaliser", value=True, key=f"cmp_norm_{key}",
                                     help=(
                                         "Divise chaque grecque par son max |arbre| : rend comparables "
                                         "des grecques d'ordres de grandeur différents."
                                     ))
            saved = st.session_state.get("res_gparam")

            def _mettre_a_jour_grecques() -> None:
                """Réutilise les sensibilités calculées pour le paramètre choisi."""
                if saved is None or saved[0] != (key, which):
                    with st.spinner("Grecques générées par l'arbre…"):
                        days = (option.maturity - pricer.pricing_date).days
                        cases: object; xs: object

                        def _choisir_gamme() -> None:
                            """Construit la gamme de données de marché à étudier."""
                            nonlocal cases, xs
                            if which == "S0":
                                xs = list(np.linspace(0.6 * market.s0, 1.4 * market.s0, 25))
                                cases = [(Market(**{**vars(market), "s0": x}), option) for x in xs]
                            elif which == "r":
                                xs = list(np.linspace(-0.01, 0.10, 25))
                                cases = [(Market(**{**vars(market), "rate": x}), option) for x in xs]
                            elif which == "T":
                                ds = sorted({max(2, int(d)) for d in np.linspace(0.1 * days, 2 * days, 25)})
                                xs = [d / 365.0 for d in ds]
                                cases = [
                                    (market, Option(option.strike, pricer.pricing_date + dt.timedelta(days=d),
                                                    option.call_put, option.eur_am))
                                    for d in ds
                                ]
                            else:
                                xs = list(np.linspace(0.05, 0.60, 25))
                                cases = [(Market(**{**vars(market), "vol": x}), option) for x in xs]
                        _choisir_gamme()
                        p = PriceParameters(**{**vars(pricer), "nb_steps": n_used})
                        rows = []
                        # Associer les grecques de l'arbre à leurs références analytiques.
                        for x, (mk, op) in zip(xs, cases):
                            bs = Orchestrator.greeks_bs(mk, op, p) or {}
                            rows.append({
                                "x": x, **Orchestrator.greeks_tree(mk, op, p),
                                **{f"BS {g}": v for g, v in bs.items()},
                            })
                    st.session_state["res_gparam"] = ((key, which), pd.DataFrame(rows))
            _mettre_a_jour_grecques()
            df = st.session_state["res_gparam"][1]
            fig = Affichage.new_fig(f"Grecques selon {which}", which, "Grecque normalisée" if normalize else "Valeur")
            g: object

            def _tracer_grecques_parametre() -> None:
                """Trace les grecques de l’arbre et les références analytiques."""
                nonlocal g
                for g in greeks:
                    color = Affichage.COLORS[GREEKS.index(g) % 7]
                    scale = (float(np.nanmax(np.abs(df[g]))) or 1.0) if normalize else 1.0
                    fig.add_trace(go.Scatter(x=df["x"], y=df[g] / scale, mode="lines+markers", name=f"{g} – arbre",
                                             line=dict(color=color)))
                    if f"BS {g}" in df:
                        fig.add_trace(go.Scatter(x=df["x"], y=df[f"BS {g}"] / scale, name=f"{g} – BS",
                                                 line=dict(color=color, dash="dot")))
            _tracer_grecques_parametre()
            Affichage.show_chart(fig, f"cmp_gparam_plot_{key}")


# ====================================================================================
# Onglet 3 : analyse des performances
# ====================================================================================


class EtudePerformances:
    """Temps, mémoire, backward / récursif / vectorisé, méthodes Delta / Gamma, erreurs dues au pruning."""

    PMINS = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]

    @classmethod
    def render(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        Affichage.params_expander(market, option, pricer)
        if st.button("Lancer l'analyse des performances", type="primary", key="run_perf"):
            st.session_state["perf_run"] = key
        if st.session_state.get("perf_run") != key:
            st.info("Cliquez sur « Lancer l'analyse des performances » (de quelques secondes à quelques minutes).")
            return
        with Affichage.safe("Temps d'exécution"):
            cls.temps_execution(market, option, pricer, key)
        with Affichage.safe("Mémoire"):
            cls.memoire(market, option, pricer, key)
        with Affichage.safe("Backward, récursif et vectorisé"):
            cls.backward_recursif_vectorise(market, option, pricer, key)
        with Affichage.safe("Méthodes Delta / Gamma"):
            cls.methodes_delta_gamma(market, option, pricer, key)
        with Affichage.safe("Erreurs dues au pruning"):
            cls.erreurs_pruning(market, option, pricer, key)
        gc.collect()

    @staticmethod
    def temps_execution(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Temps total (construction + valorisation) : moteur vectorisé vs objets, pruning ON / OFF."""
        n_max = min(max(1000, pricer.nb_steps), 10000)
        grid = sorted({n for n in (50, 100, 200, 500, 1000, 2000, 5000, 10000) if n <= n_max} | {pricer.nb_steps})
        with Affichage.section(f"Temps d'exécution : vectorisé vs objets, avec et sans pruning – 50 à {grid[-1]:,} pas",
                               "Trait plein : pruning ON ; pointillés : OFF. Les deux moteurs construisent le même arbre."):
            saved = st.session_state.get("res_time")
            if saved is None or saved[0] != key:
                n_max = min(max(1000, pricer.nb_steps), 10_000)
                grid = sorted({n for n in (50, 100, 200, 500, 1000, 2000, 5000, 10000) if n <= n_max} | {pricer.nb_steps})
                frames = {}
                with st.spinner("Mesure des temps (les deux moteurs)…"):

                    def _mesurer_moteurs() -> None:
                        """Mesure les deux moteurs avec et sans élagage."""
                        for engine, name in (("vector", "Vectorisé"), ("objects", "Objets")):
                            for prune in (True, False):
                                p = PriceParameters(**{**vars(pricer), "engine": engine, "pruning": prune})
                                frames[f"{name} – {'ON' if prune else 'OFF'}"] = pd.DataFrame([
                                    Orchestrator.benchmark(market, option, PriceParameters(**{**vars(p), "nb_steps": n}))
                                    for n in grid if prune or n <= p.max_unpruned_steps()])
                    _mesurer_moteurs()
                st.session_state["res_time"] = (key, frames)
            frames = st.session_state["res_time"][1]
            Affichage.lines({k: (v, "Total (s)", "solid" if k.endswith("ON") else "dot") for k, v in frames.items()},
                            "Temps total selon N", "Temps (s)", "perf_time", log_x=True, log_y=True)

    @staticmethod
    def memoire(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Pic de mémoire Python (tracemalloc) et nombre de nœuds : pruning ON vs OFF, N = 10 → 1000."""
        with Affichage.section("Mémoire et nœuds conservés : pruning ON vs OFF – 10 à 1000 pas", ""):
            saved = st.session_state.get("res_memory")
            if saved is None or saved[0] != key:
                with st.spinner("Mesures mémoire…"):
                    frames = {prune: pd.DataFrame([Orchestrator.benchmark(market, option, PriceParameters(
                        **{**vars(pricer), "nb_steps": n, "pruning": prune}), memory=True) for n in range(10, 1001, 10)])
                        for prune in (True, False)}
                st.session_state["res_memory"] = (key, frames)
            on, off = st.session_state["res_memory"][1][True], st.session_state["res_memory"][1][False]
            c1, c2 = st.columns(2)
            with c1:
                Affichage.lines({
                    "Pruning ON": (on, "Pic mémoire (Mo)", "solid"),
                    "Pruning OFF": (off, "Pic mémoire (Mo)", "dot"),
                },
                                "Pic de mémoire Python", "Mo", "perf_mem")
            with c2:
                Affichage.lines({"Pruning ON": (on, "Nœuds", "solid"), "Pruning OFF": (off, "Nœuds", "dot")},
                                "Nœuds conservés", "Nombre de nœuds", "perf_nodes")

    @staticmethod
    def backward_recursif_vectorise(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Valorisation seule : backward et récursif (même arbre de Node) contre vectorisé (numpy)."""
        n_max = min(max(1000, pricer.nb_steps), 10000 if pricer.pruning else pricer.max_unpruned_steps("objects"))
        grid = [n for n in (50, 100, 250, 500, 1000, 2000, 5000, 10000) if n <= n_max]
        grid_hi = max(grid) if grid else n_max
        with Affichage.section(
            f"Valorisation : backward (objets) vs récursif (objets) "
            f"vs vectorisé (numpy) – 50 à {grid_hi:,} pas", "",
        ):
            saved = st.session_state.get("res_br")
            if saved is None or saved[0] != key:
                n_max = min(max(1000, pricer.nb_steps), 5000 if pricer.pruning else pricer.max_unpruned_steps("objects"))
                rows = []
                with st.spinner("Backward, récursif et vectorisé…"):
                    n: object

                    def _mesurer_valorisations() -> None:
                        """Mesure les variantes de valorisation sur une grille de pas."""
                        nonlocal n
                        for n in (n for n in (50, 100, 250, 500, 1000, 2000, 5000, 10000) if n <= n_max):
                            obj = Orchestrator.benchmark(market, option, PriceParameters(
                                **{**vars(pricer), "nb_steps": n, "engine": "objects"}), with_recursive=True)
                            vec = Orchestrator.benchmark(market, option, PriceParameters(
                                **{**vars(pricer), "nb_steps": n, "engine": "vector"}))
                            rows.append({"N": n, "Backward (s)": obj["Valorisation (s)"], "Récursif (s)": obj["Récursif (s)"],
                                         "Vectorisé (s)": vec["Valorisation (s)"]})
                    _mesurer_valorisations()
                st.session_state["res_br"] = (key, pd.DataFrame(rows))
            br = st.session_state["res_br"][1]
            Affichage.lines({
                "Backward (objets)": (br, "Backward (s)", "solid"),
                "Récursif (objets)": (br, "Récursif (s)", "dot"),
                             "Vectorisé (numpy)": (br, "Vectorisé (s)", "dash")},
                            "Temps de valorisation selon N", "Temps (s)", "perf_br", log_x=True, log_y=True)

    @staticmethod
    def methodes_delta_gamma(market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Delta et Gamma par les trois méthodes du cours : temps de calcul et écart à Black-Scholes."""
        bs = Orchestrator.greeks_bs(market, option, pricer, ("Delta", "Gamma"))
        with Affichage.section("Delta et Gamma : trois méthodes de calcul – 25 à 1 000 pas", ""):
            saved = st.session_state.get("res_dg")
            if saved is None or saved[0] != key:
                rows = []
                with st.spinner("Delta et Gamma par les trois méthodes…"):
                    m: object

                    def _mesurer_delta_gamma() -> None:
                        """Mesure les trois méthodes de Delta et Gamma."""
                        nonlocal m
                        for n in (25, 50, 100, 200, 400, 700, 1000):
                            for m in Orchestrator.DELTA_METHODS:
                                t0 = time.perf_counter()
                                delta, gamma = Orchestrator.delta_gamma(market, option, PriceParameters(
                                    **{**vars(pricer), "nb_steps": n}), m)
                                rows.append({"N": n, "Méthode": m, "Delta": delta, "Gamma": gamma,
                                             "Temps (s)": time.perf_counter() - t0})
                    _mesurer_delta_gamma()
                st.session_state["res_dg"] = (key, pd.DataFrame(rows))
            gm = st.session_state["res_dg"][1]
            views = ["Temps de calcul", "Delta selon N", "Gamma selon N"]
            view = st.radio("Graphique", views, horizontal=True, key="perf_dg_view")
            col = {"Temps de calcul": "Temps (s)", "Delta selon N": "Delta", "Gamma selon N": "Gamma"}[view]
            Affichage.lines({
                m: (gm[gm["Méthode"] == m], col, d)
                for m, d in zip(Orchestrator.DELTA_METHODS, ("solid", "dot", "dash"))
            },
                            view, col, f"perf_dg_{key}", log_x=True, log_y=view == "Temps de calcul")
            if bs:
                st.caption(f"Black-Scholes : Delta {bs['Delta']:.6f}, Gamma {bs['Gamma']:.6f}.")

    @classmethod
    def erreurs_pruning(cls, market: Market, option: Option, pricer: PriceParameters, key: str) -> None:
        """Heatmaps p_min × N : erreur de prix, erreur de distribution terminale (Kolmogorov-Smirnov), nœuds conservés."""
        with Affichage.section("Erreur due au pruning selon p_min et N – 25 à 1000 pas",
                               "Erreur = |avec pruning − sans pruning| à N identique (couleur en log10)."):
            saved = st.session_state.get("res_prune")
            if saved is None or saved[0] != key:
                rows = []
                with st.spinner("Erreurs dues au pruning…"):

                    def _calculer_erreurs_pruning() -> None:
                        """Compare les lois terminales avec et sans élagage."""
                        for n in (25, 50, 100, 200, 400, 700, 1000):
                            ref = Orchestrator.vector_tree(market, option, PriceParameters(
                                **{**vars(pricer), "nb_steps": n, "pruning": False})).levels
                            for p_min in cls.PMINS:
                                lv = Orchestrator.vector_tree(market, option, PriceParameters(
                                    **{**vars(pricer), "nb_steps": n, "pruning": True, "p_min": p_min})).levels
                                grid = np.union1d(lv[-1]["S"], ref[-1]["S"])          # distance KS des lois terminales
                                cdfs = []
                                for level in (lv[-1], ref[-1]):
                                    order = np.argsort(level["S"])
                                    idx = np.searchsorted(level["S"][order], grid, side="right") - 1
                                    cum = np.cumsum(level["P"][order])
                                    cdfs.append(np.where(idx >= 0, cum[np.clip(idx, 0, None)], 0.0))
                                rows.append({"N": n, "p_min": p_min, "Erreur de prix": abs(lv[0]["V"][0] - ref[0]["V"][0]),
                                             "Erreur de distribution": float(np.max(np.abs(cdfs[0] - cdfs[1]))),
                                             "Nœuds conservés": sum(len(level["S"]) for level in lv)
                                             / sum(len(level["S"]) for level in ref)})
                    _calculer_erreurs_pruning()
                st.session_state["res_prune"] = (key, pd.DataFrame(rows))
            err = st.session_state["res_prune"][1]
            c1, c2, c3 = st.columns(3)
            with c1:
                Affichage.error_heatmap(err, "Erreur de prix", "Erreur de prix", f"perf_prune_price_{key}")
            with c2:
                Affichage.error_heatmap(
                    err, "Erreur de distribution", "Erreur de distribution terminale (KS)",
                    f"perf_prune_dist_{key}",
                )
            with c3:
                Affichage.error_heatmap(err, "Nœuds conservés", "Nœuds conservés (part de l'arbre complet)",
                                        f"perf_prune_nodes_{key}", log=False)
