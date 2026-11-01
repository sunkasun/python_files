from __future__ import annotations

import gc
import math
import time
import tracemalloc

import numpy as np
from scipy.stats import norm

from black_scholes import BlackScholes, GreeksBS
from level import LevelVectorized
from market import Market
from option import Option
from option_pricer import OptionPricerTree
from price_parameters import PriceParameters, TreeMethod
from sensibilities import GREEKS, GreeksTree
from tree import Tree, TreeNode, TreeVectorized


class Orchestrator:
    """Construit, valorise et libère les arbres ; relie pricing, Black-Scholes et grecques."""
    @staticmethod
    def build(market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> Tree:
        """Arbre du moteur demandé : TreeNode (objets Node) ou TreeVectorized (tableaux numpy)."""
        tree_class = TreeNode if pricer.engine == "objects" else TreeVectorized
        return tree_class(market, option, pricer, pre_steps)

    @staticmethod
    def price(market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> float:
        """Prix de l'arbre (backward, ou récursif si demandé en moteur objets) ; l'arbre est libéré aussitôt."""
        tree = Orchestrator.build(market, option, pricer, pre_steps)
        try:
            return OptionPricerTree(tree).price(pricer.pricing_method if pricer.engine == "objects" else "backward")
        finally:
            tree.release()

    @staticmethod
    def pricing_report(market: Market, option: Option, pricer: PriceParameters) -> dict:
        """Pricing principal (bouton « Lancer le pricer ») : prix, temps, nombre de nœuds, et l'arbre vectorisé
        valorisé gardé pour les analyses (en moteur objets, l'arbre de Node est libéré et l'arbre vectorisé,
        identique, est construit pour l'affichage)."""
        t0 = time.perf_counter()
        tree = Orchestrator.build(market, option, pricer)
        t1 = time.perf_counter()
        pricer_tree = OptionPricerTree(tree)
        price = pricer_tree.price(pricer.pricing_method if pricer.engine == "objects" else "backward")
        t2 = time.perf_counter()
        if isinstance(tree, TreeNode):
            tree.release()
            tree = Orchestrator.vector_tree(market, option, pricer)
        return {"price": price, "nodes": pricer_tree.nodes_priced, "build_s": t1 - t0, "price_s": t2 - t1,
                "tree": tree, "bs": Orchestrator.black_scholes(market, option, pricer)}

    @staticmethod
    def vector_tree(market: Market, option: Option, pricer: PriceParameters, pre_steps: int = 0) -> TreeVectorized:
        """Arbre vectorisé valorisé, gardé pour les analyses (distribution, dessin de l'arbre, Delta / Gamma).

        C'est le même arbre qu'en moteur objets ; seule la représentation change. L'appelant le libère.
        """
        tree = TreeVectorized(market, option, PriceParameters(**{**vars(pricer), "engine": "vector"}), pre_steps)
        OptionPricerTree(tree).backward_pricing()
        return tree

    @staticmethod
    def price_many(market: Market, option: Option, pricer: PriceParameters, strikes, is_call,
                   is_american) -> np.ndarray:
        """Plusieurs calls / puts valorisés sur une seule grille vectorisée (une colonne par option)."""
        tree = TreeVectorized(market, option, PriceParameters(**{**vars(pricer), "engine": "vector"}))
        try:
            return OptionPricerTree(tree).price_options_vectorized(strikes, is_call, is_american)
        finally:
            tree.release()

    @staticmethod
    def black_scholes(market: Market, option: Option, pricer: PriceParameters) -> float | None:
        """Prix Black-Scholes, ou None si la formule ne s'applique pas (américaine ou dividende)."""
        if not BlackScholes.is_available(market, option):
            return None
        return BlackScholes(pricer, market, option).pricing_bs()

    @staticmethod
    def greeks_tree(market: Market, option: Option, pricer: PriceParameters, greeks=GREEKS,
                    spot_shift: float = 0.01, vol_shift: float = 0.01, rate_shift: float = 0.01,
                    theta_days: int = 1) -> dict:
        """Grecques de l'arbre par différences finies : GreeksTree rappelle Orchestrator.price sur des paramètres
        choqués (arbres vectorisés, plus rapides, même résultat)."""
        vector_pricer = PriceParameters(**{**vars(pricer), "engine": "vector"})
        gt = GreeksTree(vector_pricer, market, option, Orchestrator.price, spot_shift, vol_shift, rate_shift,
                        theta_days)
        return {g: float(getattr(gt, g.lower())()) for g in greeks}

    @staticmethod
    def greeks_bs(market: Market, option: Option, pricer: PriceParameters, greeks: tuple[str, ...] = GREEKS) -> dict | None:
        """Grecques Black-Scholes, ou None si la formule ne s'applique pas."""
        if not BlackScholes.is_available(market, option):
            return None
        return GreeksBS(pricer, market, option).compute(greeks)

    DELTA_METHODS = ("Choc du spot", "Nœuds de l'arbre", "Nœuds autour de la racine")

    @staticmethod
    def delta_gamma(market: Market, option: Option, pricer: PriceParameters, method: str) -> tuple[float, float]:
        """Delta et Gamma par les trois méthodes du cours.

        * Choc du spot : GreeksTree (arbre revalorisé à S0 ± 1 %).
        * Nœuds de l'arbre : 3 nœuds du premier niveau qui en compte 3, sur un seul arbre.
        * Nœuds autour de la racine : arbre démarré 1 pas (trinomial) ou 2 pas (binomial) avant la date de pricing,
          qui a donc 3 nœuds à la date de pricing, S0 au milieu.
        Delta et Gamma sont lus sur 3 nœuds voisins (pas inégaux), Delta ramené en S0 par Taylor.
        """
        if method == "Choc du spot":
            g = Orchestrator.greeks_tree(market, option, pricer, ("Delta", "Gamma"))
            return g["Delta"], g["Gamma"]
        pre = (1 if pricer.tree_method.nb_nodes == 3 else 2) if method == "Nœuds autour de la racine" else 0
        tree = Orchestrator.vector_tree(market, option, pricer, pre)
        try:
            levels = tree.levels
            level = levels[pre] if pre else next((lv for lv in levels if len(lv["S"]) >= 3), None)
            if level is not None and len(level["S"]) >= 3:
                order = np.argsort(level["S"])
                spot_prices, option_values = level["S"][order], level["V"][order]
                j = int(np.clip(np.argmin(np.abs(spot_prices - market.s0)), 1, len(spot_prices) - 2))
                (s0, s1, s2), (v0, v1, v2) = spot_prices[j - 1:j + 2], option_values[j - 1:j + 2]
                if s0 < s1 < s2:
                    gamma = 2.0 * ((v2 - v1) / (s2 - s1) - (v1 - v0) / (s1 - s0)) / (s2 - s0)
                    delta = (v2 - v0) / (s2 - s0) + gamma * (market.s0 - s1)
                    return float(delta), float(gamma)
        finally:
            tree.release()
        # Un arbre plat ou trop court ne permet pas de calculer trois pentes nodales.
        return Orchestrator.delta_gamma(market, option, pricer, "Choc du spot")

    # ------------------------------------------------------------------------------------------------------------
    # Débogage et contrôles de cohérence (sanity checks)
    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def debug_report(market: Market, option: Option, pricer: PriceParameters, tree: TreeVectorized,
                     build_s: float, price_s: float) -> list[dict]:
        """Rapport de débogage de l'arbre construit : paramètres de grille, taille, recombinaison, probabilités.

        Chaque ligne : Métrique, Valeur, Critère, Verdict (« Validé », « Attention », « Échec » ou « — » si la
        ligne est une information). Les transitions sont relues niveau par niveau (même calcul que la construction).
        """
        n, levels, steps = tree.n_steps, tree.levels, tree._steps
        tri_steps = sum(1 for c in steps[1:] if c["tri"])
        widths = [len(lv["S"]) for lv in levels]
        branches, merges, singles, clipped, clipped_mass, worst_proba, rel_var = (
            Orchestrator._debug_counts(tree, market, widths)
        )
        ratios = [widths[i] / widths[i - 1] for i in range(3, n + 1) if i != tree.div_step]
        growth = max(ratios) if ratios else 1.0
        sum_p = float(levels[-1]["P"].sum())
        full = (n + 1) ** 2 if tree.nb_nodes == 3 else (n + 1) * (n + 2) // 2
        mode = ("Trinomial" if tree.nb_nodes == 3 else
                ("Binomial (pas du dividende trinomial)" if tree.div_step else "Binomial"))
        div_text = Orchestrator._debug_dividend(market, pricer, tree)
        rows = []
        rows.extend(Orchestrator._debug_rows_intro(mode, n, tree, div_text, pricer, singles))
        rows.extend(Orchestrator._debug_rows_structure(widths, full, tri_steps, n, merges, branches, growth))
        rows.extend(Orchestrator._debug_rows_quality(worst_proba, clipped, clipped_mass, rel_var, sum_p, build_s, price_s))
        return rows

    @staticmethod
    def sanity_checks(market: Market, option: Option, pricer: PriceParameters, tree: TreeVectorized) -> list[dict]:
        """Contrôles de cohérence du prix : Black-Scholes, parité call-put, forward, martingale, bornes d'arbitrage,
        américaine / européenne, égalité des moteurs.

        Chaque ligne : Contrôle, Valeur, Tolérance, Verdict (« Validé », « Échec », « Attention » ou « N/A »).
        Les prix complémentaires (call / put, européenne / américaine) sont calculés sur la même grille de pas.
        """
        n, levels, r, maturity_years = tree.n_steps, tree.levels, market.rate, tree.dt_ * tree.n_steps
        discount_at_maturity = math.exp(-r * maturity_years)
        t_div = tree.div_step * tree.dt_ if tree.div_step else None
        pv_div = market.dividend * math.exp(-r * t_div) if t_div is not None else 0.0
        s_adj = market.s0 - pv_div
        vector = PriceParameters(**{**vars(pricer), "engine": "vector"})
        prices = {(option.call_put, option.eur_am.value): float(levels[0]["V"][0])}
        for kind, ex in (("call", "europeen"), ("put", "europeen"), (option.call_put, "americaine"),
                         (option.call_put, "europeen")):
            if (kind, ex) not in prices:
                prices[(kind, ex)] = Orchestrator.price(market, Option(option.strike, option.maturity, kind, ex), vector)
        price, euro, amer = prices[(option.call_put, option.eur_am.value)], prices[(option.call_put, "europeen")], \
            prices[(option.call_put, "americaine")]
        tol_inv = max(1e-8, 10.0 * pricer.p_min if pricer.pruning else 0.0) * max(market.s0, option.strike)
        checks = []

        # Les huit contrôles sont réalisés séparément, dans leur ordre initial.
        tol_mart = 1e-9 * max(abs(price), 1.0)
        Orchestrator._check_bs(checks, market, option, tree, n, maturity_years, r, discount_at_maturity, price, tol_inv)
        Orchestrator._check_parity(checks, prices, s_adj, option, discount_at_maturity, tol_inv, market)
        Orchestrator._check_forward(checks, levels, tree, tol_inv, market, option)
        Orchestrator._check_martingale(checks, levels, r, tree, price, option, n)
        Orchestrator._check_arbitrage(checks, s_adj, option, discount_at_maturity, market, price, tol_inv)
        Orchestrator._check_american_premium(checks, option, amer, euro, tol_mart)
        Orchestrator._check_exercise_equivalence(checks, tree, option, r, amer, euro, tol_mart, pricer)
        Orchestrator._check_engines(checks, n, pricer, market, option)
        return checks

    @staticmethod
    def _check_bs(checks: list[dict], market: Market, option: Option, tree: TreeVectorized, n: int, maturity_years: float,
                  r: float, discount_at_maturity: float, price: float, tol_inv: float) -> None:
        """Applique le contrôle de cohérence numéro 1."""
        # 1. Européenne sans dividende ≈ Black-Scholes, borne C / (6N) de l'erreur de discrétisation
        if not option.is_american() and not tree.div_step and market.vol > 0.0 and maturity_years > 0.0:
            bs = BlackScholes.price_from(market.s0, option.strike, maturity_years, r, market.vol, option.call_put)
            d2 = ((math.log(market.s0 / option.strike) + (r - 0.5 * market.vol ** 2) * maturity_years)
                  / (market.vol * math.sqrt(maturity_years)))
            bound_constant = ((1.5 if tree.nb_nodes == 3 else 2.0) * discount_at_maturity
                              * option.strike * market.vol * math.sqrt(maturity_years) * norm.pdf(d2))
            tol = 1.25 * bound_constant / (6 * n) + tol_inv
            ok = abs(price - bs) <= tol
            checks.append({"Contrôle": "Européenne sans dividende ≈ Black-Scholes",
                           "Valeur": f"arbre {price:.6f} · BS {bs:.6f} · écart {price - bs:+.2e}", "Tolérance": f"{tol:.2e}",
                           "Verdict": "N/A" if n < 30 else ("Validé" if ok else "Échec")})
        else:
            checks.append({"Contrôle": "Européenne sans dividende ≈ Black-Scholes", "Valeur": "—", "Tolérance": "—",
                           "Verdict": "N/A"})

    @staticmethod
    def _check_parity(checks: list[dict], prices: dict, s_adj: float, option: Option, discount_at_maturity: float,
                  tol_inv: float, market: Market) -> None:
        """Applique le contrôle de cohérence numéro 2."""
        # 2. Parité call-put européenne (avec la date de dividende de la grille)
        lhs, rhs = prices[("call", "europeen")] - prices[("put", "europeen")], s_adj - option.strike * discount_at_maturity
        checks.append({"Contrôle": "Parité call-put : C − P = S0 − D·e^(−r·t_D) − K·e^(−rT)",
                       "Valeur": f"C − P = {lhs:.8f} · membre de droite = {rhs:.8f} · écart {lhs - rhs:+.2e}",
                       "Tolérance": f"{tol_inv:.2e}", "Verdict": "Validé" if abs(lhs - rhs) <= tol_inv else
                       ("Attention" if abs(lhs - rhs) <= 1e-3 * max(market.s0, option.strike) else "Échec")})

    @staticmethod
    def _check_forward(checks: list[dict], levels: list, tree: TreeVectorized, tol_inv: float,
                  market: Market, option: Option) -> None:
        """Applique le contrôle de cohérence numéro 3."""
        # 3. Espérance terminale de l'arbre = forward
        mean_at_maturity, fwd = float((levels[-1]["P"] * levels[-1]["S"]).sum()), tree._steps[-1]["F"]
        checks.append({"Contrôle": "Espérance du sous-jacent à maturité = forward",
                       "Valeur": f"arbre {mean_at_maturity:.8f} · forward {fwd:.8f} · écart {mean_at_maturity - fwd:+.2e}",
                       "Tolérance": f"{tol_inv:.2e}", "Verdict": "Validé" if abs(mean_at_maturity - fwd) <= tol_inv else
                       ("Attention" if abs(mean_at_maturity - fwd) <= 1e-3 * max(market.s0, option.strike) else "Échec")})

    @staticmethod
    def _check_martingale(checks: list[dict], levels: list, r: float, tree: TreeVectorized, price: float,
                  option: Option, n: int) -> None:
        """Applique le contrôle de cohérence numéro 4."""
        # 4. Valeur moyenne actualisée par date : martingale (européenne) ou surmartingale (américaine)
        avg = [float((lv["P"] * lv["V"]).sum()) * math.exp(-r * i * tree.dt_) for i, lv in enumerate(levels)]
        tol_mart = 1e-9 * max(abs(price), 1.0)
        if option.is_american():
            gap = max(avg[i + 1] - avg[i] for i in range(n)) if n else 0.0
            checks.append({"Contrôle": "Valeur moyenne actualisée décroissante par date (surmartingale)",
                           "Valeur": f"à t0 {avg[0]:.6f} · à maturité {avg[-1]:.6f} · hausse max {gap:+.2e}",
                           "Tolérance": f"{tol_mart:.2e}", "Verdict": "Validé" if gap <= tol_mart else "Échec"})
        else:
            gap = max(abs(a - avg[0]) for a in avg)
            checks.append({"Contrôle": "Valeur moyenne actualisée constante par date (martingale)",
                           "Valeur": f"à t0 {avg[0]:.6f} · à maturité {avg[-1]:.6f} · écart max {gap:.2e}",
                           "Tolérance": f"{tol_mart:.2e}", "Verdict": "Validé" if gap <= tol_mart else "Échec"})

    @staticmethod
    def _check_arbitrage(checks: list[dict], s_adj: float, option: Option, discount_at_maturity: float, market: Market,
                  price: float, tol_inv: float) -> None:
        """Applique le contrôle de cohérence numéro 5."""
        # 5. Bornes d'arbitrage
        if option.call_put == "call":
            low, high = max(s_adj - option.strike * discount_at_maturity, 0.0), (market.s0 if option.is_american() else s_adj)
        else:
            low = max(option.strike * discount_at_maturity - s_adj, 0.0)
            high = option.strike if option.is_american() else option.strike * discount_at_maturity
        if option.is_american():
            low = max(low, option.payoff(market.s0))
        checks.append({"Contrôle": "Bornes d'arbitrage", "Valeur": f"{low:.6f} ≤ {price:.6f} ≤ {high:.6f}",
                       "Tolérance": f"{tol_inv:.2e}",
                       "Verdict": "Validé" if low - tol_inv <= price <= high + tol_inv else "Échec"})

    @staticmethod
    def _check_american_premium(checks: list[dict], option: Option, amer: float, euro: float, tol_mart: float) -> None:
        """Applique le contrôle de cohérence numéro 6."""
        # 6. Américaine ≥ européenne
        checks.append({"Contrôle": f"{option.call_put.capitalize()} américain ≥ européen",
                       "Valeur": f"Am {amer:.6f} · Eu {euro:.6f} · prime {amer - euro:+.2e}", "Tolérance": f"{tol_mart:.2e}",
                       "Verdict": "Validé" if amer - euro >= -tol_mart else "Échec"})

    @staticmethod
    def _check_exercise_equivalence(checks: list[dict], tree: TreeVectorized, option: Option, r: float,
                  amer: float, euro: float, tol_mart: float, pricer: PriceParameters) -> None:
        """Applique le contrôle de cohérence numéro 7."""
        # 7. Américaine = européenne quand l'exercice anticipé est sous-optimal (sans dividende)
        if not tree.div_step and ((option.call_put == "call" and r >= 0.0) or (option.call_put == "put" and r <= 0.0)):
            checks.append({"Contrôle": f"{option.call_put.capitalize()} américain = européen "
                                       f"(sans dividende, r {'≥' if option.call_put == 'call' else '≤'} 0)",
                           "Valeur": f"Am {amer:.6f} · Eu {euro:.6f} · écart {amer - euro:+.2e}",
                           "Tolérance": f"{tol_mart:.2e}",
                           "Verdict": "Validé" if abs(amer - euro) <= tol_mart else ("Attention" if pricer.pruning and
                                      abs(amer - euro) <= 1e-4 * max(abs(euro), 1.0) else "Échec")})
        else:
            checks.append({"Contrôle": "Américaine = européenne (exercice anticipé sous-optimal)", "Valeur": "—",
                           "Tolérance": "—", "Verdict": "N/A"})

    @staticmethod
    def _check_engines(checks: list[dict], n: int, pricer: PriceParameters, market: Market, option: Option) -> None:
        """Applique le contrôle de cohérence numéro 8."""
        # 8. Moteurs vectorisé et objets identiques (sur au plus 200 pas)
        n_test = min(n, 200)
        test = {**vars(pricer), "nb_steps": n_test}
        v_vec = Orchestrator.price(market, option, PriceParameters(**{**test, "engine": "vector"}))
        v_obj = Orchestrator.price(
            market, option, PriceParameters(**{**test, "engine": "objects", "pricing_method": "backward"})
        )
        tol = 1e-9 * max(abs(v_vec), 1.0)
        checks.append({"Contrôle": "Moteurs vectorisé et objets identiques",
                       "Valeur": f"N = {n_test} : vectorisé {v_vec:.10f} · objets {v_obj:.10f} · écart {v_vec - v_obj:+.1e}",
                       "Tolérance": f"{tol:.1e}", "Verdict": "Validé" if abs(v_vec - v_obj) <= tol else "Échec"})


    # ------------------------------------------------------------------------------------------------------------
    # Analyses ajoutées pour le rapport : exercice anticipé
    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def exercise_profile(market: Market, option: Option, pricer: PriceParameters) -> dict:
        """Option américaine : probabilité d'exercer pour la première fois à chaque date (passe avant absorbante :
        la probabilité n'est propagée qu'à travers les nœuds non exercés), valeur moyenne actualisée par date
        e^(-rt) Σ P·V de l'américaine et de l'européenne, et frontière d'exercice (prix critique S* par date)."""
        out = {}
        for ex in ("americaine", "europeen"):
            opt = Option(option.strike, option.maturity, option.call_put, ex)
            tree = Orchestrator.vector_tree(market, opt, PriceParameters(**{**vars(pricer), "engine": "vector"}))
            levels, n = tree.levels, tree.n_steps
            times = np.arange(n + 1) * tree.dt_
            out[ex] = np.array([float((lv["P"] * lv["V"]).sum()) for lv in levels]) * np.exp(-market.rate * times)
            if ex == "americaine":
                alive, first = np.array([1.0]), np.zeros(n + 1)
                for i in range(n + 1):
                    first[i] = float((alive * levels[i]["ex"]).sum())
                    if i == n:
                        break
                    tr = tree.transitions(i + 1)
                    iu, im, idn = LevelVectorized.child_index(levels[i + 1]["pos"], tr["k"], tr["has"])
                    nxt, cont = np.zeros(len(levels[i + 1]["S"])), alive * ~levels[i]["ex"]
                    for idx, p in zip((iu, im, idn), tr["p"]):
                        np.add.at(nxt, idx, cont * p)
                    alive = nxt
                out["times"], out["first"], out["div_time"] = times, first, tree.div_step * tree.dt_
                p_min = pricer.p_min if pricer.pruning else 0.0
                frontier = []
                for i, lv in enumerate(levels[:-1]):
                    mask = lv["ex"] & (lv["P"] >= p_min)
                    if mask.any():
                        s_star = lv["S"][mask].max() if option.call_put == "put" else lv["S"][mask].min()
                        frontier.append((float(times[i]), float(s_star)))
                out["frontier"] = frontier
            tree.release()
        return out

    @staticmethod
    def exercise_frontier_scenarios(market: Market, option: Option, pricer: PriceParameters,
                                    scenarios: dict) -> tuple[dict, float | None]:
        """Frontière d'exercice (date, S*) de l'option américaine pour chaque scénario {nom: (taux, dividende)}.
        Spot, volatilité, date de détachement, strike et maturité restent ceux de l'utilisateur.
        Renvoie aussi la date du détachement (en années) pour la tracer sur le graphique."""
        opt = Option(option.strike, option.maturity, option.call_put, "americaine")
        params = PriceParameters(**{**vars(pricer), "engine": "vector"})
        frontiers, div_time = {}, None
        p_min = pricer.p_min if pricer.pruning else 0.0

        for name, (r, d) in scenarios.items():
            tree = Orchestrator.vector_tree(Market(market.s0, r, market.vol, d, market.ex_div_date), opt, params)
            try:
                if d > 0 and tree.div_step > 0:
                    div_time = tree.div_step * tree.dt_

                points = []
                for i, lv in enumerate(tree.levels[:-1]):
                    # On ne garde que les nœuds qui ont une vraie probabilité (non élagués)
                    mask = lv["ex"] & (lv["P"] >= p_min)
                    if mask.any():
                        s_star = lv["S"][mask].max() if opt.call_put == "put" else lv["S"][mask].min()
                        points.append((i * tree.dt_, float(s_star)))
                frontiers[name] = points
            finally:
                tree.release()
        return frontiers, div_time

    @staticmethod
    def benchmark(market: Market, option: Option, pricer: PriceParameters, method: str = "backward",
                  memory: bool = False, with_recursive: bool = False) -> dict:
        """Un pricing mesuré : temps de construction, de valorisation, nombre de nœuds (et pic mémoire Python)."""
        gc.collect()
        if memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        tree = Orchestrator.build(market, option, pricer)
        t1 = time.perf_counter()
        pricer_tree = OptionPricerTree(tree)
        price = pricer_tree.price(method)
        t2 = time.perf_counter()
        out = {"N": pricer.nb_steps, "Construction (s)": t1 - t0, "Valorisation (s)": t2 - t1,
               "Total (s)": t2 - t0, "Nœuds": pricer_tree.nodes_priced, "Prix": price}
        if memory:
            out["Pic mémoire (Mo)"] = tracemalloc.get_traced_memory()[1] / 2 ** 20
            tracemalloc.stop()
        if with_recursive and isinstance(tree, TreeNode):        # même arbre, revalorisé en récursif
            tree.reset_values()
            t3 = time.perf_counter()
            pricer_tree.recursive_pricing()
            out["Récursif (s)"] = time.perf_counter() - t3
        tree.release()
        gc.collect()
        return out
    @staticmethod
    def _debug_rows_intro(mode: str, n: int, tree: TreeVectorized, div_text: str, pricer: PriceParameters,
                  singles: int) -> list[dict]:
        """Prépare une partie des métriques du rapport de diagnostic."""
        return [
            {"Métrique": "Méthode", "Valeur": mode, "Critère": "", "Verdict": "—"},
            {"Métrique": "Nombre de pas N, pas de temps Δt", "Valeur": f"{n:,} pas, Δt = {tree.dt_:.6f} an",
             "Critère": "", "Verdict": "—"},
            {"Métrique": "Facteurs de grille", "Critère": "",
             "Valeur": f"u = √α_bin = {tree.sqrt_alpha_bin:.8f} · α_bin = u² = {tree.alpha_bin:.8f} · "
                       f"α_tri = {tree.alpha_tri:.8f}", "Verdict": "—"},
            {"Métrique": "Dividende", "Valeur": div_text, "Critère": "", "Verdict": "—"},
            {"Métrique": "Pruning", "Critère": "", "Verdict": "—",
             "Valeur": f"activé (p_min = {pricer.p_min:.0e}), {singles:,} nœuds à un seul enfant" if pricer.pruning
             else f"désactivé, {singles:,} nœuds à un seul enfant (σ = 0)"},
        ]

    @staticmethod
    def _debug_rows_structure(widths: list, full: int, tri_steps: int, n: int, merges: int, branches: int,
                  growth: float) -> list[dict]:
        """Prépare une partie des métriques du rapport de diagnostic."""
        return [
            {"Métrique": "Nœuds créés", "Critère": "", "Verdict": "—",
             "Valeur": f"{sum(widths):,} ({sum(widths) / full:.1%} d'un arbre complet sans dividende)"},
            {"Métrique": "Largeur maximale / finale", "Valeur": f"{max(widths):,} / {widths[-1]:,} nœuds",
             "Critère": "", "Verdict": "—"},
            {"Métrique": "Pas trinomiaux", "Valeur": f"{tri_steps:,} / {n:,}", "Critère": "", "Verdict": "—"},
            {"Métrique": "Recombinaisons (branches fusionnées)", "Valeur": f"{merges:,} sur {branches:,} branches",
             "Critère": "> 0 dès N ≥ 2", "Verdict": "Validé" if merges > 0 or n < 2 else "Échec"},
            {"Métrique": "Croissance maximale d'un niveau (pas ≥ 3, hors dividende)", "Valeur": f"× {growth:.4f}",
             "Critère": "< × 1,5 (sans recombinaison : × 2 ou × 3)", "Verdict": "Validé" if growth < 1.5 else "Échec"},
        ]

    @staticmethod
    def _debug_rows_quality(worst_proba: float, clipped: int, clipped_mass: float, rel_var: float,
                  sum_p: float, build_s: float, price_s: float) -> list[dict]:
        """Prépare une partie des métriques du rapport de diagnostic."""
        return [
            {"Métrique": "Probabilités de transition dans [0, 1], de somme 1", "Valeur": f"écart max {worst_proba:.1e}",
             "Critère": "< 1e-12", "Verdict": "Validé" if worst_proba < 1e-12 else "Échec"},
            {"Métrique": "Nœuds aux moments corrigés (probabilité négative ramenée à 0)",
             "Valeur": f"{clipped:,} nœuds, masse {clipped_mass:.2e}, erreur relative de variance {rel_var:.1e}",
             "Critère": "masse < 1e-4",
             "Verdict": "Validé" if clipped_mass < 1e-4 else "Attention"},
            {"Métrique": "Somme des probabilités à maturité", "Valeur": f"{sum_p:.12f}", "Critère": "|Σ − 1| < 1e-10",
             "Verdict": "Validé" if abs(sum_p - 1.0) < 1e-10 else "Échec"},
            {"Métrique": "Temps", "Critère": "", "Verdict": "—",
             "Valeur": f"construction {build_s * 1000:,.1f} ms · valorisation {price_s * 1000:,.1f} ms · "
                       f"total {(build_s + price_s) * 1000:,.1f} ms"},
        ]

    @staticmethod
    def _debug_counts(tree: TreeVectorized, market: Market, widths: list) -> tuple:
        """Agrège le nombre de branches, les écarts de moments et les probabilités."""
        branches = merges = singles = clipped = 0
        clipped_mass, worst_proba, rel_var = 0.0, 0.0, 0.0
        var_factor = math.exp(2.0 * market.rate * tree.dt_) * math.expm1(market.vol ** 2 * tree.dt_)
        for i in range(1, tree.n_steps + 1):
            count, worst, wrong, mass, variance_error = Orchestrator._debug_one_level(
                tree, i, var_factor
            )
            branches += int(count.sum())
            merges += int(count.sum()) - widths[i]
            singles += int((count == 1).sum())
            worst_proba = max(worst_proba, worst)
            clipped += int(wrong.sum())
            clipped_mass += mass
            rel_var = max(rel_var, variance_error)
        return branches, merges, singles, clipped, clipped_mass, worst_proba, rel_var

    @staticmethod
    def _debug_one_level(tree: TreeVectorized, i: int, var_factor: float) -> tuple:
        """Mesure les erreurs des transitions vers un niveau de l'arbre."""
        levels = tree.levels
        tr = tree.transitions(i)
        iu, im, idn = LevelVectorized.child_index(levels[i]["pos"], tr["k"], tr["has"])
        pu, pm, pd_ = tr["p"]
        count = tr["has"][0].astype(int) + tr["has"][1].astype(int) + tr["has"][2].astype(int)
        worst = max(float(np.max(np.abs(pu + pm + pd_ - 1.0))),
                    float(-min(pu.min(), pm.min(), pd_.min(), 0.0)),
                    float(max(pu.max(), pm.max(), pd_.max()) - 1.0))
        child_spots, parent_spots = levels[i]["S"], levels[i - 1]["S"]
        m1 = child_spots[iu] * pu + child_spots[im] * pm + child_spots[idn] * pd_
        m2 = child_spots[iu] ** 2 * pu + child_spots[im] ** 2 * pm + child_spots[idn] ** 2 * pd_
        target_var = parent_spots ** 2 * var_factor
        wrong = (count >= 2) & ((np.abs(m1 - tr["E"]) > 1e-9 * tr["E"])
                                | (np.abs(m2 - m1 ** 2 - target_var) > 1e-6 * np.maximum(target_var, 1e-300)))
        mass = float(levels[i - 1]["P"][wrong].sum())
        variance_error = float((levels[i - 1]["P"] * np.abs(m2 - m1 ** 2 - target_var)
                                / np.maximum(target_var, 1e-300))[count >= 2].sum())
        return count, worst, wrong, mass, variance_error

    @staticmethod
    def _debug_dividend(market: Market, pricer: PriceParameters, tree: TreeVectorized) -> str:
        """Décrit le détachement et le place sur la grille temporelle."""
        if tree.div_step:
            exact = PriceParameters.year_fraction(pricer.pricing_date, market.ex_div_date)
            return (f"D = {market.dividend:,.4f} détaché à la fin du pas {tree.div_step} "
                    f"(t = {tree.div_step * tree.dt_:.5f} an sur la grille, {exact:.5f} an réel)")
        return "Aucun dividende dans ]date de pricing, maturité]"
