from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from level import LevelVectorized
from tree import TreeVectorized
from tree_svg import SvgTreeExporter


class TreeVisualizer:
    Y_OPTIONS = {"Prix du sous-jacent": "S", "Prix de l'option": "V"}
    COLOR_OPTIONS = {"Probabilité cumulée": "mass", "Marge d'exercice": "margin",
                     "Écart d'espérance": "dmean", "Écart de variance": "dvar"}
    MAX_EDGES_NODES = 15_000      # au-delà, les branches ne sont pas tracées (navigateur)
    MAX_NODES = 300_000           # au-delà, un niveau sur k est affiché
    SVG_MAX_NODES = 12_000        # au-delà, le SVG serait trop volumineux

    @staticmethod
    def build(tree: TreeVectorized, y_mode: str = "S", color_mode: str = "mass",
              t_range: tuple | None = None, log_y: bool = False, height: int = 620) -> dict:
        """Construit le tracé de l'arbre et sa légende sans modifier les niveaux."""
        levels, n = tree.levels, len(tree.levels) - 1
        t0, t1 = t_range if t_range is not None else (0, n)
        size = sum(len(levels[t]["S"]) for t in range(t0, t1 + 1))
        stride = max(1, int(np.ceil(size / TreeVisualizer.MAX_NODES)))
        times = list(range(t0, t1 + 1, stride))
        draw_edges = stride == 1 and size <= TreeVisualizer.MAX_EDGES_NODES
        american, is_call = tree.option.is_american(), tree.option.call_put == "call"
        p_min = tree.pricer.p_min if tree.pricer.pruning else 0.0
        x, y, color, hover, ex_x, ex_y, front_x, front_y, edge_x, edge_y = ([] for _ in range(10))

        channels = (x, y, color, hover, ex_x, ex_y, front_x, front_y, edge_x, edge_y)
        for t in times:
            TreeVisualizer._collect_level(
                tree, t, y_mode, color_mode, draw_edges, american,
                is_call, p_min, stride, channels
            )
        # Les traces sont assemblées une fois les niveaux échantillonnés.
        fig = TreeVisualizer._figure(
            x, y, color, hover, ex_x, ex_y, front_x, front_y,
            edge_x, edge_y, size, color_mode, american, y_mode, height, log_y
        )
        caption = TreeVisualizer._caption(size, t0, t1, stride, draw_edges)
        svg = None
        if size <= TreeVisualizer.SVG_MAX_NODES and stride == 1:
            svg = SvgTreeExporter(tree, times, y, color, y_mode, log_y, draw_edges).build(
                TreeVisualizer._color_label(color_mode)
            )
        return {"fig": fig, "caption": caption, "svg": svg}

    @staticmethod
    def _color_label(color_mode: str) -> str:
        """Retrouve le libellé de l'échelle utilisé pour l'export SVG."""
        if color_mode == "mass":
            return "log10(proba cumulée)"
        return next(label for label, code in TreeVisualizer.COLOR_OPTIONS.items() if code == color_mode)

    @staticmethod
    def _collect_level(tree: TreeVectorized, t: int, y_mode: str, color_mode: str,
                       draw_edges: bool, american: bool, is_call: bool,
                       p_min: float, stride: int, channels: tuple) -> None:
        """Rassemble les données de tracé d'un niveau sans créer de figure."""
        x, y, color, hover, ex_x, ex_y, front_x, front_y, edge_x, edge_y = channels
        lv = tree.levels[t]
        spot_prices, option_values, probabilities = lv["S"], lv["V"], lv["P"]
        yv = spot_prices if y_mode == "S" else option_values
        c, links = TreeVisualizer._level_colors(tree, t, color_mode, draw_edges)
        x.append(np.full(len(spot_prices), t))
        y.append(yv)
        color.append(c)
        hover.append([
            f"Pas {t}<br>S = {s:.4f}<br>V = {v:.4f}<br>P = {p:.3e}"
            for s, v, p in zip(spot_prices, option_values, probabilities)
        ])
        if draw_edges and t != tree.n_steps:
            TreeVisualizer._append_edges(edge_x, edge_y, yv, y_mode, t, links)
        if american:
            TreeVisualizer._append_exercise(
                ex_x, ex_y, front_x, front_y, lv, spot_prices, probabilities,
                yv, is_call, p_min, t, stride
            )

    @staticmethod
    def _level_colors(tree: TreeVectorized, t: int, color_mode: str,
                      draw_edges: bool) -> tuple:
        """Lit la couleur des nœuds et les transitions nécessaires aux branches."""
        levels, n = tree.levels, tree.n_steps
        spot_prices, probabilities = levels[t]["S"], levels[t]["P"]
        links = None
        if t != n and (color_mode != "mass" or draw_edges):
            tr = tree.transitions(t + 1)
            child = levels[t + 1]
            iu, im, idn = LevelVectorized.child_index(child["pos"], tr["k"], tr["has"])
            pu, pm, pd_ = tr["p"]
            links = tr, child, (iu, im, idn)
        if color_mode == "mass":
            return np.log10(np.maximum(probabilities, 1e-300)), links
        if t == n:
            return np.zeros(len(spot_prices)), links
        if color_mode == "margin":
            cont = (child["V"][iu] * pu + child["V"][im] * pm + child["V"][idn] * pd_) * tree.discount_value
            return tree.option.payoff_array(spot_prices) - cont, links
        return TreeVisualizer._moment_difference(tree, spot_prices, child, (iu, im, idn), tr, color_mode), links

    @staticmethod
    # L’écart de moment vérifie la cohérence du maillage avec le modèle.
    def _moment_difference(tree: TreeVectorized, spot_prices: np.ndarray, child: dict,
                           indices: tuple, tr: dict, color_mode: str) -> np.ndarray:
        """Compare l'espérance ou la variance à la cible théorique du modèle."""
        iu, im, idn = indices
        pu, pm, pd_ = tr["p"]
        mean = child["S"][iu] * pu + child["S"][im] * pm + child["S"][idn] * pd_
        if color_mode == "dmean":
            return mean - tr["E"]
        second = child["S"][iu] ** 2 * pu + child["S"][im] ** 2 * pm + child["S"][idn] ** 2 * pd_
        theory = (spot_prices ** 2 * np.exp(2 * tree.market.rate * tree.dt_)
                  * np.expm1(tree.market.vol ** 2 * tree.dt_))
        return (second - mean ** 2) - theory

    @staticmethod
    def _append_edges(edge_x: list, edge_y: list, yv: np.ndarray,
                      y_mode: str, t: int, links: tuple) -> None:
        """Ajoute les segments parents-enfants lorsque leur nombre le permet."""
        tr, child, (iu, im, idn) = links
        yc = child["S"] if y_mode == "S" else child["V"]
        for h, ix in zip(tr["has"], (iu, im, idn)):
            sel = np.flatnonzero(h)
            seg = np.full((len(sel), 3), np.nan)
            seg[:, 0], seg[:, 1] = t, t + 1
            edge_x.append(seg.ravel())
            seg_y = np.full((len(sel), 3), np.nan)
            seg_y[:, 0], seg_y[:, 1] = yv[sel], yc[ix[sel]]
            edge_y.append(seg_y.ravel())

    @staticmethod
    def _append_exercise(ex_x: list, ex_y: list, front_x: list, front_y: list,
                         lv: dict, spot_prices: np.ndarray, probabilities: np.ndarray, yv: np.ndarray,
                         is_call: bool, p_min: float, t: int, stride: int) -> None:
        """Repère les exercices optimaux et la frontière sur le niveau."""
        ex = lv["ex"] & (probabilities >= p_min)
        ex_x.append(np.full(ex.sum(), t))
        ex_y.append(yv[ex])
        if ex.any():
            j = np.flatnonzero(ex)[np.argmin(spot_prices[ex]) if is_call else np.argmax(spot_prices[ex])]
            if front_x and front_x[-1] is not None and front_x[-1] != t - stride:
                front_x.append(None)
                front_y.append(None)
            front_x.append(t)
            front_y.append(float(yv[j]))

    @staticmethod
    def _figure(x: list, y: list, color: list, hover: list, ex_x: list,
                ex_y: list, front_x: list, front_y: list, edge_x: list,
                edge_y: list, size: int, color_mode: str, american: bool,
                y_mode: str, height: int, log_y: bool) -> go.Figure:
        """Assemble les traces Plotly sans changer leur ordre ni leur style."""
        fig = go.Figure()
        if edge_x:
            fig.add_trace(go.Scattergl(x=np.concatenate(edge_x), y=np.concatenate(edge_y), mode="lines",
                                       line=dict(color="rgba(110,122,140,0.35)", width=1), hoverinfo="skip",
                                       name="Branches"))
        fig.add_trace(go.Scattergl(
            x=np.concatenate(x), y=np.concatenate(y), mode="markers", name="Nœuds",
            text=[h for level in hover for h in level], hovertemplate="%{text}<extra></extra>",
            marker=dict(size=6 if size < 2_000 else 4 if size < 50_000 else 2, color=np.concatenate(color),
                        colorscale="Turbo", showscale=True,
                        colorbar=dict(title="log10(P)" if color_mode == "mass" else "écart", thickness=12))))
        if american and ex_x:
            TreeVisualizer._draw_exercise(fig, ex_x, ex_y, front_x, front_y)
        fig.update_layout(height=height, template="plotly_white", margin=dict(l=25, r=25, t=40, b=45),
                          xaxis_title="Pas de temps",
                          yaxis_title=next(k for k, v in TreeVisualizer.Y_OPTIONS.items() if v == y_mode),
                          hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        fig.update_yaxes(type="log" if log_y else "linear")
        return fig

    @staticmethod
    def _draw_exercise(fig: go.Figure, ex_x: list, ex_y: list,
                       front_x: list, front_y: list) -> None:
        """Dessine les nœuds exercés et leur limite d'exercice anticipé."""
        fig.add_trace(go.Scattergl(x=np.concatenate(ex_x), y=np.concatenate(ex_y), mode="markers",
                                   name="Exercice optimal", hoverinfo="skip",
                                   marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color="black", width=1))))
        fig.add_trace(go.Scatter(x=front_x, y=front_y, mode="lines+markers", name="Frontière d'exercice",
                                 line=dict(color="black", width=2), hoverinfo="skip"))

    @staticmethod
    def _caption(size: int, t0: int, t1: int, stride: int, draw_edges: bool) -> str:
        """Décrit l'échantillonnage et le masquage éventuel des branches."""
        return (f"{size:,} nœuds dans la fenêtre (pas {t0} → {t1})"
                + (f", un niveau sur {stride} affiché" if stride > 1 else "")
                + ("" if draw_edges else " ; branches masquées au-delà de "
                   f"{TreeVisualizer.MAX_EDGES_NODES:,} nœuds (réduisez la fenêtre pour les voir)") + ".")
