"""Génération d'un export SVG autonome pour la visualisation de l'arbre."""

from __future__ import annotations

import html
import math

import numpy as np
from plotly.colors import hex_to_rgb, sequential

from level import LevelVectorized
from tree import TreeVectorized


# La palette Turbo est interpolée comme dans la visualisation Plotly.
_TURBO_RGB = np.array([hex_to_rgb(color) for color in sequential.Turbo], dtype=float)
_TURBO_POSITIONS = np.linspace(0.0, 1.0, len(_TURBO_RGB))


# Le SVG reste interactif dans un navigateur, même sans connexion à l'application.
_SVG_SCRIPT = """<![CDATA[
(function () {
  var svg = document.documentElement, NS = 'http://www.w3.org/2000/svg';
  var tip = document.getElementById('tip'), bg = document.getElementById('tipbg');
  var txt = document.getElementById('tiptxt'), vp = document.getElementById('vp');
  var W = 1600, H = 900, pinned = false, scale = 1, tx = 0, ty = 0;
  var drag = null, moved = false;
  function pt(evt) {
    var p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
    return p.matrixTransform(svg.getScreenCTM().inverse());
  }
  function show(el, evt) {
    while (txt.firstChild) txt.removeChild(txt.firstChild);
    el.getAttribute('data-info').split('|').forEach(function (line, i) {
      var t = document.createElementNS(NS, 'tspan');
      t.setAttribute('x', 10); t.setAttribute('dy', i ? 17 : 0);
      if (i === 0) t.setAttribute('font-weight', 'bold');
      t.textContent = line; txt.appendChild(t);
    });
    tip.setAttribute('visibility', 'visible');
    var b = txt.getBBox(); bg.setAttribute('width', b.width + 20);
    bg.setAttribute('height', b.height + 14);
    var q = pt(evt), x = q.x + 14, y = q.y + 14;
    if (x + b.width + 20 > W) x = q.x - b.width - 34;
    if (y + b.height + 14 > H) y = q.y - b.height - 28;
    tip.setAttribute('transform', 'translate(' + x + ',' + y + ')');
  }
  function hide() { if (!pinned) tip.setAttribute('visibility', 'hidden'); }
  Array.prototype.forEach.call(document.querySelectorAll('[data-info]'), function (el) {
    el.addEventListener('mouseover', function (e) { if (!pinned) show(el, e); });
    el.addEventListener('mouseout', hide);
    el.addEventListener('click', function (e) {
      if (moved) return; pinned = true; show(el, e); e.stopPropagation();
    });
  });
  function apply() { vp.setAttribute('transform', 'translate(' + tx + ',' + ty + ') scale(' + scale + ')'); }
  svg.addEventListener('wheel', function (e) {
    e.preventDefault(); var q = pt(e), f = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    var ns = Math.min(Math.max(scale * f, 1), 60); f = ns / scale;
    tx = q.x - f * (q.x - tx); ty = q.y - f * (q.y - ty);
    scale = ns; apply();
  }, { passive: false });
  svg.addEventListener('mousedown', function (e) {
    var q = pt(e); drag = { x: q.x - tx, y: q.y - ty }; moved = false;
  });
  svg.addEventListener('mousemove', function (e) {
    if (!drag) return;
    var q = pt(e);
    if (Math.abs(q.x - drag.x - tx) + Math.abs(q.y - drag.y - ty) > 3) moved = true;
    if (moved) { tx = q.x - drag.x; ty = q.y - drag.y; apply(); }
  });
  svg.addEventListener('mouseup', function () {
    drag = null; setTimeout(function () { moved = false; }, 0);
  });
  svg.addEventListener('click', function () {
    if (moved) return; pinned = false; tip.setAttribute('visibility', 'hidden');
  });
  svg.addEventListener('dblclick', function () { scale = 1; tx = 0; ty = 0; apply(); });
})();
]]>"""


class SvgTreeExporter:
    """Produit un SVG avec les mêmes nœuds et branches que le graphique affiché."""

    WIDTH = 1600
    HEIGHT = 900
    LEFT = 90
    RIGHT = 120
    TOP = 70
    BOTTOM = 80

    def __init__(self, tree: TreeVectorized, times: list[int],
                 y_values: list[np.ndarray], colors: list[np.ndarray],
                 y_mode: str, log_y: bool, draw_edges: bool) -> None:
        self.tree = tree
        self.times = times
        self.y_mode = y_mode
        self.log_y = log_y
        self.draw_edges = draw_edges
        self.x = np.concatenate([np.full(len(y), t) for t, y in zip(times, y_values)])
        self.y = np.concatenate(y_values)
        self.colors = np.concatenate(colors)
        # Les données de l'export correspondent exactement aux niveaux affichés.
        self.spots = self._level_array("S")
        self.values = self._level_array("V")
        self.probabilities = self._level_array("P")
        self.positions = self._level_array("pos")
        self.exercised = self._level_array("ex")
        self.filtered_y = self._transform(self.y)
        self._set_bounds()

    def _level_array(self, key: str) -> np.ndarray:
        return np.concatenate([self.tree.levels[t][key] for t in self.times])

    def _transform(self, values: np.ndarray) -> np.ndarray:
        if not self.log_y:
            return values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(values), np.nan)

    def _set_bounds(self) -> None:
        # Les limites excluent les valeurs non affichables en échelle logarithmique.
        finite = self.filtered_y[np.isfinite(self.filtered_y)]
        self.x_min, self.x_max = float(self.x.min()), float(self.x.max())
        if self.draw_edges and self.times[-1] < self.tree.n_steps:
            # La dernière branche atteint le niveau suivant, comme sur Plotly.
            self.x_max += 1.0
            next_y = self._transform(self.tree.levels[self.times[-1] + 1][self.y_mode])
            finite = np.concatenate((finite, next_y[np.isfinite(next_y)]))
        self.x_max = self.x_max if self.x_max != self.x_min else self.x_min + 1.0
        self.y_min = float(finite.min()) if finite.size else 0.0
        self.y_max = float(finite.max()) if finite.size else 1.0
        self.y_max = self.y_max if abs(self.y_max - self.y_min) > 1e-12 else self.y_min + 1.0
        padding = 0.04 * (self.y_max - self.y_min)
        self.y_min, self.y_max = self.y_min - padding, self.y_max + padding
        # La palette conserve les bornes de couleur du tracé.
        finite_colors = self.colors[np.isfinite(self.colors)]
        self.color_min = float(finite_colors.min()) if finite_colors.size else 0.0
        self.color_max = float(finite_colors.max()) if finite_colors.size else 1.0
        if abs(self.color_max - self.color_min) < 1e-12:
            self.color_max = self.color_min + 1.0

    def _px(self, value: float) -> float:
        return self.LEFT + (value - self.x_min) / (self.x_max - self.x_min) * self.plot_width

    def _py(self, value: float) -> float:
        return self.TOP + self.plot_height - (value - self.y_min) / (self.y_max - self.y_min) * self.plot_height

    # Les conversions de coordonnées utilisent les mêmes marges pour tous les éléments.
    @property
    def plot_width(self) -> int:
        return self.WIDTH - self.LEFT - self.RIGHT

    @property
    def plot_height(self) -> int:
        return self.HEIGHT - self.TOP - self.BOTTOM

    def _hex_color(self, value: float) -> str:
        ratio = (value - self.color_min) / (self.color_max - self.color_min)
        rgb = [int(np.interp(np.clip(ratio, 0.0, 1.0), _TURBO_POSITIONS, _TURBO_RGB[:, i])) for i in range(3)]
        return "#%02x%02x%02x" % tuple(rgb)

    def build(self, color_label: str) -> str:
        """Assemble le graphique SVG autonome et son script d'interaction."""
        if not len(self.x):
            return '<svg xmlns="http://www.w3.org/2000/svg"/>'
        parts = self._header()
        parts.extend(self._axes(color_label))
        parts.extend(self._colorbar(color_label))
        parts.append('<g clip-path="url(#clip)"><g id="vp">')
        if self.draw_edges:
            parts.extend(self._branches())
        parts.extend(self._nodes(color_label))
        parts.extend(self._frontier())
        parts.append('</g></g>')
        parts.extend(self._tooltip())
        return "".join(parts)

    def _header(self) -> list[str]:
        """Définit les dimensions, la légende et le fond du document."""
        w, h, l, top = self.WIDTH, self.HEIGHT, self.LEFT, self.TOP
        return [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            'font-family="Arial, Helvetica, sans-serif">',
            '<style>.n{cursor:pointer}.n:hover{stroke:#000;stroke-width:2.5}'
            '.e{cursor:pointer}.e:hover{stroke:#d62728;stroke-width:3;opacity:1}</style>',
            f'<defs><clipPath id="clip"><rect x="{l}" y="{top}" width="{self.plot_width}" '
            f'height="{self.plot_height}"/></clipPath>',
            '<linearGradient id="cbar" x1="0" y1="1" x2="0" y2="0">'
            + "".join(self._color_stop(i / 4) for i in range(5)) + '</linearGradient></defs>',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{w / 2}" y="32" text-anchor="middle" font-size="20">Arbre de valorisation</text>',
            f'<text x="{w / 2}" y="54" text-anchor="middle" font-size="12" fill="#666">'
            'Survol ou clic : informations · molette : zoom · glisser : déplacer · double-clic : réinitialiser</text>',
            f'<rect x="{l}" y="{top}" width="{self.plot_width}" height="{self.plot_height}" '
            'fill="none" stroke="#999"/>',
        ]

    def _color_stop(self, offset: float) -> str:
        value = self.color_min + offset * (self.color_max - self.color_min)
        return f'<stop offset="{offset}" stop-color="{self._hex_color(value)}"/>'

    def _axes(self, color_label: str) -> list[str]:
        """Ajoute les graduations et noms des axes du graphique."""
        parts = []
        for i in range(7):
            xv = self.x_min + i * (self.x_max - self.x_min) / 6
            yv = self.y_min + i * (self.y_max - self.y_min) / 6
            label_y = 10 ** yv if self.log_y else yv
            parts.append(self._tick(xv, yv, label_y))
        ylabel = "Prix du sous-jacent" if self.y_mode == "S" else "Prix de l'option"
        parts.append(f'<text x="{self.LEFT + self.plot_width / 2}" y="{self.HEIGHT - 25}" '
                     'text-anchor="middle" font-size="14">Pas de temps</text>')
        parts.append(f'<text transform="translate(25 {self.TOP + self.plot_height / 2}) rotate(-90)" '
                     f'text-anchor="middle" font-size="14">{html.escape(ylabel)}</text>')
        return parts

    def _tick(self, xv: float, yv: float, label_y: float) -> str:
        px, py = self._px(xv), self._py(yv)
        return (f'<line x1="{px:.1f}" y1="{self.TOP + self.plot_height}" '
                f'x2="{px:.1f}" y2="{self.TOP + self.plot_height + 6}" stroke="#555"/>'
                f'<text x="{px:.1f}" y="{self.TOP + self.plot_height + 22}" text-anchor="middle" '
                f'font-size="12">{xv:.0f}</text>'
                f'<line x1="{self.LEFT - 6}" y1="{py:.1f}" x2="{self.LEFT}" y2="{py:.1f}" stroke="#555"/>'
                f'<text x="{self.LEFT - 10}" y="{py + 4:.1f}" text-anchor="end" '
                f'font-size="12">{label_y:.4g}</text>')

    # La légende conserve les limites des couleurs affichées dans les nœuds.
    def _colorbar(self, label: str) -> list[str]:
        left, top = self.WIDTH - self.RIGHT + 30, self.TOP
        return [f'<rect x="{left}" y="{top}" width="16" height="{self.plot_height}" '
                'fill="url(#cbar)" stroke="#999"/>',
                f'<text x="{left + 22}" y="{top + 10}" font-size="11">{self.color_max:.3g}</text>',
                f'<text x="{left + 22}" y="{top + self.plot_height}" '
                f'font-size="11">{self.color_min:.3g}</text>',
                f'<text transform="translate({left + 60} {top + self.plot_height / 2}) rotate(90)" '
                f'text-anchor="middle" font-size="12">{html.escape(label)}</text>']

    def _branches(self) -> list[str]:
        """Reconstitue les branches de la même fenêtre avec leurs probabilités."""
        parts = []
        for t in self.times:
            if t >= self.tree.n_steps:
                continue
            tr = self.tree.transitions(t + 1)
            parent, child = self.tree.levels[t], self.tree.levels[t + 1]
            indices = LevelVectorized.child_index(child["pos"], tr["k"], tr["has"])
            for branch, (present, ix, probabilities) in enumerate(zip(tr["has"], indices, tr["p"])):
                parts.extend(self._branch_lines(t, parent, child, present, ix, probabilities, branch))
        return parts

    # Les transitions absentes ne doivent créer aucun segment dans le SVG.
    def _branch_lines(self, t: int, parent: dict, child: dict, present: np.ndarray,
                      indices: np.ndarray, probabilities: np.ndarray, branch: int) -> list[str]:
        lines = []
        key = self.y_mode
        for i in np.flatnonzero(present):
            ya, yb = self._transform(np.array([parent[key][i], child[key][indices[i]]]))
            if not (math.isfinite(ya) and math.isfinite(yb)):
                continue
            lines.append(self._branch_line(t, ya, yb, parent, child, indices[i], probabilities[i], i, branch))
        return lines

    def _branch_line(self, t: int, ya: float, yb: float, parent: dict, child: dict,
                     j: int, probability: float, i: int, branch: int) -> str:
        kind = ("hausse", "milieu", "baisse")[branch]
        info = html.escape(f"Branche {kind}|Pas : {t} → {t + 1}|"
                           f"S : {parent['S'][i]:.6g} → {child['S'][j]:.6g}|"
                           f"V : {parent['V'][i]:.6g} → {child['V'][j]:.6g}|"
                           f"Probabilité de transition : {probability:.4f}", quote=True)
        return (f'<line class="e" data-info="{info}" x1="{self._px(t):.2f}" y1="{self._py(ya):.2f}" '
                f'x2="{self._px(t + 1):.2f}" y2="{self._py(yb):.2f}" stroke="#aab4c0" '
                'stroke-width="1.2" opacity="0.75" vector-effect="non-scaling-stroke"/>')

    def _nodes(self, color_label: str) -> list[str]:
        """Dessine chaque nœud et son infobulle indépendante."""
        parts = []
        for i, value in enumerate(self.filtered_y):
            if not math.isfinite(value):
                continue
            parts.append(self._node_circle(i, value, color_label))
        return parts

    # Les infobulles réutilisent les valeurs numériques déjà calculées.
    def _node_circle(self, i: int, value: float, label: str) -> str:
        ex = bool(self.exercised[i]) and self.probabilities[i] >= self._exercise_threshold()
        radius, stroke, sw = (4.5, "#111", 1.4) if ex else (3.2, "white", 0.6)
        info = html.escape(f"Nœud – pas {self.x[i]:.0f} (position {self.positions[i]})|"
                           f"Sous-jacent : {self.spots[i]:.8g}|Prix option : {self.values[i]:.8g}|"
                           f"Proba cumulée : {self.probabilities[i]:.5g}|{label} : {self.colors[i]:.4g}|"
                           f"Exercice optimal : {'Oui' if ex else 'Non'}", quote=True)
        color = self.colors[i] if math.isfinite(self.colors[i]) else (self.color_min + self.color_max) / 2
        return (f'<circle class="n" data-info="{info}" cx="{self._px(self.x[i]):.2f}" '
                f'cy="{self._py(value):.2f}" r="{radius}" fill="{self._hex_color(color)}" '
                f'stroke="{stroke}" stroke-width="{sw}"/>')

    def _exercise_threshold(self) -> float:
        return self.tree.pricer.p_min if self.tree.pricer.pruning else 0.0

    def _frontier(self) -> list[str]:
        """Trace la limite d'exercice anticipé sans modifier les calculs."""
        if not self.tree.option.is_american():
            return []
        paths, points, previous_t = [], [], None
        for t in self.times:
            point = self._frontier_point(t)
            if point is None:
                continue
            if previous_t is not None and t != previous_t + 1:
                paths.extend(self._frontier_line(points))
                points = []
            points.append(point)
            previous_t = t
        paths.extend(self._frontier_line(points))
        return paths

    @staticmethod
    def _frontier_line(points: list[str]) -> list[str]:
        if not points:
            return []
        return ['<polyline points="' + " ".join(points) + '" fill="none" '
                'stroke="black" stroke-width="2" vector-effect="non-scaling-stroke"/>']

    # La frontière reprend le filtrage des probabilités de la vue Plotly.
    def _frontier_point(self, t: int) -> str | None:
        level = self.tree.levels[t]
        exercised = level["ex"] & (level["P"] >= self._exercise_threshold())
        if not exercised.any():
            return None
        spots = level["S"]
        index = np.flatnonzero(exercised)
        j = index[np.argmin(spots[exercised])] if self.tree.option.call_put == "call" else (
            index[np.argmax(spots[exercised])])
        value = self._transform(np.array([level[self.y_mode][j]]))[0]
        return f"{self._px(t):.2f},{self._py(value):.2f}" if math.isfinite(value) else None

    @staticmethod
    def _tooltip() -> list[str]:
        return ['<g id="tip" visibility="hidden" pointer-events="none">'
                '<rect id="tipbg" rx="6" fill="white" stroke="#333" stroke-width="1" opacity="0.96"/>'
                '<text id="tiptxt" y="20" font-size="13" fill="#111"/></g>',
                '<script>' + _SVG_SCRIPT + '</script></svg>']
