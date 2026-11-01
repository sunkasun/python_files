from __future__ import annotations

import sys
import threading

import numpy as np

from level import LevelVectorized
from node import Node
from tree import Tree, TreeNode, TreeVectorized


class OptionPricerTree:
    """Valorise l'option portée par un arbre (TreeNode ou TreeVectorized)."""

    def __init__(self, tree: Tree):
        self.tree = tree
        self.nodes_priced = 0

    def price(self, method: str = "backward") -> float:
        return self.recursive_pricing() if method == "recursive" else self.backward_pricing()

    def backward_pricing(self) -> float:
        """Valorise l'arbre depuis la maturité, avec le moteur demandé."""
        if isinstance(self.tree, TreeNode):
            return self._backward_nodes()
        return self._backward_vectors()

    def _backward_nodes(self) -> float:
        """Parcourt les niveaux d'objets du dernier au premier."""
        tree = self.tree
        option, disc, last_step = tree.option, tree.discount_value, tree.n_steps
        strike, is_call, american = option.strike, option.call_put == "call", option.is_american()
        count = 0
        trunk = tree.last_trunk
        while trunk is not None:
            count += self._price_node_level(trunk.node_ref, strike, is_call, american, last_step, disc)
            trunk = trunk.prec
        self.nodes_priced = count
        return float(tree.root.option_value)

    @staticmethod
    def _price_node_level(node: Node, strike: float, is_call: bool,
                          american: bool, last_step: int, discount: float) -> int:
        """Valorise les nœuds d'un niveau de bas en haut."""
        while node.down is not None:
            node = node.down
        maturity = node.time == last_step
        count = 0
        while node is not None:
            payoff = node.und_price - strike if is_call else strike - node.und_price
            if payoff < 0.0:
                payoff = 0.0
            value, exercised = OptionPricerTree._node_value(node, payoff, maturity, american, discount)
            node.option_value = value
            node.is_exerced = exercised
            count += 1
            node = node.up
        return count

    @staticmethod
    def _node_value(node: Node, payoff: float, maturity: bool,
                    american: bool, discount: float) -> tuple[float, bool]:
        """Calcule la continuation actualisée ou la valeur d'exercice."""
        if maturity:
            return payoff, american and payoff > 0.0
        value = 0.0
        child = node.next_up
        if child is not None:
            value += child.option_value * node.p_up
        child = node.next_middle
        if child is not None:
            value += child.option_value * node.p_middle
        child = node.next_down
        if child is not None:
            value += child.option_value * node.p_down
        value *= discount
        exercised = american and payoff > value
        if exercised:
            value = payoff
        return value, exercised

    def _backward_vectors(self) -> float:
        """Actualise simultanément les valeurs d'un niveau vectorisé."""
        tree = self.tree
        option, levels = tree.option, tree.levels
        american = option.is_american()
        last = levels[-1]
        last["V"] = option.payoff_array(last["S"])
        last["ex"] = (last["V"] > 0.0) if american else np.zeros(len(last["S"]), dtype=bool)
        for i in range(tree.n_steps, 0, -1):
            lv, child = levels[i - 1], levels[i]
            tr = tree.transitions(i)
            iu, im, idn = LevelVectorized.child_index(child["pos"], tr["k"], tr["has"])
            pu, pm, pd_ = tr["p"]
            value = (child["V"][iu] * pu + child["V"][im] * pm + child["V"][idn] * pd_) * tree.discount_value
            ex = np.zeros(len(value), dtype=bool)
            if american:
                payoff = option.payoff_array(lv["S"])
                ex = payoff > value
                value = np.where(ex, payoff, value)
            lv["V"], lv["ex"] = value, ex
        self.nodes_priced = int(sum(len(lv["S"]) for lv in levels))
        return float(levels[0]["V"][0])

    def recursive_pricing(self) -> float:
        """Récursion depuis la racine : Node.price_with_child_recursive valorise d'abord les enfants, puis le nœud
        avec la même formule que le backward.

        N appels imbriqués : on relève la limite de récursion de Python (1 000 par défaut) et on lance le calcul
        dans un thread à grande pile (256 Mo), sinon la pile déborde dès quelques milliers de pas.
        """
        tree = self.tree
        if not isinstance(tree, TreeNode):
            raise ValueError("Le pricing récursif parcourt des objets Node : utilisez le moteur « objets ».")
        result = self._recursive_worker(tree)
        if "value" not in result:
            raise RuntimeError("Le pricing récursif a échoué (détail de l'erreur dans la console).")
        self.nodes_priced = self._count_nodes(tree)
        return float(result["value"])

    @staticmethod
    def _recursive_worker(tree: TreeNode) -> dict:
        """Exécute le parcours récursif avec une pile adaptée à la taille de l'arbre."""
        result = {}
        old_limit, old_stack = sys.getrecursionlimit(), threading.stack_size()
        sys.setrecursionlimit(max(old_limit, 4 * tree.n_steps + 1000))
        threading.stack_size(256 * 2 ** 20)
        try:
            worker = threading.Thread(target=lambda: result.update(value=tree.root.price_with_child_recursive()))
            worker.start()
            worker.join()
        finally:
            threading.stack_size(old_stack)
            sys.setrecursionlimit(old_limit)
        return result

    @staticmethod
    def _count_nodes(tree: TreeNode) -> int:
        """Compte les nœuds réellement parcourus, sans modifier les valeurs."""
        count = 0
        trunk = tree.last_trunk
        while trunk is not None:
            node = trunk.node_ref
            while node.down is not None:
                node = node.down
            while node is not None:
                count += 1
                node = node.up
            trunk = trunk.prec
        return count

    def price_options_vectorized(self, strikes: np.ndarray, is_call: np.ndarray,
                                 is_american: np.ndarray) -> np.ndarray:
        """Plusieurs calls / puts valorisés d'un coup sur la même grille (une colonne de V par option)."""
        tree = self.tree
        if not isinstance(tree, TreeVectorized):
            raise ValueError("Le pricing de plusieurs options demande un arbre vectorisé.")
        levels = tree.levels
        strikes_array = np.asarray(strikes, float)[None, :]
        sign = np.where(np.asarray(is_call, bool), 1.0, -1.0)[None, :]
        am = np.asarray(is_american, bool)[None, :]
        option_values = np.maximum(sign * (levels[-1]["S"][:, None] - strikes_array), 0.0)          # Gains à l’exercice à maturité
        for i in range(tree.n_steps, 0, -1):
            tr = tree.transitions(i)
            iu, im, idn = LevelVectorized.child_index(levels[i]["pos"], tr["k"], tr["has"])
            pu, pm, pd_ = (p[:, None] for p in tr["p"])
            option_values = (option_values[iu] * pu + option_values[im] * pm + option_values[idn] * pd_) * tree.discount_value
            if am.any():
                exercise = np.maximum(sign * (levels[i - 1]["S"][:, None] - strikes_array), 0.0)
                option_values = np.where(am & (exercise > option_values), exercise, option_values)
        return option_values[0]
