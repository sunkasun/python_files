from __future__ import annotations


class Node:
    __slots__ = (
        "time", "position", "und_price", "option_value", "is_exerced",
        "tree", "probas_cum", "up", "down", "next_up", "next_middle",
        "next_down", "p_up", "p_middle", "p_down",
    )

    def __init__(self, und_price: float, time: int, position: int,
                 probas_cum: float, tree: Tree, is_exerced: bool = False) -> None:
        self.time = time
        self.position = position
        self.und_price = float(und_price)
        self.option_value = None
        self.is_exerced = is_exerced
        self.tree = tree
        self.probas_cum = float(probas_cum)
        self.up = None
        self.down = None
        self.next_up = None
        self.next_middle = None
        self.next_down = None
        self.p_up = None
        self.p_middle = None
        self.p_down = None

    def _continuation_value(self) -> float:
        """Actualise les valeurs des enfants présents sur l'arbre."""
        value = 0.0
        has_child = False
        for child, probability in (
            (self.next_up, self.p_up),
            (self.next_middle, self.p_middle),
            (self.next_down, self.p_down),
        ):
            if child is not None:
                value += child.option_value * probability
                has_child = True
        if not has_child:
            raise ValueError("Nœud non terminal sans enfant : arbre invalide.")
        return value * self.tree.discount_value

    def price_with_child(self) -> float:
        """Valorise le nœud et mémorise sa décision d'exercice."""
        tree = self.tree
        option = tree.option
        if option is None:
            raise ValueError("Aucun objet option n'est associé à cet arbre.")
        payoff = option.payoff(self.und_price)
        if self.time == tree.n_steps:
            self.option_value = payoff
            self.is_exerced = option.is_american() and payoff > 0.0
            return payoff

        value = self._continuation_value()
        self.is_exerced = option.is_american() and payoff > value
        if self.is_exerced:
            value = payoff
        self.option_value = float(value)
        return value

    # Le parcours récursif réutilise la même formule de valorisation.
    def price_with_child_recursive(self) -> float:
        if self.option_value is not None:
            return self.option_value
        for child in (self.next_up, self.next_middle, self.next_down):
            if child is not None and child.option_value is None:
                child.price_with_child_recursive()
        return self.price_with_child()

    # Les deux moments utilisent les seules branches effectivement présentes.
    def esperance_with_child(self) -> float:
        mean = 0.0
        for child, probability in ((self.next_up, self.p_up), (self.next_middle, self.p_middle),
                                   (self.next_down, self.p_down)):
            if child is not None and probability:
                mean += child.und_price * probability
        return mean

    def variance_with_child(self) -> float:
        mean = self.esperance_with_child()
        second = 0.0
        for child, probability in ((self.next_up, self.p_up), (self.next_middle, self.p_middle),(self.next_down, self.p_down)):
            if child is not None and probability:
                second += child.und_price ** 2 * probability
        return second - mean * mean


class Trunk:
    """Repère d'un niveau, sans les attributs superflus d'un nœud."""
    __slots__ = ("node_ref", "_forward", "_time", "prec")
    def __init__(self, node_ref: Node | None = None, forward: float | None = None,
                 time: int | None = None, prec_trunk: Trunk | None = None) -> None:
        self.node_ref = node_ref
        self._forward = forward
        self._time = time
        self.prec = prec_trunk
