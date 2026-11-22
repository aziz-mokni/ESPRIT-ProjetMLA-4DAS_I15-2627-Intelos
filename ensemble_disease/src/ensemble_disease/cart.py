"""CART-style decision tree from scratch, following the course (Chap. 4 + annexes).

Splitting criterion: weighted Gini index, minimised (Gini(S) = 1 - sum p_i^2).
As in the annexes, nominal features give one branch per value, and a feature
already used on a path is not reused. Ties are broken by column order.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import pandas as pd


def gini(labels) -> float:
    """Gini impurity of a collection of labels."""
    labels = list(labels)
    n = len(labels)
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in Counter(labels).values())


def weighted_gini(df: pd.DataFrame, feature: str, target: str) -> float:
    """Weighted Gini of splitting `df` on `feature`."""
    n = len(df)
    return sum(len(g) / n * gini(g[target]) for _, g in df.groupby(feature))


def gini_table(df: pd.DataFrame, features: list, target: str) -> dict:
    return {f: weighted_gini(df, f, target) for f in features}


@dataclass
class Node:
    feature: str | None = None
    label: object = None                 # majority class at this node
    counts: dict = field(default_factory=dict)
    children: dict = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class CARTClassifier:
    def __init__(self, max_depth: int | None = None):
        self.max_depth = max_depth
        self.root: Node | None = None
        self.split_log: list = []  # (path, gini table) for teaching / report

    def fit(self, df: pd.DataFrame, target: str) -> "CARTClassifier":
        self.target = target
        feats = [c for c in df.columns if c != target]
        self.split_log = []
        self.root = self._build(df, feats, depth=0, path="root")
        return self

    def _build(self, df, feats, depth, path) -> Node:
        counts = dict(Counter(df[self.target]))
        majority = max(counts, key=counts.get)
        node = Node(label=majority, counts=counts)
        if len(counts) == 1 or not feats or (self.max_depth is not None and depth >= self.max_depth):
            return node
        table = gini_table(df, feats, self.target)
        self.split_log.append((path, table))
        best = min(feats, key=lambda f: (table[f], feats.index(f)))  # first wins ties
        node.feature = best
        rest = [f for f in feats if f != best]
        for value, sub in df.groupby(best):
            node.children[value] = self._build(sub, rest, depth + 1, f"{path}/{best}={value}")
        return node

    def predict_one(self, row):
        node = self.root
        while not node.is_leaf:
            nxt = node.children.get(row[node.feature])
            if nxt is None:          # unseen value -> majority class at this node
                return node.label
            node = nxt
        return node.label

    def predict(self, df: pd.DataFrame) -> list:
        return [self.predict_one(r) for _, r in df.iterrows()]

    def score(self, df: pd.DataFrame) -> float:
        pred = self.predict(df.drop(columns=[self.target]))
        return sum(p == t for p, t in zip(pred, df[self.target])) / len(df)

    def rules(self) -> str:
        """Human-readable IF ... THEN rules (course: tree -> decision rules)."""
        out: list = []

        def walk(node, conds):
            if node.is_leaf:
                out.append(f"IF {' AND '.join(conds) or 'True'} THEN {self.target} = {node.label}  {node.counts}")
                return
            for v, ch in node.children.items():
                walk(ch, conds + [f"{node.feature} = {v}"])

        walk(self.root, [])
        return "\n".join(out)
