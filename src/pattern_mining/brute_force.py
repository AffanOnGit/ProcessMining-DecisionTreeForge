from __future__ import annotations

from itertools import combinations
from math import ceil
from typing import Dict, Iterable, List, Set, FrozenSet


Transaction = Set[str]
Itemset = FrozenSet[str]


def _compute_min_count(min_support: float | int, n_transactions: int) -> int:
    """Convert a support threshold into an absolute minimum count.

    - If `min_support` < 1, it is treated as a fraction of the dataset.
    - If `min_support` >= 1, it is treated as an absolute count.
    """

    if n_transactions <= 0:
        return 0

    if min_support < 1:
        return max(1, int(ceil(min_support * n_transactions)))
    return int(min_support)


def brute_force_frequent_itemsets(
    transactions: Iterable[Transaction],
    min_support: float | int,
    max_length: int | None = None,
) -> Dict[Itemset, float]:
    """Brute-force frequent itemset mining.

    Parameters
    ----------
    transactions:
        Iterable of sets, where each set represents a transaction.
    min_support:
        Minimum support threshold. If < 1, it is interpreted as a
        relative fraction (e.g. 0.1 for 10%). If >= 1, it is interpreted
        as an absolute minimum count.
    max_length:
        Maximum itemset size to explore. If ``None``, all possible sizes
        are considered (which may be very expensive for large item
        universes).

    Returns
    -------
    dict
        Mapping ``frozenset(items) -> support`` where support is the
        relative frequency in [0, 1].
    """

    # Materialise transactions once because we need multiple passes
    transactions_list: List[Transaction] = [set(t) for t in transactions]
    n_transactions = len(transactions_list)

    if n_transactions == 0:
        return {}

    # Universe of all items
    all_items: Set[str] = set()
    for t in transactions_list:
        all_items.update(t)

    if not all_items:
        return {}

    if max_length is None:
        max_length = len(all_items)

    min_count = _compute_min_count(min_support, n_transactions)

    frequent_itemsets: Dict[Itemset, float] = {}

    # Try every possible item combination up to max_length
    for k in range(1, max_length + 1):
        for combo in combinations(sorted(all_items), k):
            candidate = frozenset(combo)
            count = 0
            for t in transactions_list:
                if candidate.issubset(t):
                    count += 1
                    # simple early stopping if we already know it will be frequent
                    if count >= min_count and min_support >= 1:
                        break
            if count >= min_count:
                frequent_itemsets[candidate] = count / n_transactions

    return frequent_itemsets
