from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import ceil
from typing import Dict, Iterable, List, Set, FrozenSet


Transaction = Set[str]
Itemset = FrozenSet[str]


def _compute_min_count(min_support: float | int, n_transactions: int) -> int:
    if n_transactions <= 0:
        return 0
    if min_support < 1:
        return max(1, int(ceil(min_support * n_transactions)))
    return int(min_support)


def _generate_candidates(prev_frequents: List[Itemset], k: int) -> List[Itemset]:
    """Generate size-k candidates from size-(k-1) frequent itemsets.

    Standard Apriori join-and-prune step: two sets are joined if their
    first k-2 items are identical; resulting candidate is kept only if
    all size-(k-1) subsets are known to be frequent.
    """

    candidates: List[Itemset] = []
    prev_frequents_sorted = [sorted(fs) for fs in prev_frequents]
    prev_frequents_set = set(prev_frequents)

    n = len(prev_frequents_sorted)
    for i in range(n):
        for j in range(i + 1, n):
            a = prev_frequents_sorted[i]
            b = prev_frequents_sorted[j]
            if a[: k - 2] != b[: k - 2]:
                break
            candidate_items = frozenset(a).union(b)
            if len(candidate_items) != k:
                continue

            # Prune step: all (k-1)-subsets must be frequent
            all_subsets_frequent = True
            for subset in combinations(candidate_items, k - 1):
                if frozenset(subset) not in prev_frequents_set:
                    all_subsets_frequent = False
                    break
            if all_subsets_frequent:
                candidates.append(candidate_items)

    return candidates


def apriori_frequent_itemsets(
    transactions: Iterable[Transaction],
    min_support: float | int,
    max_length: int | None = None,
) -> Dict[Itemset, float]:
    """Apriori frequent itemset mining.

    Parameters
    ----------
    transactions:
        Iterable of sets, where each set represents a transaction.
    min_support:
        Minimum support threshold. If < 1, interpreted as relative
        support (fraction). If >= 1, treated as an absolute count.
    max_length:
        Maximum itemset size to explore. If ``None``, continues until no
        larger frequent itemsets exist.

    Returns
    -------
    dict
        Mapping ``frozenset(items) -> support`` where support is the
        relative frequency in [0, 1].
    """

    transactions_list: List[Transaction] = [set(t) for t in transactions]
    n_transactions = len(transactions_list)

    if n_transactions == 0:
        return {}

    min_count = _compute_min_count(min_support, n_transactions)

    # L1: frequent 1-itemsets
    item_counts: defaultdict[str, int] = defaultdict(int)
    for t in transactions_list:
        for item in t:
            item_counts[item] += 1

    Lk: Dict[Itemset, float] = {
        frozenset([item]): count / n_transactions
        for item, count in item_counts.items()
        if count >= min_count
    }

    frequent_itemsets: Dict[Itemset, float] = {}
    k = 1

    while Lk:
        frequent_itemsets.update(Lk)

        if max_length is not None and k >= max_length:
            break

        prev_frequents = list(Lk.keys())
        k += 1

        candidates = _generate_candidates(prev_frequents, k)
        if not candidates:
            break

        candidate_counts: defaultdict[Itemset, int] = defaultdict(int)
        for t in transactions_list:
            for candidate in candidates:
                if candidate.issubset(t):
                    candidate_counts[candidate] += 1

        Lk = {
            fs: count / n_transactions
            for fs, count in candidate_counts.items()
            if count >= min_count
        }

    return frequent_itemsets
