from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, FrozenSet, Set

import pandas as pd

from .adult_loader import load_adult_transactions
from .brute_force import brute_force_frequent_itemsets
from .apriori import apriori_frequent_itemsets


Transaction = Set[str]
Itemset = FrozenSet[str]


def _itemsets_to_dataframe(
    itemsets: Dict[Itemset, float],
    algorithm: str,
) -> pd.DataFrame:
    """Convert a dictionary of itemsets to a tabular representation.

    Columns:
    - algorithm: name of the mining algorithm used
    - pattern: human-readable conjunction of items (e.g. "sex= Male & income=>50K")
    - length: number of items in the pattern
    - support: relative support in [0, 1]
    """

    rows = []
    for items, support in itemsets.items():
        sorted_items = sorted(items)
        pattern_str = " & ".join(sorted_items)
        rows.append(
            {
                "algorithm": algorithm,
                "pattern": pattern_str,
                "length": len(items),
                "support": support,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["support", "length"], ascending=[False, False])
    return df


def run_adult_pattern_mining(
    limit: int | None,
    min_support: float,
    max_length: int | None,
    algorithms: list[str],
    output_dir: str = "results",
) -> None:
    """Run Brute Force and/or Apriori on the Adult Census dataset.

    Parameters
    ----------
    limit:
        Optional upper limit on the number of Adult rows to use.
    min_support:
        Relative minimum support threshold in [0, 1].
    max_length:
        Maximum itemset size to explore (None = no explicit cap).
    algorithms:
        List containing any of {"brute", "apriori"}.
    output_dir:
        Directory into which CSV result files are written.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading Adult Census dataset as transactions ...")
    transactions = load_adult_transactions(limit=limit)
    n_tx = len(transactions)
    print(f"[INFO] Loaded {n_tx} transactions.")

    if "brute" in algorithms:
        print("[INFO] Running Brute Force frequent itemset mining ...")
        bf_itemsets = brute_force_frequent_itemsets(
            transactions,
            min_support=min_support,
            max_length=max_length,
        )
        print(f"[INFO] Found {len(bf_itemsets)} frequent itemsets (Brute Force).")
        bf_df = _itemsets_to_dataframe(bf_itemsets, algorithm="brute_force")
        bf_path = os.path.join(output_dir, "adult_patterns_bruteforce.csv")
        bf_df.to_csv(bf_path, index=False)
        print(f"[INFO] Brute Force patterns saved to: {bf_path}")

    if "apriori" in algorithms:
        print("[INFO] Running Apriori frequent itemset mining ...")
        ap_itemsets = apriori_frequent_itemsets(
            transactions,
            min_support=min_support,
            max_length=max_length,
        )
        print(f"[INFO] Found {len(ap_itemsets)} frequent itemsets (Apriori).")
        ap_df = _itemsets_to_dataframe(ap_itemsets, algorithm="apriori")
        ap_path = os.path.join(output_dir, "adult_patterns_apriori.csv")
        ap_df.to_csv(ap_path, index=False)
        print(f"[INFO] Apriori patterns saved to: {ap_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mine patterns on the Adult Census dataset using Brute Force "
            "and Apriori algorithms."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Optional limit on number of Adult rows to use. "
            "By default (0), all rows are used. "
            "Use a smaller positive value for quicker experimentation."
        ),
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.1,
        help="Relative minimum support threshold in [0, 1] (default: 0.1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=3,
        help=(
            "Maximum itemset length to explore (default: 3). "
            "Larger values can grow exponentially in runtime."
        ),
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["brute", "apriori"],
        choices=["brute", "apriori"],
        help="Which algorithms to run: brute, apriori (default: both)",
    )

    args = parser.parse_args()

    limit = args.limit if args.limit and args.limit > 0 else None

    run_adult_pattern_mining(
        limit=limit,
        min_support=args.min_support,
        max_length=args.max_length,
        algorithms=args.algorithms,
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
