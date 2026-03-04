import pandas as pd
from typing import Iterable, List, Set

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_adult_transactions(limit: int | None = None) -> List[Set[str]]:
    """Load the Adult Census dataset as a list of transactions.

    Each row is mapped to a set of "column=value" strings for a subset of
    interpretable (mainly categorical) attributes. This format is suitable
    for frequent itemset / association pattern mining algorithms.

    Parameters
    ----------
    limit: optional int
        If provided, only the first `limit` rows are used. This can be
        helpful when experimenting with very low support thresholds.
    """

    df = pd.read_csv(ADULT_URL, names=ADULT_COLUMNS, na_values=" ?")

    # Drop rows with missing values in the core categorical attributes we use
    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "income",
    ]
    df = df.dropna(subset=categorical_columns)

    if limit is not None:
        df = df.head(limit)

    transactions: List[Set[str]] = []

    for _, row in df[categorical_columns].iterrows():
        items: Set[str] = set()
        for col in categorical_columns:
            value = str(row[col]).strip()
            items.add(f"{col}={value}")
        transactions.append(items)

    return transactions
