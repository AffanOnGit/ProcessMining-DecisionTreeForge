# Adult Census Patterns Report

This document accompanies the pattern-mining components implemented in
`src/pattern_mining/` and is intended to be used together with the
exported CSV files produced by the Adult Census experiment script.

The goal is to **identify and interpret patterns** in the Adult Census
Income dataset using two classic algorithms:

- **Brute Force frequent itemset mining**
- **Apriori frequent itemset mining**

Both algorithms operate on a transactional representation of the dataset,
where each row is mapped to items of the form `column=value` (e.g.
`education=Bachelors`, `sex= Male`, `income=>50K`).

---

## 1. How to Run the Adult Pattern Experiment

From the project root, ensure your virtual environment is activated and
required packages are installed (see README). Then run:

```powershell
.\.venv\Scripts\python.exe -m src.pattern_mining.adult_patterns_experiment \
    --min-support 0.1 \
    --max-length 3 \
    --algorithms brute apriori
```

Key arguments:

- `--limit` – maximum number of Adult rows to use. A smaller limit runs
  faster; by default (`0`) all rows are used.
- `--min-support` – relative minimum support threshold in `[0, 1]`.
  Higher values keep only very common patterns; lower values surface
  more specific/rare patterns (but can be slower).
- `--max-length` – maximum number of items allowed in a pattern.
- `--algorithms` – choose any subset of `{brute, apriori}`.

The script writes its outputs to the `results/` directory:

- `results/adult_patterns_bruteforce.csv`
- `results/adult_patterns_apriori.csv`

---

## 2. CSV File Structure

Each CSV contains one row per discovered **frequent itemset** with the
following columns:

- `algorithm` – which algorithm produced the pattern
  (`brute_force` or `apriori`).
- `pattern` – a human-readable conjunction of items, e.g.
  `sex= Male & marital-status=Married-civ-spouse & income=>50K`.
- `length` – number of items in the pattern.
- `support` – relative support in `[0, 1]`, defined as:

  $$\text{support}(X) = \frac{\text{# transactions containing all items in } X}{\text{total # transactions}}$$

Patterns are sorted in descending order of support (and, for ties, by
length) so that the **most common patterns appear first**.

---

## 3. Interpreting the Patterns

Each row in the Adult dataset represents an individual, with attributes
such as `age`, `education`, `marital-status`, `occupation`, `race`,
`sex`, `native-country`, and their binary `income` class (e.g. `<=50K`
vs `>50K`).

The mined itemsets reveal **combinations of attribute-values that occur
frequently together**. For example, you may observe patterns like:

- `sex= Male & marital-status=Married-civ-spouse`
- `education=Bachelors & income=>50K`
- `relationship=Not-in-family & income=<=50K`

Interpreting such patterns typically involves:

- **Prevalence:** Higher `support` means the pattern describes a larger
  portion of the population in the dataset.
- **Specificity vs generality:** Shorter patterns (length 1–2) often
  capture broad demographic trends; longer patterns can highlight more
  specific subgroups.
- **Income association:** Including the `income` item in a pattern lets
  you reason about which combinations of demographic attributes are most
  commonly associated with `>50K` or `<=50K` income.

> Note: These patterns describe associations in the dataset, not causal
> relationships. They should be interpreted with care and, ideally,
> cross-checked with domain knowledge and further analysis.

---

## 4. Suggested Analysis Workflow

1. **Run the experiment script** with a moderate `--min-support` (e.g.
   `0.1`) and `--max-length` (e.g. `3`) to obtain initial CSV files.
2. **Open the CSVs** (e.g. in a spreadsheet, Python, or R) and inspect
   the highest-support patterns.
3. **Filter by patterns containing `income`** to focus on income-related
   associations.
4. **Compare Brute Force vs Apriori outputs:**
   - With the same parameters, both should return the **same set of
     frequent itemsets**, but Apriori is more efficient.
   - Differences typically indicate parameter or implementation choices.
5. **Refine thresholds:** Lower `--min-support` or increase
   `--max-length` if you need more detailed patterns, keeping in mind
   the computational cost.

Use this document to summarise key patterns you observe (e.g., listing
high-support itemsets and their interpretation) once you have generated
and inspected the CSV outputs.
