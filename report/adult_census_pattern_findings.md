# Adult Census Pattern Findings

This report summarises key patterns discovered in the Adult Census
Income dataset using **Brute Force** and **Apriori** frequent itemset
mining.

The underlying result files are:

- [results/adult_patterns_bruteforce.csv](../results/adult_patterns_bruteforce.csv)
- [results/adult_patterns_apriori.csv](../results/adult_patterns_apriori.csv)

Both algorithms were run on the full dataset with `min_support = 0.1`
and `max_length = 3`. Support values reported below are **relative**
frequencies in the interval $[0, 1]$.

> Important: These are **associations** in the data, not causal
> relationships.

---

## 1. High-Level Population Characteristics

From the top single-item patterns (length = 1):

- **Country of origin**
  - `native-country=United-States` has support ≈ **0.91**.
  - Interpretation: About 91% of individuals in the dataset are recorded
    as coming from the United States.
- **Race**
  - `race=White` has support ≈ **0.86**.
  - Interpretation: Roughly 86% of the population is labelled as White.
- **Income distribution**
  - `income=<=50K` has support ≈ **0.75**.
  - `income=>50K` has support ≈ **0.25**.
  - Interpretation: Around three quarters of records correspond to
    income `<=50K`, confirming that the dataset is **imbalanced** toward
    the lower-income class.
- **Workclass and sex**
  - `workclass=Private` has support ≈ **0.74**.
  - `sex=Male` has support ≈ **0.68**; `sex=Female` ≈ **0.32**.
  - Interpretation: Most individuals work in the private sector and
    there are roughly twice as many males as females in the dataset.
- **Marital status and relationship**
  - `marital-status=Married-civ-spouse` has support ≈ **0.47**.
  - `marital-status=Never-married` ≈ **0.32**.
  - `relationship=Husband` ≈ **0.41**; `relationship=Not-in-family`
    ≈ **0.26**; `relationship=Own-child` ≈ **0.15**.
  - Interpretation: A large portion of records are married individuals
    labelled as "Husband", followed by never-married and
    not-in-family/own-child relationships.
- **Education**
  - `education=HS-grad` has support ≈ **0.33**.
  - `education=Some-college` ≈ **0.22**; `education=Bachelors`
    ≈ **0.17**.
  - Interpretation: High school graduates form the largest single
    education category, followed by some college and then bachelor’s
    degrees.

These frequent 1-item patterns describe the dominant demographic
structure of the Adult dataset.

---

## 2. Demographic Co-Occurrence Patterns

### 2.1 Country, race, and workclass

- `native-country=United-States & race=White` has support ≈ **0.80**.
- `native-country=United-States & workclass=Private` ≈ **0.67**.
- `race=White & workclass=Private` ≈ **0.63**.

Interpretation:

- The majority of records are **White individuals from the United
  States**, and a large fraction of them work in the **private sector**.

### 2.2 Sex, marital status, and relationship

- `marital-status=Married-civ-spouse & sex=Male` ≈ **0.42**.
- `relationship=Husband & sex=Male` ≈ **0.41**.
- `marital-status=Married-civ-spouse & relationship=Husband` ≈ **0.41**.
- Triple pattern `marital-status=Married-civ-spouse & relationship=Husband & sex=Male` ≈ **0.41**.

Interpretation:

- Married male "Husbands" are a very large and coherent subgroup in the
  dataset, as expected given the coding of `relationship` and
  `marital-status`.

### 2.3 Education and race

- `education=HS-grad & race=White` ≈ **0.28**.
- `education=Some-college & race=White` ≈ **0.19**.
- `education=Bachelors & race=White` ≈ **0.15**.

Interpretation:

- Among White individuals, **High School graduates** form the largest
  education subgroup, followed by some-college and bachelor’s.

### 2.4 Occupation

- `occupation=Prof-specialty` support ≈ **0.13**.
- `occupation=Craft-repair` ≈ **0.13**.
- `occupation=Exec-managerial` ≈ **0.13**.
- `occupation=Adm-clerical` ≈ **0.12**.
- `occupation=Sales` ≈ **0.12**.
- `occupation=Other-service` ≈ **0.11**.

Interpretation:

- Professional specialty, craft/repair, and executive/managerial jobs
  are among the most common occupations, followed closely by
  administrative clerical and sales positions.

---

## 3. Income-Related Patterns

The following patterns include the income attribute and therefore
highlight demographic profiles associated with each income class.

### 3.1 Patterns associated with `income=<=50K`

Selected frequent itemsets:

- `income=<=50K` alone has support ≈ **0.75** (overall prevalence).
- `income=<=50K & native-country=United-States` ≈ **0.68**.
- `income=<=50K & race=White` ≈ **0.63**.
- `income=<=50K & workclass=Private` ≈ **0.58**.
- `income=<=50K & sex=Male` ≈ **0.46**; `income=<=50K & sex=Female`
  ≈ **0.29**.
- `income=<=50K & marital-status=Never-married` ≈ **0.31**.
- `income=<=50K & relationship=Not-in-family` ≈ **0.23**.
- `income=<=50K & relationship=Husband` ≈ **0.22**.
- Education combinations:
  - `income=<=50K & education=HS-grad` ≈ **0.27**.
  - `income=<=50K & education=Some-college` ≈ **0.18**.

Interpretation:

- Lower income (`<=50K`) is widespread across many demographic groups,
  particularly **US-born White individuals working in the private
  sector**.
- There is a substantial contribution from **never-married** and
  **not-in-family** relationships, indicating that many low-income
  records are not part of a married-couple household.
- A significant share of the lower-income group has **high school** or
  **some college** education.

### 3.2 Patterns associated with `income=>50K`

Selected frequent itemsets:

- `income=>50K` alone has support ≈ **0.25**.
- `income=>50K & native-country=United-States` ≈ **0.23**.
- `income=>50K & race=White` ≈ **0.23**.
- `income=>50K & marital-status=Married-civ-spouse` ≈ **0.21**.
- `income=>50K & sex=Male` ≈ **0.21**.
- `income=>50K & workclass=Private` ≈ **0.16**.
- Relationship-based patterns:
  - `income=>50K & relationship=Husband` ≈ **0.19**.
  - Triple `income=>50K & marital-status=Married-civ-spouse & relationship=Husband` ≈ **0.19**.

Interpretation:

- Higher income (`>50K`) is strongly associated with being **male,
  married (Married-civ-spouse), and coded as `relationship=Husband`**.
- Most higher-income individuals are still **White and US-born**.
- Private-sector work is still dominant among >50K earners, but with a
  higher concentration in **Exec-managerial**, **Prof-specialty**, and
  related occupations (visible by combining the income patterns with the
  high-support occupation patterns listed earlier).

---

## 4. Comparison of Brute Force vs Apriori Results

- For the chosen parameters (`min_support = 0.1`, `max_length = 3`),
  both **Brute Force** and **Apriori** produce **very similar sets of
  frequent itemsets** with identical support values.
- Brute Force explicitly enumerates many 3-item combinations (e.g.
  `native-country=United-States & race=White & workclass=Private`),
  while Apriori focuses on itemsets that survive the Apriori pruning
  step; with the same thresholds, both cover the main demographic and
  income-related trends.
- The agreement between supports in
  [adult_patterns_bruteforce.csv](../results/adult_patterns_bruteforce.csv)
  and [adult_patterns_apriori.csv](../results/adult_patterns_apriori.csv)
  validates the correctness of the Apriori implementation relative to
  the exhaustive baseline.

---

## 5. Summary

- The Adult dataset is **dominated by US-born White individuals**,
  largely working in the **private sector**.
- The majority of records correspond to **income `<=50K`**, and this
  class spans many demographic combinations.
- **Higher income (`>50K`)** is most frequently associated with
  **married male "Husbands"**, often in white-collar occupations.
- Education patterns highlight **HS-grad** as the most common level,
  with **Some-college** and **Bachelors** following.

These frequent itemsets provide a compact, interpretable summary of the
joint distribution of demographic attributes and income in the Adult
Census dataset, and can be used to complement decision-tree-based
analyses in this project.
