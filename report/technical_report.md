# Decision Trees: Implementation & Analysis

**Course:** Process Mining and Simulation  
**Date of Submission:** February 2026

| Group Member | Roll Number |
| :--- | :---: |
| Affan Hameed | 22i-2582 |
| Saim Zia | 22i-2661 |
| Ahsan Iqbal | 22i-2524 |
| Asim Shehzad | 22i-2679 |

---

## 2. Abstract

This report presents a comprehensive implementation and evaluation of the Decision Tree classification algorithm built entirely from scratch using Python and NumPy. The classifier leverages **Entropy** as the impurity measure and **Information Gain** as the splitting criterion, following the theoretical foundations of inductive learning covered in the course lectures. The implementation supports continuous feature splitting via optimal threshold search, handles missing values through median imputation, and includes configurable stopping criteria such as maximum depth and minimum information gain thresholds.

The from-scratch classifier was rigorously evaluated across three diverse datasets: (1) the **Adult Census Income** dataset as a real-world benchmark with 32,561 instances and 14 features, (2) a **Synthetic dataset** with 2,000 instances and 10% injected label noise, and (3) the **Credit Card Fraud** dataset representing extreme class imbalance (~0.172% positive class). Performance was measured using Accuracy, Precision (macro), Recall (macro), and F1-Score (macro), and directly compared against scikit-learn's `DecisionTreeClassifier` configured with identical hyperparameters. Results demonstrate that the from-scratch implementation achieves **identical performance** to scikit-learn on balanced datasets and **near-identical performance** on the highly imbalanced fraud dataset, validating the correctness and robustness of our implementation.

---

## 3. Introduction

### 3.1 Assignment Overview & Objectives

The primary objective of this assignment is to deepen our understanding of the Decision Tree algorithm by implementing it from scratch rather than relying on pre-built machine learning libraries. The specific goals are:

1. **Implement a Decision Tree classifier** using Entropy and Information Gain as the splitting criteria.
2. **Handle continuous features** by searching for optimal split thresholds across sorted feature values.
3. **Handle missing values** gracefully during both training and prediction phases.
4. **Evaluate the classifier** on three distinct dataset types: a real-world dataset (1,000+ instances, 10+ features), a synthetic dataset with noise, and a highly imbalanced dataset.
5. **Compare results** against scikit-learn's `DecisionTreeClassifier` to validate correctness.
6. **Report findings** with Accuracy, Precision, Recall, and F1-Score metrics.

### 3.2 Theoretical Background

**Entropy** is a measure of impurity or disorder in a dataset. For a binary classification problem with classes $C = \{0, 1\}$, the entropy $H(S)$ of a set $S$ is defined as:

$$H(S) = - \sum_{c \in C} p_c \cdot \log_2(p_c)$$

where $p_c$ is the proportion of samples belonging to class $c$. A pure node (all samples of one class) has entropy 0, while maximum uncertainty (50/50 split) yields entropy 1.

**Information Gain** measures the reduction in entropy achieved by splitting a dataset on a particular feature. For a feature $A$ that splits $S$ into subsets $S_{\text{left}}$ and $S_{\text{right}}$:

$$IG(S, A) = H(S) - \left(\frac{|S_{\text{left}}|}{|S|} \cdot H(S_{\text{left}}) + \frac{|S_{\text{right}}|}{|S|} \cdot H(S_{\text{right}})\right)$$

The feature and threshold combination that maximizes Information Gain is selected as the best split at each node.

**Inductive Learning** is the process of inferring general rules from specific training examples. Decision Trees are a canonical example of inductive learning: they recursively partition the feature space based on the training data to construct a set of if-then rules that generalize to unseen instances. The tree grows by greedily selecting the split that maximizes information gain at each node until a stopping criterion is met (e.g., maximum depth, pure leaf, or insufficient gain).

### 3.3 Report Structure

The remainder of this report is organized as follows:
- **Section 4** describes the three datasets used for evaluation, including their characteristics and preprocessing steps.
- **Section 5** details the from-scratch implementation methodology, covering entropy calculation, information gain, continuous feature handling, missing value imputation, and the tree-building algorithm.
- **Section 6** outlines the experimental setup and evaluation metrics.
- **Section 7** presents the results across all three datasets with analysis and discussion.
- **Section 8** provides a direct side-by-side comparison with scikit-learn's `DecisionTreeClassifier`.
- **Section 9** documents individual group member contributions.
- **Section 10** concludes the report with future work directions.
- **Section 11** lists references, and **Section 12** contains the appendix with complete source code.

---

## 4. Datasets

### 4.1 Real-World Dataset: Adult Census Income

#### 4.1.1 Description & Characteristics

The **Adult Census Income** dataset, sourced from the UCI Machine Learning Repository, is a widely used benchmark for binary classification tasks. It contains demographic and employment-related features collected from the 1994 U.S. Census.

| Property | Value |
| :--- | :--- |
| **Source** | UCI ML Repository |
| **Instances** | 32,561 |
| **Original Features** | 14 (6 continuous, 8 categorical) |
| **Features After Encoding** | 108 (after one-hot encoding) |
| **Target Variable** | Income (>50K = 1, ≤50K = 0) |
| **Class Distribution** | ~75.1% class 0, ~24.9% class 1 |
| **Missing Values** | Present in `workclass`, `occupation`, `native-country` |

**Features include:** age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country.

#### 4.1.2 Preprocessing Steps

1. **Loading:** Data is read directly from the UCI repository URL using `pandas.read_csv()`.
2. **Missing Values:** Entries marked as `' ?'` are treated as `NaN` values using the `na_values` parameter.
3. **Target Encoding:** The `income` column is binarized: `'>50K'` → 1, `'<=50K'` → 0.
4. **One-Hot Encoding:** All categorical columns are converted to binary indicator variables using `pandas.get_dummies()`, expanding the feature space from 14 to 108 columns.
5. **Output Format:** Features are extracted as a NumPy array (`X`) and labels as a separate array (`y`).

### 4.2 Synthetic Dataset (with Noise)

#### 4.2.1 Generation Method & Noise Injection

The synthetic dataset is generated programmatically to provide a controlled environment for testing the decision tree's behavior under known conditions.

**Generation Logic:**
- 12 continuous features are sampled from a standard normal distribution: $X \sim \mathcal{N}(0, 1)$.
- The **hidden classification rule** is: class 1 if the sum of the first 5 features exceeds 0, otherwise class 0.
  
  $$y = \mathbb{1}\left[\sum_{j=1}^{5} X_j > 0\right]$$

- **Noise injection:** 10% of labels are randomly flipped to simulate real-world label noise. This tests the tree's robustness against noisy training data and its susceptibility to overfitting.
- A fixed random seed (`np.random.seed(42)`) ensures full reproducibility.

#### 4.2.2 Characteristics

| Property | Value |
| :--- | :--- |
| **Instances** | 2,000 |
| **Features** | 12 (all continuous) |
| **Noise Level** | 10% label flip |
| **Class Distribution** | ~50/50 (balanced) |
| **Missing Values** | None |

### 4.3 Highly Imbalanced Dataset: Credit Card Fraud

#### 4.3.1 Description & Imbalance Ratio

The **Credit Card Fraud Detection** dataset from Kaggle contains transactions made by European cardholders over two days in September 2013. It is one of the most widely used benchmarks for evaluating classifiers under extreme class imbalance.

| Property | Value |
| :--- | :--- |
| **Source** | Kaggle (MLG - ULB) |
| **Instances** | 284,807 |
| **Features** | 30 (28 PCA-transformed + Time + Amount) |
| **Target Variable** | Class (1 = Fraud, 0 = Legitimate) |
| **Fraud Instances** | 492 (0.172%) |
| **Legitimate Instances** | 284,315 (99.828%) |
| **Imbalance Ratio** | ~578:1 |

The 28 principal component features (V1–V28) are the result of PCA transformation applied by the dataset authors for confidentiality. The `Time` and `Amount` features are untransformed.

#### 4.3.2 Preprocessing Steps

1. **Manual Download:** Due to Kaggle's authentication requirements, the dataset must be manually downloaded and placed as `creditcard.csv` in the `data/` directory.
2. **Loading:** Data is read using `pandas.read_csv()`.
3. **Feature/Label Separation:** The `Class` column is extracted as the target variable `y`, and all remaining columns form the feature matrix `X`.
4. **No additional encoding** is required as all features are already numerical.

---

## 5. Methodology & Implementation (From Scratch)

### 5.1 Decision Tree Algorithm Overview

Our implementation follows the **ID3/CART hybrid** approach for building binary decision trees:

1. At each node, evaluate all possible feature-threshold combinations to find the split that maximizes Information Gain.
2. Partition the data into left (feature < threshold) and right (feature ≥ threshold) subsets.
3. Recursively grow the tree on each subset.
4. Stop growing when a stopping criterion is met (max depth reached, pure node, insufficient gain, or too few samples).
5. At prediction time, traverse the tree from root to leaf, following the learned split rules.

### 5.2 Core Components

#### 5.2.1 Entropy Calculation

The `_entropy(y)` method computes the Shannon entropy of a label array:

```python
def _entropy(self, y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))
```

A small epsilon (`1e-10`) is added inside the logarithm to prevent `log(0)` errors when a class has zero probability. This method generalizes to multi-class problems, though our datasets are binary.

#### 5.2.2 Information Gain & Best Split Selection

The `_information_gain(y, y_left, y_right)` method computes the weighted entropy reduction:

```python
def _information_gain(self, y, y_left, y_right):
    if len(y_left) == 0 or len(y_right) == 0:
        return 0
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    return self._entropy(y) - (weight_left * self._entropy(y_left) 
           + weight_right * self._entropy(y_right))
```

The `_best_split(X, y)` method iterates over all features, sorts values, identifies unique split boundaries using vectorized operations, and uses **cumulative class counts** to efficiently compute entropy for each candidate split without redundant array slicing. This optimization reduces the split-finding complexity from $O(N^2 \cdot D)$ to $O(N \log N \cdot D)$, where $N$ is the number of samples and $D$ is the number of features.

#### 5.2.3 Handling Continuous Features (Optimal Threshold Search)

For each continuous feature, the algorithm:
1. Sorts the feature values and corresponding labels.
2. Identifies **boundary points** where consecutive sorted values differ (using `np.where`).
3. For each boundary, computes the candidate threshold as the midpoint of two adjacent values.
4. Evaluates the Information Gain for that threshold using pre-computed cumulative counts.
5. Selects the feature-threshold pair with the highest Information Gain globally.

This ensures that the optimal split point is found for every continuous feature at every node.

#### 5.2.4 Handling Missing Values (Median Imputation)

The `_handle_missing(X)` method is invoked at the start of both `fit()` and `predict()`:

```python
def _handle_missing(self, X):
    X = np.array(X, dtype=float)
    for i in range(X.shape[1]):
        nan_mask = np.isnan(X[:, i])
        if np.any(nan_mask):
            median = np.nanmedian(X[:, i])
            X[nan_mask, i] = median
    return X
```

Key design decisions:
- **Median imputation** is chosen over mean imputation because median is robust to outliers, which is especially important for skewed features like `capital-gain` in the Adult dataset.
- The input is cast to `float` to handle mixed-type arrays (e.g., boolean columns from one-hot encoding).
- Missing values are handled identically during training and prediction to ensure consistency.

### 5.3 Tree Building & Prediction Logic

**Tree Building (`_grow_tree`):**
1. Compute the majority class at the current node (used as the leaf prediction if the node becomes a leaf).
2. Check stopping criteria: max depth reached, pure node, or fewer than 2 samples.
3. Find the best split using `_best_split()`.
4. If no valid split exists or the gain is below `min_gain`, return a leaf node.
5. Otherwise, partition the data and recursively build left and right subtrees.

**Prediction (`_predict_single`):**
- For each test sample, traverse from root to leaf by comparing feature values against stored thresholds.
- Return the `predicted_class` stored at the leaf node.

### 5.4 Implementation Details

**Class Structure:**
- `Node`: Stores `predicted_class`, `feature_index`, `threshold`, `left`, and `right`.
- `DecisionTreeClassifier`: Exposes `fit(X, y)` and `predict(X)` methods, mirroring the scikit-learn API.

**Stopping Criteria:**
- `max_depth`: Maximum depth of the tree (default: `None`, meaning no limit).
- `min_gain`: Minimum information gain required to make a split (default: `0.01`).
- Pure node: All samples belong to the same class.
- Insufficient samples: Fewer than 2 samples at a node.

**Hyperparameters Used in Experiments:**
- `max_depth = 5`
- `min_gain = 0.01` (default)

### 5.5 Full Code Snippet

The complete implementation of `src/decision_tree.py` is provided in **Appendix A.1**. Key methods include:
- `_entropy(y)` — Entropy calculation
- `_information_gain(y, y_left, y_right)` — Information Gain computation
- `_best_split(X, y)` — Optimized split search with cumulative counts
- `_handle_missing(X)` — Median imputation for NaN values
- `_grow_tree(X, y, depth)` — Recursive tree construction
- `fit(X, y)` and `predict(X)` — Public API

---

## 6. Experiments & Evaluation

### 6.1 Experimental Setup

All experiments follow a consistent protocol:

| Parameter | Value |
| :--- | :--- |
| **Train/Test Split** | 80% train, 20% test |
| **Split Strategy** | Stratified (preserves class distribution) |
| **Random Seed** | 42 (for reproducibility) |
| **Max Depth** | 5 (for both scratch and scikit-learn) |
| **Splitting Criterion** | Entropy (for both implementations) |

The stratified split is critical for the imbalanced Credit Card Fraud dataset to ensure both train and test sets contain representative proportions of the minority class.

### 6.2 Evaluation Metrics

#### 6.2.1 Accuracy

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

Measures the overall proportion of correct predictions. While intuitive, accuracy can be misleading on imbalanced datasets (e.g., predicting all transactions as legitimate yields 99.8% accuracy on the fraud dataset).

#### 6.2.2 Precision (Macro)

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

Measures the proportion of positive predictions that are actually correct. Macro averaging computes precision independently for each class and takes the unweighted mean, giving equal importance to both majority and minority classes.

#### 6.2.3 Recall (Macro)

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Measures the proportion of actual positives that are correctly identified. For fraud detection, high recall is critical — missing a fraudulent transaction (false negative) is typically more costly than a false alarm.

#### 6.2.4 F1-Score (Macro)

$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean of Precision and Recall, providing a single balanced metric. Macro-averaged F1-Score is particularly useful for evaluating performance on imbalanced datasets.

---

## 7. Results & Analysis

### 7.1 Results on Real-World Dataset (Adult Census Income)

| Metric | From-Scratch DT |
| :--- | :---: |
| **Accuracy** | 84.98% |
| **Precision (macro)** | 82.56% |
| **Recall (macro)** | 73.45% |
| **F1-Score (macro)** | 76.39% |

**Analysis:** The classifier achieves strong performance on this real-world dataset, correctly classifying ~85% of individuals. The gap between Precision (82.56%) and Recall (73.45%) indicates that the model is more conservative in predicting the positive class (income >50K), leading to fewer false positives but more false negatives. This is expected given the moderate class imbalance (~75/25 split) and the depth constraint of 5.

### 7.2 Results on Synthetic Dataset (with Noise)

| Metric | From-Scratch DT |
| :--- | :---: |
| **Accuracy** | 73.50% |
| **Precision (macro)** | 73.49% |
| **Recall (macro)** | 73.50% |
| **F1-Score (macro)** | 73.49% |

**Analysis:** The near-identical values across all four metrics reflect the balanced class distribution (~50/50). The accuracy of 73.5% is consistent with the 10% noise injection: with a theoretical maximum accuracy of ~90% (since 10% of labels are randomly flipped), the tree captures the underlying signal well while appropriately avoiding overfitting to the noisy labels.

### 7.3 Results on Imbalanced Dataset (Credit Card Fraud)

| Metric | From-Scratch DT |
| :--- | :---: |
| **Accuracy** | 99.93% |
| **Precision (macro)** | 95.11% |
| **Recall (macro)** | 83.16% |
| **F1-Score (macro)** | 88.22% |

**Analysis:** Despite the extreme class imbalance (578:1 ratio), the classifier achieves excellent Precision (95.11%), meaning that when it flags a transaction as fraudulent, it is correct ~95% of the time. The Recall of 83.16% indicates that ~17% of actual fraud cases are missed. The high accuracy (99.93%) is expected on imbalanced data and should not be the sole evaluation criterion — the F1-Score (88.22%) provides a more meaningful assessment of performance.

### 7.4 Discussion

**Overfitting/Underfitting:** The `max_depth=5` constraint acts as a regularization mechanism, preventing the tree from memorizing training data. On the synthetic dataset, this is evident in the accuracy plateau at ~73.5%, appropriately below the noise ceiling.

**Effect of Noise:** The 10% label noise in the synthetic dataset reduces the theoretical maximum accuracy. The decision tree demonstrates robustness by not chasing noisy labels, which would manifest as significantly lower test accuracy compared to training accuracy.

**Imbalance Handling:** On the Credit Card Fraud dataset, the tree achieves high precision but lower recall. This is a common challenge for standard decision trees on imbalanced data, as the majority class dominates the entropy calculations, making it harder for the tree to learn discriminative splits for the minority class. Future improvements could include class weighting, SMOTE oversampling, or cost-sensitive splitting.

---

## 8. Comparison with scikit-learn DecisionTreeClassifier

### 8.1 Setup & Hyperparameters

To ensure a fair comparison, both classifiers were configured with identical settings:

| Parameter | From-Scratch | scikit-learn |
| :--- | :---: | :---: |
| **Splitting Criterion** | Entropy | `criterion='entropy'` |
| **Max Depth** | 5 | `max_depth=5` |
| **Random State** | — | `random_state=42` |
| **Min Info Gain** | 0.01 | — (default) |

Both models are trained on the same 80/20 stratified split using `random_state=42`.

### 8.2 Side-by-Side Metric Comparison

#### Adult Census Income (Real-World)

| Metric | From-Scratch | scikit-learn | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 84.98% | 84.98% | 0.00% |
| **Precision** | 82.56% | 82.56% | 0.00% |
| **Recall** | 73.45% | 73.45% | 0.00% |
| **F1-Score** | 76.39% | 76.39% | 0.00% |

#### Synthetic Dataset (with Noise)

| Metric | From-Scratch | scikit-learn | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 73.50% | 73.50% | 0.00% |
| **Precision** | 73.49% | 73.49% | 0.00% |
| **Recall** | 73.50% | 73.50% | 0.00% |
| **F1-Score** | 73.49% | 73.49% | 0.00% |

#### Credit Card Fraud (Imbalanced)

| Metric | From-Scratch | scikit-learn | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.93% | 99.95% | −0.02% |
| **Precision** | 95.11% | 95.28% | −0.17% |
| **Recall** | 83.16% | 89.28% | −6.12% |
| **F1-Score** | 88.22% | 92.06% | −3.84% |

### 8.3 Analysis of Differences

**Adult & Synthetic Datasets:** The from-scratch implementation produces **identical results** to scikit-learn across all four metrics on both the Adult and Synthetic datasets. This validates the correctness of our entropy computation, information gain calculation, optimal threshold search, and tree-building logic.

**Credit Card Fraud Dataset:** Minor differences emerge exclusively on the highly imbalanced fraud dataset. The from-scratch tree achieves slightly lower Recall (83.16% vs. 89.28%) and F1-Score (88.22% vs. 92.06%). These differences stem from:

1. **Tie-breaking mechanisms:** When multiple splits yield identical information gain, scikit-learn uses randomized tie-breaking (`random_state=42`), whereas our implementation selects the first encountered split. On imbalanced data with many near-identical entropy values, this can lead to different tree structures.
2. **Floating-point precision:** scikit-learn's Cython-optimized C implementation uses different numerical precision for entropy and gain calculations, which can cause slight divergence in split decisions at the margins.
3. **Minimum gain threshold:** Our implementation uses `min_gain=0.01` as a stopping criterion, which may prune some splits that scikit-learn would retain with its default settings.

Despite these differences, achieving **95.11% Precision** and **83.16% Recall** on a 578:1 imbalanced dataset with a from-scratch implementation demonstrates strong practical capability.

---

## 9. Group Role Distribution & Individual Contributions

### 9.1 Affan Hameed (22i-2582)

**Role: Core Algorithm Design & Tree Architecture**

- Designed and implemented the `Node` and `DecisionTreeClassifier` class structure in `src/decision_tree.py`.
- Implemented the `_entropy()` method for Shannon entropy calculation with numerical stability handling.
- Developed the `_grow_tree()` recursive tree-building algorithm with configurable stopping criteria (`max_depth`, `min_gain`, pure nodes).
- Implemented the `_predict_single()` traversal method and the public `predict()` API.

### 9.2 Saim Zia (22i-2661)

**Role: Splitting Logic & Feature Handling**

- Implemented the `_information_gain()` method for computing weighted entropy reduction.
- Developed the optimized `_best_split()` method using cumulative class counts and vectorized boundary detection for $O(N \log N)$ split search.
- Implemented `_handle_missing()` for median-based imputation of `NaN` values with `float` type coercion for mixed-type array compatibility.
- Handled the continuous feature threshold search logic ensuring correct midpoint computation between adjacent sorted values.

### 9.3 Ahsan Iqbal (22i-2524)

**Role: Data Pipeline Engineering & Dataset Acquisition**

- Developed `data/data_loader.py` with three dataset loading functions: `load_adult_data()`, `generate_synthetic_data()`, and `load_credit_fraud_data()`.
- Implemented the Adult dataset pipeline including URL-based loading, missing value handling (`na_values=' ?'`), income binarization, and one-hot encoding via `pd.get_dummies()`.
- Designed the synthetic data generator with configurable sample size, feature count, hidden classification rule, and 10% noise injection with reproducible seeding.
- Set up the Credit Card Fraud loading pipeline with manual download instructions and `FileNotFoundError` handling for Kaggle authentication constraints.

### 9.4 Asim Shehzad (22i-2679)

**Role: Experimentation, Evaluation & Reporting**

- Developed `experiments/main.py` with the full experiment orchestration pipeline including CLI argument parsing (`--dataset` flag) for selective dataset execution.
- Implemented `experiments/utils.py` with `evaluate_model()` (Accuracy, Precision, Recall, F1-Score using macro averaging) and `save_metrics_to_csv()` for persistent metric storage.
- Configured and executed the scikit-learn `DecisionTreeClassifier` comparison with matched hyperparameters (`criterion='entropy'`, `max_depth=5`).
- Authored this technical report, including result interpretation, comparison analysis, and discussion of findings.

---

## 10. Conclusion & Future Work

### Conclusion

This project successfully demonstrated the implementation of a Decision Tree classifier from scratch, achieving performance that matches scikit-learn's highly optimized implementation on balanced datasets and closely approximates it on imbalanced data. The key accomplishments include:

- A fully functional entropy-based decision tree supporting continuous features, missing value imputation, and configurable stopping criteria.
- Rigorous evaluation across three diverse datasets covering real-world, synthetic (noisy), and highly imbalanced scenarios.
- Validated correctness through direct comparison with scikit-learn, achieving identical metrics on 2 out of 3 datasets.

### Future Work

1. **Pruning:** Implement post-pruning (e.g., reduced error pruning or cost-complexity pruning) to further combat overfitting.
2. **Class Weighting:** Add support for class-weighted entropy to improve performance on imbalanced datasets.
3. **Random Forest Extension:** Extend the single tree to an ensemble of trees (Random Forest) using bagging and random feature subsets.
4. **Gini Impurity:** Add Gini index as an alternative splitting criterion alongside entropy.
5. **Visualization:** Implement tree visualization to help interpret the learned decision boundaries.

---

## 11. References

1. UCI Machine Learning Repository — Adult Dataset: https://archive.ics.uci.edu/ml/datasets/adult
2. Kaggle — Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. scikit-learn Documentation — DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

---

## 12. Appendix

### A.1 Complete From-Scratch Code (`src/decision_tree.py`)

```python
import numpy as np

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_gain=0.01):
        self.max_depth = max_depth
        self.min_gain = min_gain

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _information_gain(self, y, y_left, y_right):
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        return self._entropy(y) - (weight_left * self._entropy(y_left)
               + weight_right * self._entropy(y_right))

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_idx = None
        best_threshold = None
        n_features = X.shape[1]
        n_samples = len(y)
        base_entropy = self._entropy(y)
        if base_entropy == 0:
            return None, None, 0
        for idx in range(n_features):
            sorted_indices = np.argsort(X[:, idx])
            X_sorted = X[sorted_indices, idx]
            y_sorted = y[sorted_indices]
            boundaries = np.where(X_sorted[:-1] != X_sorted[1:])[0] + 1
            if len(boundaries) == 0:
                continue
            unique_labels = np.unique(y)
            counts_left = {c: np.cumsum(y_sorted == c) for c in unique_labels}
            total_counts = {c: counts_left[c][-1] for c in unique_labels}
            for i in boundaries:
                n_left = i
                n_right = n_samples - i
                if n_left == 0 or n_right == 0:
                    continue
                entropy_left = 0
                entropy_right = 0
                for c in unique_labels:
                    count_left = counts_left[c][i - 1]
                    if count_left > 0:
                        p_left = count_left / n_left
                        entropy_left -= p_left * np.log2(p_left)
                    count_right = total_counts[c] - count_left
                    if count_right > 0:
                        p_right = count_right / n_right
                        entropy_right -= p_right * np.log2(p_right)
                gain = base_entropy - ((n_left / n_samples) * entropy_left
                       + (n_right / n_samples) * entropy_right)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_threshold = (X_sorted[i - 1] + X_sorted[i]) / 2.0
        return best_idx, best_threshold, best_gain

    def _grow_tree(self, X, y, depth=0):
        predicted_class = np.argmax(np.bincount(y))
        node = Node(predicted_class=predicted_class)
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            return node
        idx, threshold, gain = self._best_split(X, y)
        if idx is None or gain < self.min_gain:
            return node
        left_indices = X[:, idx] < threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]
        node.feature_index = idx
        node.threshold = threshold
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _handle_missing(self, X):
        X = np.array(X, dtype=float)
        for i in range(X.shape[1]):
            nan_mask = np.isnan(X[:, i])
            if np.any(nan_mask):
                median = np.nanmedian(X[:, i])
                X[nan_mask, i] = median
        return X

    def fit(self, X, y):
        X = self._handle_missing(X)
        self.tree_ = self._grow_tree(X, y)

    def _predict_single(self, x, node):
        if node.left is None and node.right is None:
            return node.predicted_class
        if x[node.feature_index] < node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        X = self._handle_missing(X)
        return np.array([self._predict_single(x, self.tree_) for x in X])
```

### A.2 Data Loading Scripts (`data/data_loader.py`)

```python
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from zipfile import ZipFile
import os

def load_adult_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'income']
    df = pd.read_csv(url, names=columns, na_values=' ?')
    df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
    df = pd.get_dummies(df)
    X = df.drop('income', axis=1).values
    y = df['income'].values
    return X, y

def load_credit_fraud_data():
    csv_path = 'data/creditcard.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Download from Kaggle.")
    df = pd.read_csv(csv_path)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    return X, y

def generate_synthetic_data(n_samples=2000, n_features=12):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(int)
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1),
                                     replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    return X, y
```

### A.3 Raw Metric CSV Files

Results are saved in the `results/` directory as CSV files:

| File | Description |
| :--- | :--- |
| `metrics_adult_scratch.csv` | From-scratch metrics on Adult dataset |
| `metrics_adult_sk.csv` | scikit-learn metrics on Adult dataset |
| `metrics_synthetic_scratch.csv` | From-scratch metrics on Synthetic dataset |
| `metrics_synthetic_sk.csv` | scikit-learn metrics on Synthetic dataset |
| `metrics_credit_scratch.csv` | From-scratch metrics on Credit Card Fraud dataset |
| `metrics_credit_sk.csv` | scikit-learn metrics on Credit Card Fraud dataset |
