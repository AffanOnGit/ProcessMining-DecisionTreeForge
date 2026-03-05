# Viva Preparation Guide: Decision Tree Implementation

This document organizes the core concepts, full code implementations, evaluations, and potential viva questions based on the exact work division of the group. **Code snippets are included with their exact locations to help you trace and explain the logic during your demo.**

---

## 1. Affan Hameed (Core Algorithm Design & Tree Architecture)

### 1.1 Entropy Calculation
**Concept:** 
Entropy measures the impurity or disorder within a dataset. In binary classification, a pure node (e.g., all fraudulent transactions) has an entropy of `0`. An evenly mixed node (e.g., 50% fraud, 50% genuine) has a maximum entropy of `1`. The goal of the tree is to find splits that minimize this entropy.

**Handwritten Calculation:**
$$ H(S) = - \sum_{c} p_c \cdot \log_2(p_c) $$

**Code Implementation & Rationale:**
*Location:* `src/decision_tree.py` (Inside `DecisionTreeClassifier._entropy`)

```python
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # Add 1e-10 to probabilities to avoid log2(0) mathematical errors
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
```
* **Why `np.unique`?** Instead of hardcoding for just 2 classes, this allows the tree to handle multiclass datasets seamlessly by calculating the count of every unique class present.
* **Why the `1e-10`?** Without this, if a class has 0 elements inside a node, $log_2(0)$ evaluates to `-infinity` and crashes the program in Python. The small epsilon guarantees mathematical stability.

**Potential Viva Questions:**
*   **Q: How did you implement the Decision Tree core logic without using `sklearn`?**
    *   *A: I designed a custom `Node` class that acts as a binary tree (containing `.left`, `.right`, `.feature_index`, and `.threshold`). The `DecisionTreeClassifier` logic traverses and builds this tree recursively via NumPy array slicing.*

### 1.2 Movement to Left or Right (Growing the Tree)
**Concept:**
When a node splits, we physically segment our dataset into two smaller arrays: a Left child (`feature < threshold`) and a Right child (`feature >= threshold`). We then recursively repeat the split on both children.

**Code Implementation & Rationale:**
*Location:* `src/decision_tree.py` (Inside `DecisionTreeClassifier._grow_tree`)

```python
    def _grow_tree(self, X, y, depth=0):
        predicted_class = np.argmax(np.bincount(y))
        node = Node(predicted_class=predicted_class)
        
        # Stopping Criteria
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            return node
            
        ... # (Information Gain calculated here, see Saim's section)
        
        # Array Sub-setting using Boolean Masks
        left_indices = X[:, idx] < threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]
        
        node.feature_index = idx
        node.threshold = threshold
        
        # Recursion
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
```
* **Why `X[left_indices]`?** This is "Boolean Masking" in NumPy. It is the fastest way to extract the subsets of data without writing slower, nested `for` loops in native Python.
* **Why use recursion (`self._grow_tree()`)?** A Decision tree is inherently a recursive data structure. Recursion allows the code to brilliantly cascade down branches until one of the strict stopping criteria is met (like `depth == max_depth` or pure nodes).

### 1.3 Inference (Predicting New Data)
**Concept:**
To classify a new sample, we start at the root node. We compare the sample's feature against the root's threshold, moving left or right until we hit a terminal "leaf" node, where the prediction is made.

**Code Implementation & Rationale:**
*Location:* `src/decision_tree.py` (Inside `DecisionTreeClassifier._predict_single`)

```python
    def _predict_single(self, x, node):
        # Base Case: It's a leaf node because it has no children
        if node.left is None and node.right is None:
            return node.predicted_class
            
        # Decision logic mapping to the tree structure
        if x[node.feature_index] < node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)
```

---

## 2. Saim Zia (Splitting Logic & Feature Handling)

### 2.1 Information Gain
**Concept:**
Information Gain (IG) calculates how much structural "uncertainty" we removed by splitting our node. The algorithm always greedily selects the feature+threshold pair that produces the *highest* Information Gain.
$$ IG = H(Parent) - \left( \frac{N_{left}}{N_{total}} H(Left) + \frac{N_{right}}{N_{total}} H(Right) \right) $$

### 2.2 Optimizing the Node Choice (Finding the Best Split)
**Concept:**
In handwritten calculations, you check every single midpoint threshold value sequentially. In the code, evaluating thousands of rows this way computationally freezes the program ($O(N^2 \times D)$). We had to hyper-optimize it.

**Code Implementation & Rationale:**
*Location:* `src/decision_tree.py` (Inside `DecisionTreeClassifier._best_split`)

```python
        for idx in range(n_features):
            sorted_indices = np.argsort(X[:, idx])
            X_sorted = X[sorted_indices, idx]
            y_sorted = y[sorted_indices]

            # Fast way to find boundaries where consecutive values differ
            boundaries = np.where(X_sorted[:-1] != X_sorted[1:])[0] + 1
            
            # Cumulative frequency tracing
            counts_left = {c: np.cumsum(y_sorted == c) for c in unique_labels}
            
            for i in boundaries:
                # Dynamic Threshold Calculation
                best_threshold = (X_sorted[i - 1] + X_sorted[i]) / 2.0
```
* **Why `np.argsort` and `np.where`?** Sorting the array lets us find the "Boundary Midpoints" (where a class changes from 0 to 1) instead of wildly testing overlapping continuous values. 
* **Why cumulative counts (`np.cumsum`)?** Instead of re-calculating the entropy completely from scratch on the left and right arrays for *every single threshold*, we just keep a running tally of how many labels we've seen so far moving left to right. This brings execution speed down to $O(N \log N \times D)$!

**Potential Viva Questions:**
*   **Q: How does your implementation handle continuous features efficiently based on the assignment requirements?**
    *   *A: The code sorts the array feature by feature, immediately isolates boundaries via `np.where(X_sorted[:-1] != X_sorted[1:])`, and calculates optimal threshold candidates exactly at the mathematical midpoints of these transitions. This fulfills the "optimal threshold search" requirement natively inside vectorized NumPy.*

### 2.3 Handling Missing Values
**Code Implementation & Rationale:**
*Location:* `src/decision_tree.py` (Inside `DecisionTreeClassifier._handle_missing`)

```python
    def _handle_missing(self, X):
        X = np.array(X, dtype=float)    # Coerce everything to float
        for i in range(X.shape[1]):
            nan_mask = np.isnan(X[:, i])
            # If NaNs exist, replace them exclusively with the column Median
            if np.any(nan_mask):
                median = np.nanmedian(X[:, i])
                X[nan_mask, i] = median
        return X
```
* **Why `dtype=float`?** Categorical arrays, like the one-hot encoded strings from the Adult dataset, contain booleans. The mathematical functions like `np.nanmedian` and `np.isnan` require arrays typed as floats.
* **Why use Median Imputation over Mean Imputation?** The Adult datasets include heavily skewed features (like `capital-gain`). The *Median* is statistically robust against these extreme outliers, providing much safer generated data replacements for `?` rows than the Mean.

---

## 3. Ahsan Iqbal (Data Pipeline Engineering & Dataset Acquisition)

### 3.1 Datasets Preprocessing
**Implementation Details & Rationale:**
*Location:* `data/data_loader.py`

*   **Adult Census Income (Real-World Benchmark):**
    *   `na_values=' ?'`: Replaced the dataset's unique character format for missing values into explicit `NaN`s so the tree could process them.
    *   `pd.get_dummies()`: Machine learning models cannot natively split on raw text strings (like "Married" vs "Single"). We One-Hot encoded them, converting those columns into multiple binary (0/1) columns, inflating dimension features safely to 108.
*   **Synthetic Dataset (with Noise):**
    *   *Reasoning for Noise:* The assignment required generating 2,000 array instances and artificially forcing 10% of them to flip their answer labels. This explicitly evaluates if the tree "overfits." If the strict logic rigidly memorizes these wrong answers, it proves our tree is failing at generalized inductive learning.
*   **Credit Card Fraud (Extremely Imbalanced):**
    *   *Reasoning:* Contains 284k legitimate rows versus only 492 fraud rows (~578:1 unbalance). Used straight from Kaggle. Required for testing the algorithm under immense asymmetric entropy conditions.

**Potential Viva Questions:**
*   **Q: How did you ensure data splitting preserved class proportions during your `train_test_split`?**
    *   *A: We utilize "Stratified Sampling". Because Fraud represents only 0.17% of the dataset, a standard random split might place zero fraud cases cleanly into the Test set. Stratification actively guaranteed the exact minority ratios transitioned uniformly into both Train and Test environments.*

---

## 4. Asim Shehzad (Experimentation, Evaluation & Reporting)

### 4.1 Evaluation Metrics Structure
*Location:* `experiments/utils.py`

To legitimately evaluate our classifier against Scikit-Learn without bias, `utils.py` computes metrics utilizing macro-averaging logic:
*   **Accuracy:** Evaluates overall correctness. Highly misleading on imbalanced configurations (predicting *all* transactions as "Genuine" automatically acquires 99.8% Accuracy, making it look excellent despite instantly failing all Fraud cases).
*   **Precision (Macro):** "Out of all points flagged positively, how many were genuine positives?" Minimizes *False Positives*.
*   **Recall (Macro):** "Out of all true actual positives, how many did the model capture?" High recall is the gold standard for Fraud; missing an anomaly (False Negative) is substantially costlier than triggering a false alarm.
*   **F1-Score (Macro):** The Harmonic mean pairing Precision and Recall. Delivering equally scaled weighting to both majority and minority classes.

### 4.2 Explanation of Scikit-Learn Results Comparison
**Analysis Context & Rationale:**
*   **Adult Dataset (Real-World Benchmark):** Our `DecisionTreeClassifier` achieved **84.98% Accuracy, exactly mirroring `sklearn` (0.00% difference).** The variance between Precision (82%) and Recall (73%) clearly reflects the moderate income inequality skew and our strict `max_depth=5` restriction functioning as intended.
*   **Synthetic Dataset (10% Noise):** Reached **~73.5% Accuracy across metrics matching Scikit-Learn perfectly (0.00% diff).** The mathematical proof of "robustness" holds here: because we injected a flat 10% error rate, the general accuracy strictly plateaued safely under 90%. Thus demonstrating the tree *did not* aggressively overfit to memorize noisy values.
*   **Credit Card Fraud Dataset (Imbalanced):** True performance rested at **95.11% Precision** and **83.16% Recall.**

**Potential Viva Questions:**
*   **Q: Why was there a slight deviation from Scikit-learn (lower recall) uniquely on the Imbalanced Fraud dataset but not on the others?**
    *   *A: Scikit-learn algorithms encode randomized tie-breaking on identically valued entropy splits, and leverage deeper floating-point precision down inside their compiled Cython core. Moreover, our manual tree respects the strict `min_gain` variable stop condition natively, aggressively pruning tiny informational gains trailing inside massively skewed minority subsets.*
*   **Q: Based on your evaluation, what is the best metric for the fraud dataset?**
    *   *A: Recall and Macro F1-Score. Accuracy is virtually useless because predicting 0 for everything guarantees 99.8% correctness. We want to maximize Recall strictly to isolate anomalous behaviors.*

---

## 5. Codebase Architecture & Structure Overview

This section maps out where everything lives in the project folder so you can quickly navigate the code during the viva demo.

### Directory Layout
```text
ProcessMining-DecisionTreeForge/
│
├── src/                # Core Algorithm Logic
│   └── decision_tree.py ➔ Contains the custom `Node` and `DecisionTreeClassifier` classes.
│
├── data/               # Data Acquisition & Preprocessing
│   └── data_loader.py   ➔ Handlers for downloading Adult Census, generating Synthetic noise, & reading Fraud CSV.
│
├── experiments/        # Execution & Evaluation
│   ├── main.py          ➔ The main entry point to run the model (e.g., `python experiments/main.py --dataset adult`).
│   └── utils.py         ➔ Stores `evaluate_model()` (calculating Accuracy/Precision/Recall/F1) & saving CSVs.
│
├── notebooks/          # Exploratory Analysis (Optional)
│   └── exploratory_analysis.ipynb ➔ Jupyter notebooks used for initial dataset investigation.
│
├── results/            # Outputs
│   └── *.csv            ➔ Auto-generated metric comparison charts between our manual tree and scikit-learn.
│
└── report/             # Documentation
    ├── technical_report.md  ➔ The highly detailed academic breakdown.
    └── viva_preparation_guide.md ➔ This current guide designed for demo Q&A.
```

---

## 6. End-to-End Data Flow (How the Solution Works)

If you are asked to explain exactly what happens from the moment you run the program until it spits out a prediction, explain this flow:

### Step 1: Loading & Preprocessing (`data_loader.py`)
1.  **Acquisition:** Based on the user argument (`--dataset`), the program activates a loader function. It either downloads the Adult dataset from the UCI URL, generates arrays using `np.random.randn` for synthetic data, or reads the local `creditcard.csv`.
2.  **Structuring:** The target column (like `income` or `Class`) is isolated into vector `y`. All remaining feature columns are grouped into matrix `X`.
3.  **Refining:** Categorical strings are dynamically pushed into `pd.get_dummies()` to transform them into ML-readable 0/1 binary columns.

### Step 2: Training & Splitting Setup (`experiments/main.py`)
1.  **Stratified Split:** Data is sent into `train_test_split(X, y)`. The program specifically uses `stratify=y` to ensure that if 1% of the data is fraudulent, exactly 1% of the Training set and 1% of the Testing set are fraudulent.
2.  **Instantiation:** Two models are created locally: our custom `DecisionTreeClassifier(max_depth=5)` and `sklearn.tree.DecisionTreeClassifier(max_depth=5)`.

### Step 3: Imputation & Tree Growth (`src/decision_tree.py`)
*(This is what happens when `$ model.fit(X_train, y_train)` is called)*
1.  **Imputation:** The code instantly scans `X_train` looking for `np.nan`. If found, it computes the mathematical median of that entire column and injects it into the empty spots.
2.  **Recursive Growth:** The program enters `_grow_tree()`.
3.  **Finding Information Gain:** It loops over all features. For each feature, it sorts the array, finds everywhere the class jumps from 0 to 1 (`np.where`), finds the midpoints there, and measures the Entropy of breaking the dataset at that midpoint.
4.  **The Split:** The tree locates the single best split globally and permanently records that feature index and threshold into the `Node` object.
5.  **Branching:** The data array forces rows < threshold to a Left variable, and rows >= threshold to a Right variable. `_grow_tree()` acts recursively, completely starting over Step 3 on both the new Left bucket and new Right bucket.
6.  **Halting:** This loops infinitely creating Left/Right branches until a bucket is either 100% pure (e.g. all 1s), hits the `max_depth` restriction, or gets too small.

### Step 4: Inference & Evaluation (`experiments/utils.py`)
*(This is what happens when `$ predictions = model.predict(X_test)` is called)*
1.  **Deployment:** For every single row in `X_test`, the program drops the sample into the root of the freshly built tree.
2.  **Descent:** The sample checks its own value against the node's threshold. E.g., "Is my index `[4]` less than 12.5? Yes." -> Move Left. "Is my index `[9]` less than 1.0? No." -> Move right.
3.  **Result:** It hits a terminal Leaf node. It adopts the majority class calculated by `np.argmax(np.bincount(y))` at training time. This is appended to the `predictions` array.
4.  **Grading:** `utils.py` compares the `predictions` array against the true `y_test` hidden answers, calculating the formulas for True Positives, False Positives (Precision), and False Negatives (Recall).
5.  **Outputting:** Both the manual model and Scikit-learn model grades are printed side-by-side to the terminal and stored permanently to `/results/`.
