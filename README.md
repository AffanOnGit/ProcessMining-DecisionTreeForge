# Process Mining - Decision Tree Forge

This repository contains the implementation of a Decision Tree classifier created completely from scratch, fulfilling the requirements for the **Process Mining and Simulation** course assignment. 

The core goal of this project is to implement an interpretable decision tree using standard Python libraries (like `numpy` and `pandas`) without relying on pre-built machine learning libraries structurally, and then benchmark its performance against `scikit-learn`'s industrial standard implementation.

## 🚀 Key Features
- **Built from Scratch:** The core logic inside `src/decision_tree.py` is written purely in NumPy.
- **Support for Entropy & Information Gain:** Calculates optimal splits natively.
- **Continuous Features:** Optimized $O(N \log N)$ splitting boundaries for non-categorical values.
- **Missing Values:** Robust inline median imputation handles `NaN` inputs safely.
- **Scikit-Learn Benchmarking:** Generates direct comparative metrics against `sklearn` for Accuracy, Precision, Recall, and F1-score.

---

## 📂 Project Structure

```text
ProcessMining-DecisionTreeForge/
│
├── data/
│   ├── data_loader.py        # Pipelines for downloading/processing Adult, Synthetic, and Kaggle data
│   ├── generate_synthetic.py # Generator for synthetic datasets with noise
│   └── (CSV files are downloaded/extracted here)
│
├── experiments/
│   ├── main.py               # Main experiment runner (comparing Scratch vs Sklearn models)
│   └── utils.py              # Metric evaluation logic (accuracy, precision, recall, f1)
│
├── report/
│   └── technical_report.md   # Detailed performance and technical analysis of the algorithms
│
├── results/                  # Generated CSV output files containing benchmarking metrics
│
├── src/
│   ├── decision_tree.py      # The CORE implementation of the custom Decision Tree
│   └── pattern_mining/       # Brute Force & Apriori pattern mining on Adult dataset
│       ├── adult_loader.py   # Loads Adult Census as transactions (column=value)
│       ├── brute_force.py    # Brute-force frequent itemset mining
│       └── apriori.py        # Apriori frequent itemset mining
│
├── requirements.txt          # Required Python packages
└── README.md                 # This file
```

---

## 🛠️ Installation & Setup

Before running the project, you need to set up a Python virtual environment and install the required dependencies:

### 1. Clone the repository
```bash
git clone https://github.com/AffanOnGit/ProcessMining-DecisionTreeForge.git
cd ProcessMining-DecisionTreeForge
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to isolate dependencies inside a virtual environment.

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

The assignment benchmarks the decision tree against three specific datasets. **You must perform a manual step for the Credit Card Fraud dataset.**

1. **Adult Income Dataset (Real-world):** Auto-downloads dynamically using `pandas`. No action required.
2. **Synthetic Dataset (Noisy):** Programmatically generated inside the code. No action required.
3. **Credit Card Fraud (Highly Imbalanced):** 
   - `data_loader.py` is configured to look for this dataset, but downloading from Kaggle programmatically requires authentication tokens.
   - **Manual Action Required:**
     1. Go to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
     2. Download the ZIP archive.
     3. Extract the `creditcard.csv` file into the `data/` folder of this project.

---

## ▶️ How to Run the Experiments

The primary entry point for the project is `experiments/main.py`. This script trains both the **Scratch** and **Sklearn** Decision Trees on the specified dataset(s), evaluates them, prints the results, and saves the metrics as CSV files in the `results/` folder.

> **Note for Windows Users:** If you activated the environment but PowerShell is stubbornly using your global python installation (throwing `ModuleNotFoundError`), always use the explicit path `.\.venv\Scripts\python.exe` instead of just `python` or `python3`.

### Option 1: Run on ALL Datasets
*Ensure `creditcard.csv` is correctly placed in `data/` before running this, or it will throw an error.*
```powershell
.\.venv\Scripts\python.exe .\experiments\main.py --dataset all
```

### Option 2: Run on Specific Datasets
If you want to test the algorithms sequentially or avoid datasets you haven't downloaded:

**Run only Adult Income:**
```powershell
.\.venv\Scripts\python.exe .\experiments\main.py --dataset adult
```

**Run only the Synthetic generator:**
```powershell
.\.venv\Scripts\python.exe .\experiments\main.py --dataset synthetic
```

**Run only Credit Card Fraud:**
```powershell
.\.venv\Scripts\python.exe .\experiments\main.py --dataset credit
```

### Viewing Results
After execution, the terminal will print dictionaries summarizing the performance (Accuracy, Precision, Recall, F1-Score). You can also find persistent copies of this data written to `results/` (e.g., `metrics_adult_scratch.csv`, `metrics_adult_sk.csv`).

---

## 🧩 Pattern Mining on Adult Dataset (Brute Force & Apriori)

In addition to the decision tree classifier, this repository now includes two classic pattern-mining algorithms implemented from scratch and applied to the **Adult Census Income** dataset:

- **Brute Force frequent itemset mining** – exhaustively enumerates candidate itemsets and counts their support.
- **Apriori frequent itemset mining** – efficiently grows itemsets level-wise using the Apriori pruning principle.

Both algorithms operate on a transactional representation of the Adult dataset, where each row is converted into a set of interpretable items of the form `"column=value"` (e.g., `"education=Bachelors"`, `"sex= Male"`). The helper module `src/pattern_mining/adult_loader.py` fetches the Adult data from the original UCI URL and constructs these transactions.

### Loading Adult as Transactions

```python
from src.pattern_mining.adult_loader import load_adult_transactions

transactions = load_adult_transactions()  # optional row limit via `limit=` if needed
```

### Running Brute Force

```python
from src.pattern_mining.brute_force import brute_force_frequent_itemsets

transactions = load_adult_transactions()
frequent_bf = brute_force_frequent_itemsets(
  transactions,
  min_support=0.1,   # 10% relative support
  max_length=2,      # up to pairs of items
)

# Example: print top 10 patterns by support
for items, support in sorted(frequent_bf.items(), key=lambda x: -x[1])[:10]:
  print(set(items), support)
```

### Running Apriori

```python
from src.pattern_mining.apriori import apriori_frequent_itemsets

transactions = load_adult_transactions()
frequent_ap = apriori_frequent_itemsets(
  transactions,
  min_support=0.1,
  max_length=3,
)
```

Both functions return a dictionary mapping `frozenset({"item1", "item2", ...})` to its **relative support** in the dataset. These frequent itemsets can be interpreted as discovered patterns in the Adult Census data and can be further extended to association rules if needed.
