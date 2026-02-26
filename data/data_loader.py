import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from zipfile import ZipFile
import os


# real world dataset
def load_adult_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(url, names=columns, na_values=' ?')
    df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
    df = pd.get_dummies(df)
    X = df.drop('income', axis=1).values
    y = df['income'].values
    return X, y


# heavily imbalanced dataset
def load_credit_fraud_data():
    url = 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'
    csv_path = 'data/creditcard.csv'
    if not os.path.exists(csv_path):
        print("\n[!] Please download the Credit Card Fraud dataset manually.")
        print(f"[!] 1. Go to: {url}")
        print("[!] 2. Download the archive.")
        print("[!] 3. Extract the contents and place 'creditcard.csv' in the 'data/' directory.\n")
        raise FileNotFoundError(f"Missing {csv_path}. Kaggle datasets cannot be downloaded without authentication.")
    df = pd.read_csv(csv_path)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    return X, y


# synthetic dataset
def generate_synthetic_data(n_samples=2000, n_features=12):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(int)
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    return X, y