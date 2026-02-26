import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.decision_tree import DecisionTreeClassifier
from data.data_loader import load_adult_data, generate_synthetic_data, load_credit_fraud_data
from sklearn.tree import DecisionTreeClassifier as SkDT
from sklearn.model_selection import train_test_split
from experiments.utils import evaluate_model, save_metrics_to_csv

def run_experiment(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Scratch DT
    dt_scratch = DecisionTreeClassifier(max_depth=5)
    dt_scratch.fit(X_train, y_train)
    y_pred_scratch = dt_scratch.predict(X_test)
    metrics_scratch = evaluate_model(y_test, y_pred_scratch)
    # Sklearn DT
    dt_sk = SkDT(criterion='entropy', max_depth=5, random_state=42)
    dt_sk.fit(X_train, y_train)
    y_pred_sk = dt_sk.predict(X_test)
    metrics_sk = evaluate_model(y_test, y_pred_sk)
    # Save
    save_metrics_to_csv(metrics_scratch, f'results/metrics_{dataset_name}_scratch.csv')
    save_metrics_to_csv(metrics_sk, f'results/metrics_{dataset_name}_sk.csv')
    print(f"{dataset_name} Scratch: {metrics_scratch}")
    print(f"{dataset_name} Sklearn: {metrics_sk}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'adult', 'synthetic', 'credit'])
    args = parser.parse_args()
    if args.dataset in ['all', 'adult']:
        X, y = load_adult_data()
        run_experiment(X, y, 'adult')
    if args.dataset in ['all', 'synthetic']:
        X, y = generate_synthetic_data()
        run_experiment(X, y, 'synthetic')
    if args.dataset in ['all', 'credit']:
        X, y = load_credit_fraud_data()
        run_experiment(X, y, 'credit')