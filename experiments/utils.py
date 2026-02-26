import numpy as np

np.random.seed(42)  # For reproducibility
n_samples = 2000
n_features = 12
X = np.random.randn(n_samples, n_features)  # Normally distributed features
# Hidden rule: class 1 if sum of first 5 features > 0, else 0
y = (np.sum(X[:, :5], axis=1) > 0).astype(int)
# Add noise: flip 10% of labels
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
y[noise_indices] = 1 - y[noise_indices]
# Save to CSV if needed: np.savetxt('synthetic_dataset.csv', np.c_[X, y], delimiter=',')# Experiment utility functions placeholder
