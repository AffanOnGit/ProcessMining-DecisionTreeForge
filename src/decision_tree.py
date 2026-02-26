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
        self.min_gain = min_gain  # Threshold for minimal information gain (from lectures)

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Avoid log(0)

    def _information_gain(self, y, y_left, y_right):
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        return self._entropy(y) - (weight_left * self._entropy(y_left) + weight_right * self._entropy(y_right))

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_idx = None
        best_threshold = None
        n_features = X.shape[1]
        n_samples = len(y)
        
        # Pre-compute base entropy for the node
        base_entropy = self._entropy(y)

        # Optimize: don't test splits if all labels are the same
        if base_entropy == 0:
            return None, None, 0
            
        for idx in range(n_features):
            sorted_indices = np.argsort(X[:, idx])
            X_sorted = X[sorted_indices, idx]
            y_sorted = y[sorted_indices]

            # Fast way to find boundaries where values actually change
            boundaries = np.where(X_sorted[:-1] != X_sorted[1:])[0] + 1
            if len(boundaries) == 0:
                continue

            unique_labels = np.unique(y)
            
            # Count occurrences of each class moving left to right
            counts_left = {c: np.zeros(n_samples, dtype=int) for c in unique_labels}
            for c in unique_labels:
                counts_left[c] = np.cumsum(y_sorted == c)
                
            total_counts = {c: counts_left[c][-1] for c in unique_labels}

            for i in boundaries:
                # Number of items on each side
                n_left = i
                n_right = n_samples - i
                
                if n_left == 0 or n_right == 0:
                    continue

                # Calculate left and right entropy using the pre-computed counts
                entropy_left = 0
                entropy_right = 0
                
                for c in unique_labels:
                    # Left side probabilities
                    count_left = counts_left[c][i - 1]
                    if count_left > 0:
                        p_left = count_left / n_left
                        entropy_left -= p_left * np.log2(p_left)
                        
                    # Right side probabilities
                    count_right = total_counts[c] - count_left
                    if count_right > 0:
                        p_right = count_right / n_right
                        entropy_right -= p_right * np.log2(p_right)

                gain = base_entropy - ((n_left / n_samples) * entropy_left + (n_right / n_samples) * entropy_right)

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

# Example Usage (after loading dataset as X, y)
# dt = DecisionTreeClassifier(max_depth=5)
# dt.fit(X_train, y_train)
# y_pred = dt.predict(X_test)