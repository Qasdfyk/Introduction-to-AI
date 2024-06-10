import numpy as np

class ID3DecisionTree:
    class Node:
        def __init__(self, feature=None, branches=None, value=None):
            self.feature = feature
            self.branches = branches
            self.value = value

    def __init__(self, X, y, max_depth=5):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.tree = self.fit()

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def information_gain(self, X, y, feature):
        values, counts = np.unique(X[:, feature], return_counts=True)
        weighted_entropy = sum((counts[i] / len(y)) * self.entropy(y[X[:, feature] == values[i]]) for i in range(len(values)))
        return self.entropy(y) - weighted_entropy

    def find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        for feature in range(X.shape[1]):
            gain = self.information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature
    
    def get_parameters(self):
        return {"X":self.X, 'y':self.y, 'max_depth':self.max_depth}

    def fit(self, X=None, y=None, depth=0):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if depth == self.max_depth or len(np.unique(y)) == 1:
            if len(y) == 0:
                return self.Node(value=None)
            else:
                return self.Node(value=np.bincount(y).argmax())
        feature = self.find_best_split(X, y)
        if feature is None:
            return self.Node(value=np.bincount(y).argmax())
        values, counts = np.unique(X[:, feature], return_counts=True)
        branches = {}
        for i in range(len(values)):
            subset_indices = X[:, feature] == values[i]
            if len(y[subset_indices]) == 0:
                branches[values[i]] = self.Node(value=None)
            else:
                branches[values[i]] = self.fit(X[subset_indices], y[subset_indices], depth+1)
        return self.Node(feature=feature, branches=branches)

    def predict(self, x):
        return self._predict(self.tree, x)

    def _predict(self, tree, x):
        if tree.value is not None:
            return tree.value
        branch = tree.branches.get(x[tree.feature])
        if branch is None:
            return np.bincount(self.y).argmax() 
        return self._predict(branch, x)
