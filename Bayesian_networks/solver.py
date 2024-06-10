import numpy as np
from scipy.stats import norm

class Solver:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.stds = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)
            self.stds[c] = X_c.std(axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
    
    def predict(self, X):
        y_pred = [self._predict_instance(x) for x in X]
        return np.array(y_pred)
    
    def _predict_instance(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(norm.pdf(x, self.means[c], self.stds[c])))
            posterior = prior + conditional
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        pass
