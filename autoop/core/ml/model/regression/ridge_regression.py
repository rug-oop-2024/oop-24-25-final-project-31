from sklearn.linear_model import Ridge


class RidgeRegression:
    """Wrapper for Ridge Regression using scikit-learn"""

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
