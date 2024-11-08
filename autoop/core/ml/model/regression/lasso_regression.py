from sklearn.linear_model import Lasso


class LassoRegression:
    """Wrapper for Lasso Regression using scikit-learn"""

    def __init__(self, alpha=1.0):
        self.model = Lasso(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
