from sklearn.linear_model import Lasso
from autoop.core.ml.model.model import Model
import numpy as np


class LassoRegression(Model):
    """Wrapper for Lasso Regression using scikit-learn"""
    def __init__(self, *args, **kwargs):
        super().__init__("regression",
                         Lasso,
                         *args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self._parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
