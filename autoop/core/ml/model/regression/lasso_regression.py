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
        """
        Fits the Lasso Regression model to the provided data.

        Args:
            X (np.ndarray): 2D array of input features.
            y (np.ndarray): 1D array of target variable.

        Returns:
            None
        """
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable based on the input features.

        Args:
            X (np.ndarray): Features (2D array) with multiple predictors.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)
