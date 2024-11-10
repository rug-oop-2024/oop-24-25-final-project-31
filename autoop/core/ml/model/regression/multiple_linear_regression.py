from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    """Facade for a scikit-learn Linear Regression model"""

    def __init__(self, *args, **kwargs):
        super().__init__("regression",
                         LinearRegression,
                         *args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the data.

        Args:
            X (np.ndarray): Features (2D array) with multiple predictors.
            y (np.ndarray): Target variable.
        """
        self.model.fit(X, y)
        self._parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable based on the input features.

        Args:
            X (np.ndarray): Features (2D array) with multiple predictors.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)
