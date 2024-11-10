from sklearn.linear_model import Ridge
from autoop.core.ml.model.model import Model
import numpy as np


class RidgeRegression(Model):
    """Wrapper for Ridge Regression using scikit-learn"""

    def __init__(self, *args, **kwargs):
        super().__init__("regression",
                         Ridge,
                         *args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the Ridge Regression model to the provided data.

        Parameters
        ----------
        X : np.ndarray
            An array of input features
            where each row represents an observation.
        y : np.ndarray
            An array of target values
            corresponding to each observation in X.
        """

        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable based on the input features.

        Parameters
        ----------
        X : np.ndarray
            An array of input features where each row represents
            an observation.

        Returns
        -------
        np.ndarray
            An array of predicted values corresponding to each
            observation in X.
        """
        return self.model.predict(X)
