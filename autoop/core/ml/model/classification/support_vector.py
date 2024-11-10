from sklearn.svm import SVC
from autoop.core.ml.model.model import Model
import numpy as np


class SupportVectorClassifier(Model):
    """
    Wrapper of SVC model from scikit-learn
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes a Support Vector Classifier model.

        Args:
            *args: Variable length argument list for model initialization.
            **kwargs: Arbitrary keyword arguments for model initialization.
        """
        super().__init__("classification",
                         SVC,
                         *args, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the model to the data.

        Args:
            X (np.ndarray): An array of Features with multiple predictors.
            Y (np.ndarray): Target variable.
        """
        self.model.fit(X, Y)
        self._parameters = self._model.get_params()

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable based on the input features.

        Args:
            input (np.ndarray): An array of Features with multiple predictors.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(input)
