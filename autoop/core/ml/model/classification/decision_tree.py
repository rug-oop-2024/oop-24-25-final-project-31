from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.model.model import Model
import numpy as np


class DTreeClassifier(Model):
    """
    Wrapper of the DecisionTreeClassifier model from scikit-learn
    """
    def __init__(self, *args, **kwargs):
        """
        Initialises a DecisionTreeClassifier model.

        Parameters
        ----------
        *args
            Positional arguments passed to DecisionTreeClassifier
        **kwargs
            Keyword arguments passed to DecisionTreeClassifier
        """
        super().__init__("classification",
                         DecisionTreeClassifier,
                         *args, **kwargs)

    def __str__(self) -> str:
        """Return a string representation of the object.

        Returns:
            str: A string representation of the object.
        """
        return "Decision Tree Classifier Model"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the Decision Tree model to the provided data.

        Parameters
        ----------
        X : np.ndarray
            An array of input features
            where each row represents an observation.
        Y : np.ndarray
            An array of target values
            corresponding to each observation in X.
        """
        self.model.fit(X, Y)
        self.parameters = self._model.get_params()

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Predicts the target values based on the input features
        using the trained Decision Tree model.

        Parameters
        ----------
        input : np.ndarray
            An array of input features
            where each row represents an observation.

        Returns
        -------
        np.ndarray
            An array of predicted values
            corresponding to each observation in the input.
        """
        return self._model.predict(input)
