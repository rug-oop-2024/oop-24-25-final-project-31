import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNearestClassifier(Model):
    """
    Wrapper of KNeighboursClassifier model from scikit-learn
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize a KNearestClassifier model.

        Parameters
        ----------
        *args :
            Arguments passed to the KNeighborsClassifier constructor.
        **kwargs :
            Keyword arguments passed to the KNeighborsClassifier constructor.

        Notes
        -----
        The KNearestClassifier parameters are set using the `get_params`
        method of the KNeighborsClassifier instance after fitting the model.
        """
        super().__init__("classification",
                         KNeighborsClassifier,
                         *args, **kwargs)

    def __str__(self) -> str:
        """Return a string representation of the object.

        Returns:
            str: A string representation of the object.
        """
        return "K Nearest Neighbours Classifier Model"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the KNearestClassifier model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        Y : np.ndarray
            Target variable.

        Returns
        -------
        None

        Notes
        -----
        The KNearestClassifier parameters are set using the `get_params`
        method of the KNeighborsClassifier instance after fitting the model.
        """
        self.model.fit(X, Y)
        self.parameters = self._model.get_params()

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the input data.

        Parameters
        ----------
        input : np.ndarray
            Input features for which predictions are to be made.

        Returns
        -------
        np.ndarray
            Predicted class labels for each input sample.
        """
        return self._model.predict(input)
