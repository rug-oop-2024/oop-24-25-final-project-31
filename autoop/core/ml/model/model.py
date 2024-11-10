from abc import abstractmethod
# from autoop.core.database import Database
# from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model:
    def __init__(
            self,
            type: Literal["classification", "regression"],
            model: 'Model',
            *args,
            **kwargs
            ):
        self.type = type
        self.model = model(*args, **kwargs)
        self._parameters: dict = {}

    @property
    def type(self) -> str:
        """
        Returns the type of the model.

        Returns
        -------
        str
            The type of the model. It is either "classification"
            or "regression".
        """
        return self._type

    @type.setter
    def type(self, value: Literal["classification", "regression"]):
        """
        Sets the type of the model.

        Parameters
        ----------
        value : Literal["classification", "regression"]
            The type of the model. It is either "classification"
            or "regression".

        Raises
        ------
        ValueError
            If the value is not a valid model type.

        Returns
        -------
        None
        """
        if isinstance(type, Literal["classification", "regression"]):
            self._type = value
        else:
            raise ValueError(f"{value} is not a valid model type.")

    @property
    def parameters(self):
        """
        Returns a deep copy of the model's parameters.

        Returns
        -------
        dict
            A dictionary containing the model's parameters, including both
            strict parameters and hyperparameters.
        """
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, observations: np.ndarray, target: np.ndarray) -> None:
        """
        Fits the model to the given observations and target values.

        Parameters
        ----------
        observations : np.ndarray
            The observations to fit the model to. It should be an array, where
            each row is an observation and each column is a feature.
        target : np.ndarray
            The target values to fit the model to. It should be an array.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the input data.

        Parameters
        ----------
        input : np.ndarray
            The input data to make predictions on. It should be an array, where
            each row is an observation and each column is a feature.

        Returns
        -------
        np.ndarray
            The predictions made by the model. It should be an array, where
            each row is the prediction of the corresponding observation in the
            input array.
        """
        pass
