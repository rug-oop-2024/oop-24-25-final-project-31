from abc import abstractmethod
import numpy as np
from copy import deepcopy
from typing import Literal

from autoop.core.ml.artifact import Artifact


class Model:
    def __init__(
            self,
            type: Literal["classification", "regression"],
            model: 'Model',
            *args,
            **kwargs
            ):
        """
        Initializes a Model instance.

        Parameters
        ----------
        type : Literal["classification", "regression"]
            The type of model, specifying whether it is a classification
            or regression model.
        model : Model
            The model class to be instantiated.
        *args :
            Positional arguments to be passed to the model instantiation.
        **kwargs :
            Keyword arguments to be passed to the model instantiation.
        """
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
        if value not in ("classification", "regression"):
            raise ValueError(f"{value} is not a valid model type.")
        self._type = value

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

    @parameters.setter
    def parameters(self, params: dict) -> None:
        """
        Sets the model's parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the model's parameters, including both
            strict parameters and hyperparameters.

        Returns
        -------
        None
        """
        self._parameters = params

    def to_artifact(self, name: str) -> Artifact:
        """
        Converts the model to an artifact.

        The artifact is created by serializing the model's parameters
        into a bytes object and saving it in the artifact's data
        attribute.

        Parameters
        ----------
        name : str
            The name of the artifact.

        Returns
        -------
        Artifact
            The created artifact.
        """
        return Artifact(name=name, data=self.parameters)

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
