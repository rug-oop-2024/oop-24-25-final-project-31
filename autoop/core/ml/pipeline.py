from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ):
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical"\
                and model.type != "classification":
            raise ValueError("Model type must be classification for \
                             categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for \
                             continuous target feature")

    def __str__(self) -> str:
        """Return a string representation of the pipeline.

        The string representation includes the model type, input features,
        target feature, split, and metrics.

        Returns:
            str: The string representation of the pipeline.
        """
        return f"""
            Pipeline(
                model={self._model.type},
                input_features={list(map(str, self._input_features))},
                target_feature={str(self._target_feature)},
                split={self._split},
                metrics={list(map(str, self._metrics))},
            )
            """

    @property
    def model(self) -> Model:
        """
        Returns the model used in the pipeline.

        Returns
        -------
        Model
            The model used in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the
        pipeline execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(
            name="pipeline_config",
            data=pickle.dumps(pipeline_data)
            ))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"
            ))
        return artifacts

    def _register_artifact(self, name: str, artifact) -> None:
        """
        Registers an artifact with the given name in the pipeline's
        artifacts dictionary.

        Args:
            name (str): The name to associate with the artifact.
            artifact: The artifact to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the features and registers the artifacts
        generated during the preprocessing process.
        The preprocessed data is stored in instance variables
        _input_vectors and _output_vector.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature],
            self._dataset
            )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
            ]

    def _split_data(self) -> None:
        # Split the data into training and testing sets
        """
        Splits the preprocessed data into training and testing sets
        according to the split ratio specified in the pipeline's
        configuration.

        The training set is the first 'split' proportion of the data,
        and the testing set is the remaining 1-split proportion.

        The training and testing sets are stored in the instance
        variables _train_X, _train_y, _test_X, and _test_y.
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))]
                         for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):]
                        for vector in self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
            ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
            ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Compact the input vectors into a single 2D numpy array.

        This method takes the list of input vectors and concatenates them
        along the columns (axis=1) into a single 2D numpy array.

        Parameters
        ----------
        vectors : List[np.array]
            List of input vectors to be compacted

        Returns
        -------
        np.array
            Compact 2D numpy array
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Train the model on the training set.

        This method compacts the training input vectors and fits the
        model.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluate the model on the test set and store the predictions
        and metrics results.

        This method compacts the test input vectors, makes predictions
        using the model,
        and evaluates the predictions against the true test outputs using
        the specified
        metrics. The results are stored in the _metrics_results and
        _predictions attributes.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_training(self) -> None:
        """
        Evaluate the model on the training set.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._training_metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._training_metrics_results.append((metric, result))
        self._train_predictions = predictions

    def execute(self) -> dict:
        """
        Execute the pipeline. This method will first preprocess the features,
        split the data, train the model, evaluate the model on the test set,
        and evaluate the model on the training set.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._evaluate_training()
        return {
            "test metrics": self._metrics_results,
            "training metrics": self._training_metrics_results,
            "predictions": self._predictions,
            "training predictions": self._train_predictions
        }
