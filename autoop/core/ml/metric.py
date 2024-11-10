from abc import ABC, abstractmethod
from typing import Any
import numpy as np


METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "accuracy",
    "weighted_precision",
    "weighted_recall"

]  # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> Any:
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    metrics = {
        "mean_squared_error": MeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "root_mean_squared_error": RootMeanSquaredError(),
        "accuracy": Accuracy(),
        "weighted_precision": WeightedPrecision(),
        "weighted_recall": WeightedRecall()
    }
    return metrics.get(name, None)


class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input
    # and return a real number
    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric given ground truth and predictions.
        Args:
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Predicted values
        Returns:
            float: Calculated metric as real number
        """
        pass


# add here concrete implementations of the Metric class
class MeanAbsoluteError(Metric):
    def __init__(self):
        super().__init__(
            name="mean_absolute_error",
            description="Measures the\
            average absolute difference"
            )

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))


class MeanSquaredError(Metric):
    def __init__(self):
        super().__init__(
            name="mean_squared_error",
            description="Measures the average squared difference"
            )

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(Metric):
    def __init__(self):
        super().__init__(
            name="root_mean_squared_error",
            description="Measures the square root\
            of the average squared difference"
            )

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse


class Accuracy(Metric):
    def __init__(self):
        super().__init__(
            name="accuracy",
            description="measures the proportion of\
            correctly determined features"
            )

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.setdiff1d(y_pred.flatten(),
                            y_true.flatten()).size/y_pred.size


def precision_recall(
        type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
        ) -> float:
    classes = np.unique(y_true)
    total_weight = y_true.size
    result = 0
    weighted_result = 0

    for cls in classes:
        true_positives = np.sum((y_pred == cls) and (y_true == cls))
        if type == "recall":
            false_pos_neg = np.sum((y_pred != cls) and (y_true == cls))
        elif type == "precision":
            false_pos_neg = np.sum((y_pred == cls) and (y_true != cls))
        try:
            result = true_positives/(true_positives+false_pos_neg)
        except ZeroDivisionError:
            result = 0
        weight = np.sum(y_true == cls)

        weighted_result += (result * weight) / total_weight
    return weighted_result


class WeightedPrecision(Metric):
    def __init__(self):
        super().__init__(
            name="weighted_precision",
            description="calculates the weighted\
            true positive proportion"
            )

    def __call__(self, **kwargs) -> float:
        return precision_recall("precision", **kwargs)


class WeightedRecall(Metric):
    def __init__(self):
        super().__init__(
            name="weighted_recall",
            description="calculates the weighted\
            true positive rate"
            )

    def __call__(self, **kwargs) -> float:
        return precision_recall("recall", **kwargs)
