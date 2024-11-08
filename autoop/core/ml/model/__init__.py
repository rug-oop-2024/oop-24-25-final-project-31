
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.regression import LassoRegression


REGRESSION_MODELS = [
    "multiple_linear_regression",
    "lasso_regression",
    "ridge_regression"
]  # add your models as str here

CLASSIFICATION_MODELS = [
]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    models = {
        "multiple_linear_regression": MultipleLinearRegression(),
        "lasso_regression": LassoRegression(),
        "ridge_regression": RidgeRegression()
    }
    return models.get(model_name, None)
