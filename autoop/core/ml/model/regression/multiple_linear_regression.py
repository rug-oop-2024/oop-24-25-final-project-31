from sklearn.linear_model import LinearRegression


class MultipleLinearRegression:
    """Facade for a scikit-learn Linear Regression model"""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Fits the model to the data.

        Args:
            X (np.ndarray): Features (2D array) with multiple predictors.
            y (np.ndarray): Target variable.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the target variable based on the input features.

        Args:
            X (np.ndarray): Features (2D array) with multiple predictors.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Evaluates the model using R^2 score by default.

        Args:
            X (np.ndarray): Features for evaluation.
            y (np.ndarray): True target values.

        Returns:
            float: R^2 score.
        """
        return self.model.score(X, y)
