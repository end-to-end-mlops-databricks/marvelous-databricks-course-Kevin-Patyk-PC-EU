from typing import Any, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class CaloriesBurnedModel:
    """
    A class to handle training, prediction, and evaluation of a Linear Regression model
    for predicting calories burned.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the CaloriesBurnedModel.

        Args:
            config (Any): Configuration settings for the model.
        """
        self.config = config
        self.model = LinearRegression()

    def train(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> None:
        """
        Train the Linear Regression model.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Target values corresponding to the training features.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Features for which to make predictions.

        Returns:
            np.ndarray: Predicted target values.
        """
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the model's performance on the test dataset.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test features.
            y_test (pd.Series or np.ndarray): True target values corresponding to the test features.

        Returns:
            Tuple[float, float]: A tuple containing:
                - mse (float): Mean squared error of the predictions.
                - r2 (float): R-squared score of the predictions.
        """
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2
