import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlops_end_to_end_project.calories_burned_model import (
    CaloriesBurnedModel,
)


@pytest.fixture
def sample_data():
    """Fixture for sample training and test data."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.Series([10, 15, 20])
    X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [7, 8]})
    y_test = pd.Series([25, 30])
    return X_train, y_train, X_test, y_test


@pytest.fixture
def model_config():
    """Fixture for a mock configuration."""
    return {"mock_key": "mock_value"}


def test_model_initialization(model_config):
    """Test initialization of the CaloriesBurnedModel."""
    model = CaloriesBurnedModel(model_config)
    assert isinstance(model.model, LinearRegression)
    assert model.config == model_config


def test_train(sample_data, model_config):
    """Test the train method."""
    X_train, y_train, _, _ = sample_data
    model = CaloriesBurnedModel(model_config)
    model.train(X_train, y_train)
    assert model.model.coef_ is not None  # Ensure model is trained


def test_predict(sample_data, model_config):
    """Test the predict method."""
    X_train, y_train, X_test, _ = sample_data
    model = CaloriesBurnedModel(model_config)
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)


def test_evaluate(sample_data, model_config):
    """Test the evaluate method."""
    X_train, y_train, X_test, y_test = sample_data
    model = CaloriesBurnedModel(model_config)
    model.train(X_train, y_train)
    mse, r2 = model.evaluate(X_test, y_test)
    assert isinstance(mse, float)
    assert isinstance(r2, float)
