import numpy as np
import pandas as pd
import pytest

from mlops_end_to_end.config import ProjectConfig
from src.mlops_end_to_end.preprocessing import DataProcessor


@pytest.fixture
def sample_data():
    """
    Returns a sample DataFrame for testing purposes
    """
    data = {
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "feature3": np.random.rand(100),
        "target": np.random.rand(100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """
    Returns a sample configuration for testing purposes
    """
    return ProjectConfig(
        num_features=["feature1", "feature2", "feature3"],
        target="target",
        catalog_name="test_catalog",
        schema_name="test_schema",
    )


@pytest.fixture
def data_processor(sample_data, sample_config):
    """
    Returns a DataProcessor instance for testing purposes
    """
    return DataProcessor(sample_data, sample_config)


def test_load_data(data_processor, sample_data):
    """
    Test the load_data method of the DataProcessor class
    """
    # Since the DataProcessor is initialized with sample_data, we can skip loading from a file.
    pd.testing.assert_frame_equal(data_processor.df, sample_data)


def test_preprocess(data_processor):
    """
    Test the preprocess method of the DataProcessor class
    """
    data_processor.preprocess()

    # After preprocessing, check if the DataFrame only contains the relevant columns
    assert set(data_processor.df.columns) == set(
        ["feature1", "feature2", "feature3", "target"]
    )
    assert data_processor.processor is not None


def test_split_data(data_processor):
    """
    Test the split_data method of the DataProcessor class
    """
    data_processor.preprocess()
    train_set, test_set = data_processor.split_data(test_size=0.2, random_state=42)

    assert train_set.shape[0] == 80  # 80% of 100
    assert test_set.shape[0] == 20  # 20% of 100


def test_preprocessor_transform(data_processor):
    """
    Test the transform method of the DataProcessor class
    """
    data_processor.preprocess()
    X_transformed = data_processor.processor.fit_transform(
        data_processor.df[data_processor.config.num_features]
    )

    assert X_transformed.shape == (
        100,
        3,
    )  # Check if the transformed shape matches the number of features
