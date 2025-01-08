import pandas as pd
import pytest

from mlops_end_to_end_project.data_processor import DataProcessor


@pytest.fixture
def sample_data():
    """Fixture for creating sample pandas DataFrame."""
    data = {
        "Id": [1, 2, 3],
        "feature1": ["10", "20", "invalid"],
        "feature2": ["30.5", "40.1", "50.2"],
        "target": [1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def config():
    """Fixture for creating a sample ProjectConfig."""

    class MockConfig:
        num_features = ["feature1", "feature2"]
        target = "target"

    return MockConfig()


def test_data_processor_initialization(sample_data, config):
    """Test the initialization of the DataProcessor."""
    processor = DataProcessor(sample_data, config)
    assert processor.df.equals(sample_data)
    assert processor.config == config


def test_preprocess(sample_data, config):
    """Test the preprocess method."""
    processor = DataProcessor(sample_data, config)
    processor.preprocess()

    # Assert numeric conversion
    assert pd.api.types.is_numeric_dtype(processor.df["feature1"])
    assert pd.api.types.is_numeric_dtype(processor.df["feature2"])

    # Assert 'Id' is converted to string
    assert pd.api.types.is_string_dtype(processor.df["Id"])

    # Assert correct columns are selected
    expected_columns = config.num_features + [config.target, "Id"]
    assert list(processor.df.columns) == expected_columns


def test_split_data(sample_data, config):
    """Test the split_data method."""
    processor = DataProcessor(sample_data, config)
    processor.preprocess()  # Preprocess before splitting
    train_set, test_set = processor.split_data(test_size=0.33, random_state=42)

    # Assert split sizes
    assert len(train_set) + len(test_set) == len(processor.df)

    # Tolerance-based check for test size
    expected_test_size = round(len(processor.df) * 0.33)
    assert len(test_set) == expected_test_size
