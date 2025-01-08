import pytest
from pydantic import ValidationError

from mlops_end_to_end_project.config import ProjectConfig


def test_project_config_basic():
    config = ProjectConfig(
        num_features=["feature1", "feature2"], target="target_column", catalog_name="catalog", schema_name="schema"
    )

    assert config.num_features == ["feature1", "feature2"]
    assert config.target == "target_column"
    assert config.catalog_name == "catalog"
    assert config.schema_name == "schema"
    assert config.categorical_features is None
    assert config.parameters is None
    assert config.ab_test is None


def test_project_config_missing_required_fields():
    with pytest.raises(ValidationError):
        ProjectConfig(target="target_column", catalog_name="catalog", schema_name="schema")
