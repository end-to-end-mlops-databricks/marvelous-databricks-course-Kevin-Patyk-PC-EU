from typing import List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    categorical_features: Optional[List[str]] = None
    num_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Optional[dict] = None
    ab_test: Optional[dict] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "ProjectConfig":
        """
        Load the project configuration from a `.yaml` file.

        Args:
            file_path (str): Path to the `.yaml` file.

        Returns:
            ProjectConfig: Instance of `ProjectConfig`.
        """
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)
