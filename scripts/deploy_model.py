"""
This script manages the serving endpoint for the calories-burned linear regression model using Databricks Serving.

Key Features:
- Dynamically fetches catalog and schema names from a project configuration file (`project_config.yml`).
- Ensures the Databricks serving endpoint named `calories-burned-model-serving` is created or updated as needed.
- Configures the endpoint to serve a specific version of the `calories-burned-lr-model-fe` model.
- Enables scale-to-zero for cost optimization and assigns the workload size to `Small`.

Workflow:
1. Initialize the Databricks `WorkspaceClient` and `SparkSession`.
2. Load project configuration to retrieve catalog and schema details.
3. Check if the serving endpoint `calories-burned-model-serving` exists:
   - If it exists, update its configuration with the specified model details.
   - If it does not exist, create a new serving endpoint with the given configuration.
4. Configure the endpoint to:
   - Serve the `calories-burned-lr-model-fe` model at the specified version.
   - Optimize costs by enabling scale-to-zero functionality.
   - Use a `Small` workload size for serving.
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from pyspark.sql import SparkSession

from mlops_end_to_end_project.config import ProjectConfig

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

config = ProjectConfig.from_yaml(config_path="project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name
model_version = 1

endpoint_name = "calories-burned-model-serving"
served_entity_config = ServedEntityInput(
    entity_name=f"{catalog_name}.{schema_name}.calories-burned-lr-model-fe",
    scale_to_zero_enabled=True,
    workload_size="Small",
    entity_version=model_version,
)

endpoints = workspace.serving_endpoints.list()
existing_endpoint = next((ep for ep in endpoints if ep.name == endpoint_name), None)

if existing_endpoint:
    workspace.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        served_entities=[served_entity_config],
    )
else:
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[served_entity_config],
        ),
    )
