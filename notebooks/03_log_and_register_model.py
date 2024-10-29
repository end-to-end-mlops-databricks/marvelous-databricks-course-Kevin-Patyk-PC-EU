# Databricks notebook source

import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops_end_to_end.config import ProjectConfig

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri(
    "databricks-uc"
)  # It must be -uc for registering models to Unity Catalog

# COMMAND ----------

# using an absolute path since the relative path isn't working and I would like to move on from this
config = ProjectConfig.from_yaml(
    config_path="/kevin/Documents/Projects/marvelous-databricks-course-Kevin-Patyk-PC-EU/project_config.yml"
)

# extract configuration details
num_features = config.num_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set[num_features]
y_train = train_set[target]

X_test = test_set[num_features]
y_test = test_set[target]

# COMMAND ----------

# This is where we would use ColumnTransformer and Pipeline to preprocess the data
# preprocessor = ColumnTransformer(
#     transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
#     remainder="passthrough",
# )

# This is where we would use ColumnTransformer and Pipeline to preprocess the data
# pipeline = Pipeline(
#     steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**parameters))]
# )

# COMMAND ----------

# setting the mlflow experiment
mlflow.set_experiment(experiment_name="/Shared/mlops_end_to_end")

# setting a mock git_sha
git_sha = "ffa63b430205ff7"

# start an mlflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    run_id = run.info.run_id

    # train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LinearRegression")
    # mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # A model signature provides metadata that describes the inputs and outputs of a model, including:
    # input schema and output schema
    # This helps with type safety, interoperability, data validation, and documentation.
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # this approach leverages MLflow’s data tracking capabilities to improve model reproducibility
    # and traceability by associating the training data with the MLflow experiment.
    dataset = mlflow.data.from_spark(
        train_set_spark,
        table_name=f"{catalog_name}.{schema_name}.train_set",
        version="0",
    )
    mlflow.log_input(dataset, context="training")

    # logging a machine learning model to MLflow, which helps track and store models
    # along with their metadata for reproducibility, deployment, and versioning.
    mlflow.sklearn.log_model(
        sk_model=model,  # The scikit-learn model to be logged (could be a pipeline or a single model).
        artifact_path="linear_regression_model",  # The path under which the model will be saved in MLflow (used for retrieval).
        # registered_model_name="linear_regression",  # Registers the model under a specific name in the MLflow Model Registry for versioning and tracking.
        signature=signature,  # The input/output schema of the model, used to validate data formats when deploying.
        input_example=X_train.head(),  # A sample of input data that can be used to validate the model's structure and expected input format.
    )

# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/linear_regression_model",  # Specifies the location of the model to register, using the run ID to locate it.
    name=f"{catalog_name}.{schema_name}.calories_burned_model_v1",  # The name under which the model is registered in the MLflow Model Registry, typically including schema or version details.
    tags={
        "git_sha": f"{git_sha}"
    },  # Tags allow additional metadata to be attached, such as the Git SHA for traceability to a specific code version.
)

# COMMAND ----------
run = mlflow.get_run(
    run_id
)  # Retrieves the MLflow run object for the specified run_id, giving access to details about this specific run.

dataset_info = run.inputs.dataset_inputs[
    0
].dataset  # Accesses the first dataset input associated with this run (assuming multiple datasets could be logged) and gets metadata about the dataset.

dataset_source = mlflow.data.get_source(
    dataset_info
)  # Retrieves the source information for the dataset, which includes details about where and how the data was stored.

dataset_source.load()  # Loads the dataset directly from its source, enabling access to the original data used in this MLflow run.

# This allows you to programmatically retrieve and reload datasets that were part of a particular MLflow run, which is useful for reproducibility or auditing.
