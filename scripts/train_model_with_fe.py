"""
This script trains a linear regression model on preprocessed data and registers the model in the MLflow model registry.

Key Functionality:
- Imports necessary libraries for Spark, MLflow, and sklearn.
- Loads the project configuration from a YAML file.
- Reads preprocessed training and testing datasets from the Databricks catalog.
- Creates and updates the `calories_burned_features` feature table with necessary columns.
- Constructs a training set using Databricks Feature Engineering.
- Trains a linear regression model to predict calories burned.
- Evaluates the model using mean squared error (MSE), mean absolute error (MAE), and R-squared (R²) metrics.
- Logs model metrics, parameters, and artifacts to MLflow.
- Registers the trained model in the Databricks MLflow model registry.

Workflow:
1. Load raw or preprocessed datasets from Databricks catalog tables.
2. Define the `calories_burned_features` feature table and populate it with feature data from training and testing sets.
3. Create a training set using Databricks Feature Engineering with feature lookups.
4. Train the linear regression model on the training set.
5. Evaluate the model on the testing set and log performance metrics.
6. Log the model to MLflow, including its signature, and register it in the MLflow registry.
"""

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops_end_to_end_project.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

config = ProjectConfig.from_yaml("../project_config.yml")

num_features = config.num_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

feature_table_name = f"{catalog_name}.{schema_name}.calories_burned_features"

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.calories_burned_features
(Id STRING NOT NULL,
Age INT,
Weight DOUBLE,
Height DOUBLE,
Fat_Percentage DOUBLE,
BMI DOUBLE
)
""")

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.calories_burned_features "
    "ADD CONSTRAINT calories_burned_features_pk PRIMARY KEY (Id)"
)

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.calories_burned_features "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.calories_burned_features "
    f"SELECT Id, Age, Weight, Height, Fat_Percentage, BMI FROM {catalog_name}.{schema_name}.train_set"
)
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.calories_burned_features "
    f"SELECT Id, Age, Weight, Height, Fat_Percentage, BMI FROM {catalog_name}.{schema_name}.test_set"
)

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(
    "Age", "Weight", "Height", "Fat_Percentage", "BMI"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["Age", "Weight", "Height", "Fat_Percentage", "BMI"],
            lookup_key="Id",
        )
    ],
    exclude_columns=["update_timestamp_utc"],
)

training_df = training_set.load_df().toPandas()

X_train = training_df[num_features]
y_train = training_df[target]
X_test = test_set[num_features]
y_test = test_set[target]

mlflow.set_experiment(experiment_name="/Shared/calories_burned_basic")
mlflow.set_experiment_tags({"repository_name": "calories_burned_model"})

with mlflow.start_run(
    tags={"branch": "week2"},
) as run:
    run_id = run.info.run_id

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    signature = infer_signature(model_input=X_train, model_output=y_pred)

    fe.log_model(
        model=model,
        flavor=mlflow.sklearn,
        artifact_path="calories-burned-lr-model-fe",
        training_set=training_set,
        signature=signature,
    )

    mlflow.register_model(
        model_uri=f"runs:/{run_id}/calories-burned-lr-model-fe",
        name=f"{catalog_name}.{schema_name}.calories-burned-lr-model-fe",
    )
