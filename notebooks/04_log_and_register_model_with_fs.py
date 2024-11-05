# Databricks notebook source
# import libraries
import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops_end_to_end.config import ProjectConfig

# initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = (
    WorkspaceClient()
)  # this is the client that interacts with the Databricks workspace from the SDK package

fe = (
    feature_engineering.FeatureEngineeringClient()
)  # this is the client that interacts with the feature engineering service


# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri(
    "databricks"
)  # It must be -uc for registering models to Unity Catalog


# using an absolute path since the relative path isn't working and I would like to move on from this
config = ProjectConfig.from_yaml(
    config_path="/Workspace/Users/k.patyk@paulaschoice-eu.com/.bundle/marvelous-databricks-course-Kevin-Patyk-PC-EU/dev/files/project_config.yml"
)

# extract configuration details
num_features = config.num_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.gym_features"
function_name = f"{catalog_name}.{schema_name}.calculate_calories_burned"

# COMMAND ----------

# load training and testing sets from Databricks tables
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------

# Create or replace the gym_features table
# this creates an empty table called `gym_features` with the specified schema
# meaning, it establishes the table structure in the catalog, but does not insert any data into it
spark.sql(
    f"""
    CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.gym_features
    (
        Id STRING NOT NULL,
        Age INT,
        Weight FLOAT,
        Height FLOAT
    );
    """
)

# add primary key constraint
# we are altering the `gym_features` table to add a primary key constraint on the `Id` column
# this is a best practice to ensure that the `Id` column is unique and not null
# `gym_pk` is the name of the contraint being defined. It's a user-defined name for the primary key constraint that identifies it uniquely in the database
# `PRIMARY KEY(Id)` specifies that the `Id` column is the primary key for the gym_features table
spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.gym_features "
    "ADD CONSTRAINT gym_pk PRIMARY KEY(Id);"
)

# we are altering the `gym_features` table to set a specific table property for a Delta table
# `TBLPROPERTIES` clause allows you to define or modify the properties of the associated table
# `delta.enableChangeDataFeed = true` enables change data feed for the `gym_features` table
# change data feed allows you to track changes, such as inserts, updates, and deletes, to the table
# change data feed also maintains a history of the changes, enabling you to query the changes that have occurred over time
# with change data feed enabled, you can perform incremental processing in downstream applications
# lastly, it can simply the integration of delta tables with other systems that need to be aware of data changes
spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.gym_features "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

# Insert data into the feature table from both train and test sets
# this executes 2 SQL statements to insert data into the `gym_features` table
# the overall purpose of these statements is to populate the `gym_features` table with data from the `train_set` and `test_set` tables
spark.sql(
    f"""
    INSERT INTO {catalog_name}.{schema_name}.gym_features
    SELECT Id,
           Age,
           Weight,
           Height
    FROM {catalog_name}.{schema_name}.train_set
    """
)

spark.sql(
    f"""
    INSERT INTO {catalog_name}.{schema_name}.gym_features
    SELECT Id,
           Age,
           Weight,
           Height
    FROM {catalog_name}.{schema_name}.test_set
    """
)

# COMMAND ----------

# Define a custom function to calculate calories burned per minute (session duration/calories burned)
# same as UDF in pyspark, but you have to define it in SQL
# this function takes input, in this case Session_Duration and Calories_Burned and returns a float
# this might not work, since it can only output integers and strings, so we will force the output to be an integer
spark.sql(
    f"""
CREATE OR REPLACE FUNCTION {function_name}(session_duration DOUBLE, workout_frequency INT)
RETURNS INT
LANGUAGE PYTHON AS
$$
return int(round(session_duration / workout_frequency))
$$
"""
)

# COMMAND ----------

# loading training and test sets

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(
    "Age", "Weight", "Height"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

train_set = train_set.withColumn("Id", train_set["Id"].cast("string"))

# Use the custom function we made earlier to make a new column in the training set
training_set = fe.create_training_set(
    df=train_set,  # specifies the dataframe to use as the training data, which was loaded in the previous step
    label=target,  # indicates the target variable for prediction, defined in the project configuration file
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,  # name of the table in the unity catalog, in this case: `gym_features`
            feature_names=[
                "Age",
                "Weight",
                "Height",
            ],  # uses the numeric features names
            lookup_key="Id",  # identifies each record by the `Id` column as the lookup key
        ),
        FeatureFunction(
            udf_name=function_name,  # specifies which function to use, which was defined earlier
            output_name="Workouts_Per_Session",  # this sets name of the new column to be added to the training set
            input_bindings={  # this maps input columns to the parameters of the function
                "session_duration": "Session_Duration",
                "workout_frequency": "Workout_Frequency",
            },
        ),
    ],
    exclude_columns=[
        "update_timestamp_utc"
    ],  # specifies any columns to exclude from the training set
)

# load feature-engineered dataframe
training_df = training_set.load_df().toPandas()

# calculate workouts per session for test set
test_set["Workouts_Per_Session"] = (
    test_set["Session_Duration"] / test_set["Workout_Frequency"]
)

# split features and target
X_train = training_df[num_features + ["Workouts_Per_Session"]]
y_train = training_df[target]
X_test = test_set[num_features + ["Workouts_Per_Session"]]
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

    # log model with feature engineering
    fe.log_model(
        model=model,
        flavor=mlflow.sklearn,
        artifact_path="linear-regression-model-fe",
        training_set=training_set,
        signature=signature,
    )

    # register the model
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/linear-regression-model-fe",
        name=f"{catalog_name}.{schema_name}.linear-regression-model-fe",
    )
