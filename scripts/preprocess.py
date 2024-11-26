"""
This script is used to preprocess the raw data and save the preprocessed data to the catalog.

Key Functionality:
- Imports the necessary libraries.
- Loads the project configuration.
- Reads the raw data.
- Preprocesses the data.
- Splits the data into training and testing sets.
- Saves the training and testing sets to the catalog.

Workflow:
1. Loads the raw dataset from the Volumes directory.
2. Preprocesses the data using the DataProcessor class by handling numeric features and extracting relevant features.
3. Splits the data into training and testing sets using an 80-20 split.
4. Adds timestamp columns to the training and testing sets.
5. Saves the training and testing sets to the catalog.
"""

from pyspark.sql import SparkSession

from mlops_end_to_end_project.config import ProjectConfig
from mlops_end_to_end_project.data_processor import DataProcessor

spark = SparkSession.builder.getOrCreate()

config = ProjectConfig.from_yaml(config_path="project_config.yml")

df = spark.read.csv(
    "/Volumes/dev_neu/adhoc/files/mlops_end_to_end/gym_members_exercise_tracking.csv", header=True, inferSchema=True
).toPandas()

data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess()

train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
