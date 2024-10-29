# Databricks notebook source


from pyspark.sql import SparkSession

from mlops_end_to_end.config import ProjectConfig
from mlops_end_to_end.preprocessing import DataProcessor

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# using an absolute path since the relative path isn't working and I would like to move on from this
config = ProjectConfig.from_yaml(
    config_path="/kevin/Documents/Projects/marvelous-databricks-course-Kevin-Patyk-PC-EU/project_config.yml"
)

# COMMAND ----------
df = spark.read.csv(
    "/Volumes/prod_neu/adhoc/files/mlops_end_to_end/gym_members_exercise_tracking.csv",
    header=True,
    inferSchema=True,
).toPandas()

# COMMAND ----------
data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
