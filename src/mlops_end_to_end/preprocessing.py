import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_end_to_end.config import ProjectConfig


class DataProcessor:
    """
    A class to preprocess data for training a machine learning model
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df  # store the DataFrame as self.df
        self.config = config  # store the configuration

    def load_data(self, filepath):
        """
        Load data from a CSV file
        """
        return pd.read_csv(filepath)

    def preprocess(self):
        """
        Preprocess data for training a machine learning model
        """

        # handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # extract target and relevant features
        target = self.config.target
        relevant_columns = num_features + [target]
        self.df = self.df[relevant_columns]

        # create preprocessing steps for numerical features
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        # combine preprocessing steps
        self.processor = ColumnTransformer(
            transformers=[("num", numeric_transformer, self.config.num_features)]
        )

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        """
        train_set, test_set = train_test_split(
            self.df, test_size=test_size, random_state=random_state
        )
        return train_set, test_set

    def save_to_catalog(
        self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession
    ):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
