import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from mlops_end_to_end_project.config import ProjectConfig


class DataProcessor:
    """
    Class to preprocess the data.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig) -> None:
        """
        Initialize the DataProcessor.

        Args:
            pandas_df (pd.DataFrame): Input data.
            config (ProjectConfig): Configuration file.
        """
        self.df = pandas_df  # store the DataFrame as `self.df`
        self.config = config  # store the configuration as `self.config`

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the data, including converting columns to numeric
        and selecting relevant columns.

        Returns:
            pd.DataFrame: Preprocessed data.
        """

        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        target = self.config.target
        relevant_columns = num_features + [target] + ["Id"]
        self.df = self.df[relevant_columns]
        self.df["Id"] = self.df["Id"].astype(str)

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets.

        Args:
            test_size (float): Fraction of the data to be used as the test set.
            random_state (int): Random seed.

        Returns:
            pd.DataFrame: Training data.
            pd.DataFrame: Testing data.
        """

        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)

        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession) -> None:
        """
        Save the train and test sets into Databricks tables.
        """

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
