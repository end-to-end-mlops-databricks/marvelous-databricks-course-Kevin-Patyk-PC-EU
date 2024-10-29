# Databricks notebook source
import json

import mlflow

# COMMAND ----------
# This sets the tracking URI to Databricks,
# indicating that MLflow should log experiments in the Databricks environment.
mlflow.set_tracking_uri("databricks")

# This method sets the current experiment to the specified name.
# If the experiment does not exist, it will be created.
mlflow.set_experiment(experiment_name="/Shared/mlops_end_to_end")

# This assigns tags to the experiment,
# providing metadata that can be used to filter or identify experiments later.
mlflow.set_experiment_tags({"repository_name": "mlops_end_to_end"})


# COMMAND ----------
# This function retrieves experiments based on a filter.
experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='mlops_end_to_end'"
)
print(experiments)

# COMMAND ----------

# This block saves the first experiment's details into a JSON file.
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)

# COMMAND ----------

# This begins a new run within the current experiment.
# The run_name, tags, and description provide context for the run.
with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": "ffa63b430205ff7", "branch": "week2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})


# COMMAND ----------
# This block retrieves the run ID of the most recent run with the specified Git SHA tag.
run_id = mlflow.search_runs(
    experiment_names=["/Shared/mlops_end_to_end"],
    filter_string="tags.git_sha='ffa63b430205ff7'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------
# Similar to the earlier JSON dump, this saves the retrieved run information into a file.
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------
print(run_info["data"]["metrics"])

# COMMAND ----------
print(run_info["data"]["params"])
