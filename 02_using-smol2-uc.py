# Databricks notebook source
# MAGIC %pip install mlflow transformers torch accelerate

# COMMAND ----------

import mlflow

# Set the MLflow registry URI to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Optionally, set the tracking URI to 'databricks' or leave it as default
mlflow.set_tracking_uri("databricks")


# COMMAND ----------

# Define the model name with Unity Catalog format
catalog_name = "dbacademy_labuser7497544_1732645016_vocareum_com"
schema_name = "default"
model_name = "SmolLM2"

full_model_name = f"{catalog_name}.{schema_name}.{model_name}"

# Specify the model version or stage
model_version = "latest"  # You can also specify a specific version number or stage like 'Production'

# Construct the model URI
model_uri = f"models:/{full_model_name}/{model_version}"

# COMMAND ----------

import mlflow.pyfunc
model_version = "6"  # Replace with your model's version number
model_uri = f"models:/{full_model_name}/{model_version}"

loaded_model = mlflow.pyfunc.load_model(model_uri)


# COMMAND ----------

import pandas as pd

# Example input text
input_text = "what is the most popular anime from the 90s?"

# Create a DataFrame with a column named 'text' (as per the model signature)
input_data = pd.DataFrame({'text': [input_text]})

# COMMAND ----------

# Get predictions
predictions = loaded_model.predict(input_data)

# Output the result
print(predictions[0])