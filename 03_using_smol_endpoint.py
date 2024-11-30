# Databricks notebook source
import requests
import json

# COMMAND ----------

# Replace with your actual Databricks personal access token
DATABRICKS_TOKEN = ""

# Replace with your actual model serving endpoint URL
SERVING_ENDPOINT_URL = "https://dbc-b9003989-5b33.cloud.databricks.com/serving-endpoints/SmolLM2/invocations"


# COMMAND ----------

# Set up the headers for authentication and content type
headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}


# COMMAND ----------

# For batch predictions
input_texts = [
    "Who is Lizzie McGuire?",
    "What is the capital of France?",
    "Tell me a joke."
]

payload = {
    "dataframe_records": [
        {"text": text} for text in input_texts
    ]
}

payload = {
  "dataframe_records": [
    {"text": "What was the most influential anime in 2000s?"}
  ]
}

# COMMAND ----------

# Send the POST request to the serving endpoint
response = requests.post(
    SERVING_ENDPOINT_URL,
    headers=headers,
    data=json.dumps(payload)
)


# COMMAND ----------

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON
    result = response.json()
    
    # Extract predictions
    predictions = result.get("predictions", [])
    
    # Print the predictions
    for idx, prediction in enumerate(predictions):
        print(f"Input: {payload['dataframe_records'][idx]['text']}")
        print(f"Output: {prediction}")
        print("-" * 50)
else:
    # Handle errors
    print(f"Request failed with status code {response.status_code}")
    print(f"Error message: {response.text}")
