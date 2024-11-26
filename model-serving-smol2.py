# Databricks notebook source
# MAGIC %pip install transformers --quiet

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
import pandas as pd
import logging

# COMMAND ----------

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SmolLM2Model(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        logger.debug("Loading tokenizer and model.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def predict(self, context, model_input):
        logger.debug("Received input for prediction.")
        try:
            if not isinstance(model_input, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame.")
            if 'prompt' not in model_input.columns:
                raise ValueError("Input DataFrame must contain a 'prompt' column.")
            if model_input['prompt'].isnull().any():
                raise ValueError("Null values found in 'prompt' column.")

            inputs = self.tokenizer(
                model_input['prompt'].tolist(),
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            logger.info("Generated responses: %s", responses)

            return pd.DataFrame({'generated_text': responses})
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            raise

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS dbacademy_labuser7497544_1732645016_vocareum_com

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
import pandas as pd

# Define the input and output schema
input_schema = Schema([ColSpec("string", "prompt")])
output_schema = Schema([ColSpec("string", "generated_text")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Create an input example
input_example = pd.DataFrame({"prompt": ["What is the capital of France?"]})

# Set the registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Define the catalog and schema
catalog_name = "dbacademy_labuser7497544_1732645016_vocareum_com"
schema_name = "default"
model_name = f"{catalog_name}.{schema_name}.smollm2_model"

# Log the model with MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="smollm2_model",
        python_model=SmolLM2Model(),
        signature=signature,
        input_example=input_example,
        registered_model_name=model_name
    )

# COMMAND ----------

# Example usage
model = SmolLM2Model()
model.load_context(None)
input_df = pd.DataFrame({'prompt': ["What is the capital of France?"]})
output_df = model.predict(None, input_df)
print(output_df)