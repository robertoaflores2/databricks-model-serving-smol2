# Databricks notebook source
# MAGIC %pip install transformers --quiet

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
import pandas as pd

# COMMAND ----------

class SmolLM2Model(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, context, model_input):
        # Ensure the input is a DataFrame with a 'prompt' column
        if 'prompt' not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'prompt' column.")
        
        # Tokenize the input prompts
        inputs = self.tokenizer(
            model_input['prompt'].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate responses
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )

        # Decode the responses
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Return the responses as a DataFrame
        return pd.DataFrame({'generated_text': responses})


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