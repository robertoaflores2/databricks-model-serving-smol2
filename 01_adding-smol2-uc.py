# Databricks notebook source
# MAGIC %pip install transformers mlflow accelerate --quiet

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the model with device mapping for efficient inference
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True  # Required if the model uses custom code
)

# COMMAND ----------

import logging
import mlflow.pyfunc
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import mlflow.pyfunc

class SmolLM2Wrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_context(self, context):
        import torch
        import transformers
        import accelerate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Log the versions
        print("Transformers version:", transformers.__version__)
        print("Torch version:", torch.__version__)
        print("Accelerate version:", accelerate.__version__)

    def predict(self, context, model_input):
        # Extract the list of input texts
        texts = model_input['text'].tolist()
        
        # Prepare messages in the format expected by the model
        messages_list = [[{"role": "user", "content": text}] for text in texts]
        
        # Apply the chat template to each message
        input_texts = [self.tokenizer.apply_chat_template(messages, tokenize=False) for messages in messages_list]
        
        # Tokenize the input texts
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Increase if needed
        ).to(self.device)

        # Generate outputs
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,  # Adjust as needed
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        # Decode the outputs
        decoded_outputs = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # Post-process the outputs to extract the assistant's reply
        final_outputs = []
        for output in decoded_outputs:
            # Split the output into turns
            splits = output.split("<|assistant|>")
            if len(splits) > 1:
                assistant_reply = splits[-1].strip()
                final_outputs.append(assistant_reply)
            else:
                final_outputs.append(output.strip())
        
        return final_outputs


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS dbacademy_labuser7497544_1732645016_vocareum_com
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow

# Correct: Set the registry URI to 'databricks-uc' for Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Optionally, set the tracking URI to 'databricks' or leave it as default
mlflow.set_tracking_uri("databricks")  # Optional; you can omit this line if 'databricks' is the default

print("Current Tracking URI:", mlflow.get_tracking_uri())
print("Current Registry URI:", mlflow.get_registry_uri())


# Define the model name with Unity Catalog format
catalog_name = "dbacademy_labuser7497544_1732645016_vocareum_com"
schema_name = "default"
model_name = "SmolLM2"

full_model_name = f"{catalog_name}.{schema_name}.{model_name}"

# Define the model signature (if not already defined)
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec

input_schema = Schema([ColSpec("string", "text")])
output_schema = Schema([ColSpec("string")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log and register the model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="smollm2_model",
        registered_model_name=full_model_name,
        python_model=SmolLM2Wrapper(model, tokenizer),
        code_paths=[],
        pip_requirements=[
            "mlflow==2.18.0",
            "transformers==4.41.2",
            "torch>=2.1.0",
            "accelerate>=0.31.0" 
        ],
        signature=signature)
