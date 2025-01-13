# Databricks notebook source
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import mlflow
import torch

# COMMAND ----------

# Log and register the model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="phi_3_5_vision_instruct_model",
        registered_model_name="labuser8942947_1736773029.default.phi_3_5_vision_instruct_model",
        python_model=AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True),
        code_paths=[],
        pip_requirements=[
            "transformers==4.43.0",
            "torch==2.3.0",
            "accelerate==0.30.0",
            "Pillow==10.3.0",
            "requests==2.31.0",
            "numpy==1.24.4",
            "flash_attn>=2.5.8"
        ],
        signature=None)

# COMMAND ----------

# Define the function to generate text
def generate_text(prompt, images, model_name):
  # Load the model from the Unity Catalog
  model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")
  
  # Load the processor
  processor = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", 
    trust_remote_code=True, 
    num_crops=4
  )
  
  # Create the input prompt
  inputs = processor(prompt, images, return_tensors="pt")
  
  # Generate the text
  generation_args = { 
    "max_new_tokens": 1000, 
    "temperature": 0.0, 
    "do_sample": False, 
  } 
  
  # Make predictions using the model
  output = model.predict({"prompt": prompt, "images": images})
  
  # Return the output
  return output

# COMMAND ----------

# Test the function
images = []
placeholder = ""
for i in range(1,2):
    url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
    images.append(Image.open(requests.get(url, stream=True).raw))
    placeholder += f"<|image_{i}|>\n"

prompt = placeholder + "Summarize the deck of slides."
output = generate_text(prompt, images, "phi_3_5_vision_instruct_model")
print(output)