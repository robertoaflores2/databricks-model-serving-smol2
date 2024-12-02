from huggingface_hub import InferenceClient

client = InferenceClient(api_key="")

messages = [
	{
		"role": "user",
		"content": "Who is Dolly Parton?"
	}
]

completion = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message)