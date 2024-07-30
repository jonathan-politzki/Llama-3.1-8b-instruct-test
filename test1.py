import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Specify the path to your local model files
model_path = "/Users/jonathanpolitzki/Desktop/Coding/Llama-3.1-8b-instruct-test/Meta-Llama-3.1-8B-Instruct"

# Load the tokenizer and model from local files
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Prepare the input
messages = [{"role": "user", "content": "How to make orange juice?"}]

# Generate the response
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)

# Print the response
print(outputs[0]['generated_text'])