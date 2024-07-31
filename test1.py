import transformers
import torch
import time
from mlx_lm import load, generate

# Specify the path to your local model files
model_path = "Meta-Llama-3.1-8B-Instruct"

# Code specific to M1 Mac
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def initialize_pipeline_with_progress():
    print("Starting pipeline initialization...")
    start_time = time.time()
    
    try:
        print("Creating pipeline (this may take a while)...")
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={
                "torch_dtype": torch.float32,  # Use float32 instead of bfloat16 for M1 Mac
                "load_in_8bit": True,  # Enable 8-bit quantization for efficiency
            },
            device_map=device,
        )
        print(f"Pipeline created successfully. Time taken: {time.time() - start_time:.2f} seconds")
        return pipeline
    except Exception as e:
        print(f"Error creating pipeline: {str(e)}")
        raise

def initialize_mlx_model():
    print("Loading MLX model...")
    start_time = time.time()
    try:
        model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
        print(f"MLX Model loaded. Time taken: {time.time() - start_time:.2f} seconds")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading MLX model: {str(e)}")
        raise

def generate_response_mlx(model, tokenizer, prompt):
    print("Generating response using MLX...")
    start_time = time.time()
    response = generate(model, tokenizer, prompt, max_tokens=256)
    print(f"MLX Response generated. Time taken: {time.time() - start_time:.2f} seconds")
    return response

# Try Transformers pipeline
try:
    pipeline = initialize_pipeline_with_progress()
    print("Pipeline initialization completed.")
    
    messages = [
        {"role": "user", "content": "How to make orange juice?"},
    ]
    
    print("Generating text using Transformers pipeline...")
    start_time = time.time()
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(f"Text generation completed. Time taken: {time.time() - start_time:.2f} seconds")
    
    print("Transformers Output:")
    print(outputs[0]["generated_text"])
except Exception as e:
    print(f"Error with Transformers pipeline: {str(e)}")

# Try MLX approach
try:
    mlx_model, mlx_tokenizer = initialize_mlx_model()
    
    prompt = "How to make orange juice?"
    mlx_response = generate_response_mlx(mlx_model, mlx_tokenizer, prompt)
    
    print("MLX Output:")
    print(mlx_response)
except Exception as e:
    print(f"Error with MLX approach: {str(e)}")