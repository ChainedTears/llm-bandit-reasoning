import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
from huggingface_hub import login
import re

login(token="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")

# Specify model ID 
model_id = "meta-llama/Llama-3.2-1B"
# model_id = "Qwen/Qwen3-4B"

# Setup device (MPS for Mac, CUDA, fallback to CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=".")
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

# Load model
model = None
if tokenizer:
    try:
        # Determine torch_dtype based on device
        if device.type == "mps":
            dtype = torch.float16
        elif device.type == "cuda":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else: # CPU
            dtype = torch.float32
        
        print(f"Loading model with dtype: {dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=".",
            # trust_remote_code=True,
        ).to(device)
        model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Set pad_token_id to eos_token_id if not already set
# This must be done AFTER the tokenizer is loaded.
if tokenizer and tokenizer.pad_token_id is None:
    print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # For Llama models, padding side is often recommended to be 'left' for batched inference.
    # tokenizer.padding_side = "left" # Keep an eye on this if you do batching

# Generate response function

# Add a stop sequence


def get_response(prompt):
    if model is None or tokenizer is None:
        return "Model or tokenizer not loaded."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # log timestamp
    start_time = time.time()
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        # temperature=0.1,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,  # Use the pad_token_id we set earlier
        eos_token_id=tokenizer.eos_token_id
    ) 
    # Only return *new* generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generation took ", time.time() - start_time, "seconds.")
    generated_text = generated_text.split("<|assistant|>")[-1].strip()
    generated_text = re.sub(r'###.*?###.*?###', '', generated_text, flags=re.DOTALL)
    return generated_text

while True:
    print(get_response(input("Enter a question: ")))

def bandit_simulation(choice):
    random_number = secrets.randbelow(100)
    
    if choice == 1:
        if random_number < 30:
            return "You win"
        else:
            return "You lose"
    if choice == 2:
        if random_number < 65:
            return "You win"
        else: 
            return "You lose"


# Main execution loop
def main():
    prompt = """You are in a Casino with 2 slot machines. Your goal is to maximize your winnings.
    After each play, I will tell you the result. Based on the history, choose the next slot machine to play (1 or 2).
    Respond with ONLY the number of your next choice."""
    
    previous_outputs = ""
    correct, ratio, total, previous_choice = 0, 0.0, 0, 1
    # Run for 10 iterations
    while not (total < 100 and ratio > 0.8):
        if previous_choice == 2:
            correct += 1
        total += 1
        ratio = correct / total
        print(f"Iteration {total}")
        if total == 0:
            choice = 1
            result = bandit_simulation(choice)
            previous_outputs += f"Choice: {choice} Result: {result}\n"
            print(f"Choice: {choice} Result: {result}\n")
        else:
            ai_response = get_response(previous_outputs, prompt)
            
            try:
                choice = int(ai_response.strip())
                if choice not in [1, 2]:
                    print(f"Invalid choice '{choice}'")
                    return
            except ValueError:
                print(f"Invalid response '{ai_response}'")
                return
            result = bandit_simulation(choice)
            previous_outputs += f"Choice: {choice} Result: {result}\n"
            print(f"Choice: {choice} Result: {result}\n")

# if __name__ == "__main__":
#     main()
