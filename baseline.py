import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
from huggingface_hub import login
import re

login(token="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")

# Specify model ID 
# model_id = "meta-llama/Llama-3.2-1B"
model_id = "Qwen/Qwen3-4B"

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
        # model.eval()  # Set model to evaluation mode
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
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Log timestamp
    start_time = time.time()
    print("Generating response...")

    generated = input_ids
    stop_token = "\n"
    stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)[0]

    model.eval()
    with torch.no_grad():
        for _ in range(100):  # max_new_tokens
            outputs = model(input_ids=generated, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)

            # Check if the last generated token is a stop token
            if next_token.item() == stop_token_id:
                break

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    generated_text = generated_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    generated_text = generated_text.strip()
    print("Generation took ", time.time() - start_time, "seconds.")
    return generated_text



def bandit_simulation(choice):
    random_number = secrets.randbelow(100)
    
    if choice == 1:
        if random_number < 30:
            return "won"
        else:
            return "lost"
    if choice == 2:
        if random_number < 65:
            return "won"
        else: 
            return "lost"


# Main execution loop
def main():
    previous_outputs = ""
    correct, ratio, total, previous_choice = 0, 0.0, 0, 1
    # Run for 10 iterations
    while total < 100 or (ratio < 0.8 and total > 20):
        prompt = f"""I am in a casino with two slot machines, 1 and 2.
I will output either a 1 or a 2, based on the history of my choices and results, which are:
{previous_outputs}
I am in a casino with two slot machines, 1 and 2.
I will output either a 1 or a 2, based on the history of my choices and results, which are:
{previous_outputs}
I will give my output in this format:
Output: <number>

Output:
"""
        if previous_choice == 2:
            correct += 1
        total += 1
        ratio = correct / total
        print(f"------------- Iteration {total} -------------")
        print(f"Correct: {correct} Ratio: {ratio} Total: {total}")
        
        if total == 0:
            choice = 1
            previous_choice = 1
            result = bandit_simulation(choice)
            previous_outputs += f"Slot Machine {choice} {result}\n"
            print(previous_outputs)
        else:
            ai_response = get_response(prompt)
            
            try:
                print(ai_response)
                choice = int(re.search(r'Output:\s*(\d+)', ai_response).group(1))
                print(choice)
                previous_choice = choice
                if choice not in [1, 2]:
                    print(f"Invalid choice '{choice}'")
                    return
            except ValueError:
                print(f"Invalid response '{ai_response}'")
                return
            result = bandit_simulation(choice)
            previous_outputs += f"Slot Machine {choice} {result}\n"
            print(previous_outputs)

if __name__ == "__main__":
    main()
