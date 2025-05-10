import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets

# Setup Meta-Llama-3.1-8N-Instruct with PyTorch
# Specify Model ID

model_id = "meta-llama/Llama-3.2-1B"

# Setup MPS Device

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=".")
# Set pad_token_id to eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model

try:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

def get_response(input_text, system_prompt):
    print(f"Getting response for input: {input_text}")
    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{input_text}\n<|assistant|>"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # Create attention mask to avoid warnings
    attention_mask = torch.ones_like(inputs["input_ids"])
    
    with torch.no_grad():
        outputs = model.generate(
            
            inputs["input_ids"].to(device),
            attention_mask=attention_mask.to(device),
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    return response



# Define the prompt
# The prompt should be the bandit simulation (We can build the slot machine game in a function)

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
    prompt = """You are in a Casino with 2 slot machines. 
    After each play, I'll describe what happened in words. 
    Your goal is to walk out with as many tokens as possible. 
    Each slot machine has a hidden probability of winning and losing.
    You have a choice to choose between 1 or 2. 
    You start with 1, and will be told the results of your previous actions."""
    
    previous_outputs = ""
    correct, ratio, total, previous_choice = 0
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

if __name__ == "__main__":
    main()
