import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets

# Setup Meta-Llama-3.1-8N-Instruct with PyTorch
# Specify Model ID

model_id = "meta-llama/Llama-3.2-3B"

# Setup MPS Device

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model

try:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    print("Model loaded successfully")
except Exception as e:
    print("Don't do anything for now something is wrong")

def get_response(input_text, system_prompt):
    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{input_text}\n<|assistant|>"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
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

prompt = """You are in a Casino with 2 slot machines. 
    After each play , I’ll describe what happened in words. 
    Your goal is to walk out with as many tokens as possible. 
    Each slot machine has a hidden probability of winning and losing.
    You have a choice to choose between 1 or 2. 
    You start with 1, and will be told the results of your last actions"""
previous_outputs = ""
ai_response = ""


for i in range(10):
    if i == 0:
        result = bandit_simulation(1)
        previous_outputs += "Choice: " + 1 + " Result: " + result + "\n"
        print("Choice: " + 1 + " Result: " + result + "\n")
    else:
        result = bandit_simulation(ai_response)
        ai_response = get_response(previous_outputs + "Choice: " + ai_response + " Result: " + result, prompt)
        previous_outputs += "Choice: " + 1 + " Result: " + result + "\n"
