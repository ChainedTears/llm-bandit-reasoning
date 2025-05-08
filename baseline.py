import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup Meta-Llama-3.1-8N-Instruct with PyTorch
# Specify Model ID

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Setup MPS Device

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)


# Define the prompt
# The prompt should be the bandit simulation (We can build the slot machine game in a function)