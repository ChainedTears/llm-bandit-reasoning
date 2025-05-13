import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
from huggingface_hub import login

login(token="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")

# Specify model ID 
model_id = "mistralai/Mixtral-8x7B-v0.1"

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
def get_response(input_text, system_prompt):
    if model is None or tokenizer is None:
        return "Model or tokenizer not loaded."

    print(f"\nGetting response for input: '{input_text}' with system prompt: '{system_prompt}'")

    # Construct messages for Llama 3 chat template
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_text})

    # Apply the chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Warning: Error applying chat template: {e}. Using manual prompt (may be suboptimal).")
        formatted_messages = ""
        if system_prompt and system_prompt.strip():
            formatted_messages += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        formatted_messages += f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|>"
        formatted_messages += f"<|start_header_id|>assistant<|end_header_id|>\n\n" # Prompt for assistant
        prompt = f"<|begin_of_text|>{formatted_messages}"
        # I made ChatGPT write this, fallback for older transformers or tokenizers not supporting apply_chat_template
        # I'm not sure if this is correct, but it would probably work.

    print(f"Formatted prompt being sent to tokenizer:\n{prompt}")

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, return_attention_mask=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Define terminator tokens for generationn.
    terminator_ids = []
    if tokenizer.eos_token_id is not None:
        terminator_ids.append(tokenizer.eos_token_id)
    
    # Attempt to get the ID for <|eot_id|>
    try:
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_token_id != tokenizer.unk_token_id and eot_token_id not in terminator_ids:
            terminator_ids.append(eot_token_id)
    except Exception as e:
        print(f"Could not get token ID for <|eot_id|>: {e}")

    if not terminator_ids: # Fallback if no valid terminators found
        if tokenizer.eos_token_id is not None:
             terminator_ids = [tokenizer.eos_token_id]
        else:
            print("CRITICAL WARNING: No EOS token ID found in tokenizer. Generation might not stop correctly.")
            terminator_ids = [] # Model will rely on max_new_tokens

    print(f"Using terminator IDs for generation: {terminator_ids} (tokens: {[tokenizer.decode([tid]) for tid in terminator_ids]})")
    print(f"Using pad_token_id for generation: {tokenizer.pad_token_id} (token: {tokenizer.decode([tokenizer.pad_token_id]) if tokenizer.pad_token_id is not None else 'None'})")


    # Generate response
    with torch.no_grad():
        # output_ids will contain the prompt + generated response
        output_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,  
            eos_token_id=terminator_ids, # Stop generation on these tokens
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,      
            do_sample=True,      
            num_return_sequences=1
        )

    # Decode only the generated part of the response
    generated_ids = output_sequences[0][input_ids.shape[-1]:]
    decoded_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Raw decoded generated output (after slicing, before final strip): '{decoded_response}'")
    final_response = decoded_response.strip()
    print(f"Cleaned final response: '{final_response}'")
    return final_response

# get_response("What is the capital of France?", "You are a helpful assistant.")

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

if __name__ == "__main__":
    main()

