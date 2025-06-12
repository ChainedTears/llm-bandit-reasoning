import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
from huggingface_hub import login
import re

login(token="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")

# Specify model ID 
# model_id = "meta-llama/Llama-3.2-1B"
model_dict = {
    '1': "Qwen/Qwen3-4B",
    '2': "Qwen/Qwen3-8B",
    '3': "meta-llama/Llama-3.1-8B",
    '4': "deepseek-ai/DeepSeek-R1",
    '5': "microsoft/phi-2",
    '6': "google/gemma-3-12b-it",
}

receive = input("Please select the model (using a number from 1-7): \n (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n Select here: ")
while receive not in model_dict:
    receive = input("Please select the model: \n (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n Select here: ")
model_id = model_dict[receive]

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

def get_response(prompt_text):
    if model is None or tokenizer is None:
        return "Model or tokenizer not loaded."

    # For dedicated chat models, using their chat template is often more robust.
    # Example for Qwen1.5-Chat:
    # messages = [
    #     {"role": "system", "content": "You are a decision-making agent. Your sole task is to choose between slot machine 1 or 2. Respond with ONLY the number '1' or '2'."},
    #     {"role": "user", "content": prompt_text} # prompt_text here would be the history part for the user message
    # ]
    # formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    # For this example, we'll stick to direct prompt construction as in the main loop.

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    start_time = time.time()
    # print("Generating response...") # Moved to main loop
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, # Pass attention_mask
            max_new_tokens=3,      # CRITICAL: Keep low for single digit output
            do_sample=True,       # Greedy decoding; set to True with low temp if output is too repetitive
            temperature=0.7,     # Use if do_sample=True
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # print(f"Generation took {time.time() - start_time:.2f} seconds.")

    newly_generated_tokens = outputs[0, input_length:]
    generated_text = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
    # The ### regex removal might not be needed if the prompt is clean
    # generated_text = re.sub(r'###.*?###.*?###', '', generated_text, flags=re.DOTALL)
    return generated_text


# def get_response(prompt):
#     if model is None or tokenizer is None:
#         return "Model or tokenizer not loaded."

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     # log timestamp
#     start_time = time.time()
#     print("Generating response...")
#     with torch.no_grad():
#         outputs = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.1,
#         top_p=1.0,
#         pad_token_id=tokenizer.pad_token_id,  # Use the pad_token_id we set earlier
#         eos_token_id=tokenizer.eos_token_id
#     ) 
#     # Only return *new* generated tokens
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("Generation took ", time.time() - start_time, "seconds.")
#     generated_text = generated_text.split("<|assistant|>")[-1].strip()
#     generated_text = re.sub(r'###.*?###.*?###', '', generated_text, flags=re.DOTALL)
#     return generated_text


# def bandit_simulation(choice):
#     random_number = secrets.randbelow(100)
    
#     if choice == 1:
#         if random_number < 30:
#             return "won"
#         else:
#             return "lost" 
#     if choice == 2:
#         if random_number < 65:
#             return "won"
#         else: 
#             return "lost"



def bandit_simulation(choice):
    random_number = secrets.randbelow(100)

    if choice == 1:  # 20% 1st, 50% 2nd, 30% 3rd
        if random_number < 20:
            return "1st"
        elif random_number < 70:  # 20–69
            return "2nd"
        else:  # 70–99
            return "3rd"

    elif choice == 2:  # 70% 1st, 20% 2nd, 10% 3rd
        if random_number < 70:
            return "1st"
        elif random_number < 90:  # 70–89
            return "2nd"
        else:  # 90–99
            return "3rd"

    elif choice == 3:  # 40% 1st, 25% 2nd, 35% 3rd
        if random_number < 40:
            return "1st"
        elif random_number < 65:  # 40–64
            return "2nd"
        else:  # 65–99
            return "3rd"
    elif choice == 4: # 35% 1st, 55% 2nd, 10% 3rd
        if random_number < 35: 
            return "1st"
        elif random_number < 90: #35-89
            return "2nd"
        else:
            return "3rd"
    print(f"Error in bandit_simulation with choice: {choice}")
    return "error"


global_history = []
correct_counter = 0

def main():
    cumulative_reward = 0
    previous_outputs = ""
    correct_ai_choices, total_ai_decisions, previous_ai_choice = 0, 0, 1 # Metrics for AI

    max_iterations = 25 # Number of decisions the AI will make
    iteration_results = [] 
    for i in range(max_iterations):
        iteration_num = i + 1
        print(f"------------- Iteration {iteration_num} -------------")

        # Construct the prompt: Strong instructions + Few-shot examples
        prompt = f"""You are a decision-making agent. Your task is to select the horse most likely to finish in 1st place—or, if not, then in 2nd.
Based on the history of 1st, 2nd, and 3rd place, decide which horse to choose next.
Output ONLY the number '1' or the number '2' or the number '3' or the number '4'. Do not include any other words, explanations, or formatting.

Example 1:
History:
Horse 4 1st
Horse 3 3rd
Horse 1 3rd
Horse 4 2nd
Horse 2 1st
Horse 3 1st
Your choice (1 or 2 or 3 or 4): 2

Example 2:
History:
Horse 2 2nd
Horse 4 1st
Horse 4 3rd
Horse 2 2nd
Horse 3 1st
Horse 1 3rd
Your choice (1 or 2 or 3 or 4): 3

Current situation:
History:
{previous_outputs}
Your choice (1 or 2 or 3 or 4):""" # The final line cues the model

        ai_response_raw = get_response(prompt)
        print(f"Raw AI Response: {ai_response_raw}")

        ai_choice = None
        # Stricter parsing: expect '1' or '2' at the beginning of the response
        match = re.match(r'^\s*([1-4])\b', ai_response_raw)
        if match:
            try:
                ai_choice = int(match.group(1))
            except ValueError:
                print(f"AI response parsing error (ValueError) from '{ai_response_raw}'.")
        else: # Fallback if no direct 1 or 2 found at the start
            match_fallback = re.search(r'\b([1-4])\b', ai_response_raw) # Look for 1 or 2 anywhere
            if match_fallback:
                try:
                    ai_choice = int(match_fallback.group(1))
                    print(f"Used fallback regex to find choice: {ai_choice}")
                except ValueError:
                    print(f"AI response parsing error (ValueError) on fallback from '{ai_response_raw}'.")


        if ai_choice not in [1, 2, 3, 4]:
            print(f"AI did not output a clear 1 or 2 or 3. Asking again...")
            while ai_choice not in [1, 2, 3]:
                ai_response_raw = get_response(prompt)
                match = re.match(r'^\s*([1-4])\b', ai_response_raw)
                if match:
                    ai_choice = int(match.group(1))



        print(f"AI chose: Horse {ai_choice}")

        total_ai_decisions += 1
        # Count as "correct" if AI picks Horse 2
        result = bandit_simulation(ai_choice)
        if ai_choice == 2:
            correct_ai_choices += 1
        if result == "1st":
            cumulative_reward += 2
        elif result == "2nd":
            cumulative_reward += 1


        current_choice_str = f"Horse {ai_choice} {result}\n"
        previous_outputs += current_choice_str # Add current result to history for next turn
        previous_ai_choice = ai_choice # Update previous AI choice
        global correct_counter
        correct_counter += 1 if result == "1st" else 0

        print(f"Outcome: Horse {ai_choice} {result}.")
        current_ratio = correct_ai_choices / total_ai_decisions if total_ai_decisions > 0 else 0
        print(f"AI 'Correct Choice' (picked Horse 2) Ratio: {correct_ai_choices}/{total_ai_decisions} = {current_ratio:.2f}")
    
    final_ratio = correct_ai_choices / total_ai_decisions if total_ai_decisions > 0 else 0
    global_history.append([final_ratio, cumulative_reward])
    print("--------------------------------")
    print(f"Final ratio: {final_ratio:.2f}")
            # For brevity, don't print full history each time
            # print(f"Updated History (last 5 lines):\n{''.join(previous_outputs.splitlines(True)[-5:])}")

if __name__ == "__main__":
    for i in range(500):
        print(f"------------- Test {i+1} -------------")
        main()
    # Writes output into result.txt; in case connection closes for runpod
    with open("result.txt", "w") as f:
        if isinstance(global_history, list):
            for item in global_history:
                f.write(str(item) + ", ")
        else:
            f.write(str(global_history) + "\n")
            
        average_ratio = sum(global_history) / len(global_history)
        print(f"Average ratio: {average_ratio:.2f}\n")
        f.write(f"Average ratio: {average_ratio:.2f}\n")
        raw_accuracy = correct_counter / 500
        print(f"Raw accuracy: {raw_accuracy}\n")
        f.write(f"Raw accuracy: {raw_accuracy}\n")
    print("Results have been written to result.txt")
