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
    '7': "openai/whisper-large-v3"
}
receive = input("Please select the model (using a number from 1-7): \n (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Deepseek R1 \n (5) Phi 2 \n (6) Gemma 3-12B \n (7) Whisper V3 \n Select here: ")
while receive not in model_dict:
    receive = input("Please select the model (using a number from 1-7): \n (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Deepseek R1 \n (5) Phi 2 \n (6) Gemma 3-12B \n (7) Whisper V3 \n Select here: ")
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
            max_new_tokens=5,      # CRITICAL: Keep low for single digit output
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


def bandit_simulation(choice):
    random_number = secrets.randbelow(100)
    if choice == 1: # 30% win rate
        return "won" if random_number < 30 else "lost"
    if choice == 2: # 65% win rate (objectively better)
        return "won" if random_number < 65 else "lost"
    print(f"Error in bandit_simulation with choice: {choice}")
    return "error"


def main():
    previous_outputs = ""
    correct_ai_choices, total_ai_decisions, previous_ai_choice = 0, 0, 1 # Metrics for AI

    # Seed history for a better start
    initial_history_seed = [
        (1, bandit_simulation(1)),
        (2, bandit_simulation(2)),
        (1, bandit_simulation(1)),
        (2, bandit_simulation(2)),
    ]
    for ch, res in initial_history_seed:
        previous_outputs += f"Slot Machine {ch} {res}\n"

    print("Initial History:")
    print(previous_outputs)

    max_iterations = 25 # Number of decisions the AI will make

    log_filename = "qwen4b.txt"

    with open(log_filename, 'w') as log_file:
        log_file.write("IterationBlock, TotalDecisions, CorrectChoices, CorrectRatio\n")
        
    
        for j in range(500):
            for i in range(max_iterations):
                iteration_num = i + 1
                print(f"------------- Iteration {iteration_num} -------------")
        
                # Construct the prompt: Strong instructions + Few-shot examples
                prompt = f"""You are a decision-making agent. Your task is to choose between slot machine 1 or 2.
        Based on the history of wins and losses, decide which machine to play next.
        Output ONLY the number '1' or the number '2'. Do not include any other words, explanations, or formatting.
        
        Example 1:
        History:
        Slot Machine 1 lost
        Slot Machine 2 won
        Slot Machine 2 won
        Slot Machine 1 lost
        Slot Machine 2 lost
        Slot Machine 1 lost
        Your choice (1 or 2): 2
        
        Example 2:
        History:
        Slot Machine 1 won
        Slot Machine 1 won
        Slot Machine 2 lost
        Slot Machine 1 lost
        Slot Machine 2 won
        Slot Machine 1 won
        Your choice (1 or 2): 1
        
        Current situation:
        History:
        {previous_outputs}Your choice (1 or 2):""" # The final line cues the model
        
                # print(f"DEBUG: Prompt sent to AI (last 300 chars):\n...{prompt[-300:]}") # For debugging
        
                ai_response_raw = get_response(prompt)
                print(f"Raw AI Response: {ai_response_raw}")
        
                ai_choice = None
                # Stricter parsing: expect '1' or '2' at the beginning of the response
                match = re.match(r'^\s*([12])\b', ai_response_raw)
                if match:
                    try:
                        ai_choice = int(match.group(1))
                    except ValueError:
                        print(f"AI response parsing error (ValueError) from '{ai_response_raw}'.")
                else: # Fallback if no direct 1 or 2 found at the start
                    match_fallback = re.search(r'\b([12])\b', ai_response_raw) # Look for 1 or 2 anywhere
                    if match_fallback:
                        try:
                            ai_choice = int(match_fallback.group(1))
                            print(f"Used fallback regex to find choice: {ai_choice}")
                        except ValueError:
                             print(f"AI response parsing error (ValueError) on fallback from '{ai_response_raw}'.")
        
        
                if ai_choice not in [1, 2]:
                    print(f"AI did not output a clear 1 or 2. Asking again...")
                    while ai_choice not in [1, 2]:
                        ai_response_raw = get_response(prompt)
                        match = re.match(r'^\s*([12])\b', ai_response_raw)
                        if match:
                            ai_choice = int(match.group(1))
        
        
        
                print(f"AI chose: Machine {ai_choice}")
        
                total_ai_decisions += 1
                # Machine 2 is objectively better (65% win rate).
                # Count as "correct" if AI picks machine 2.
                if ai_choice == 2:
                    correct_ai_choices += 1
        
                result = bandit_simulation(ai_choice)
                current_choice_str = f"Slot Machine {ai_choice} {result}\n"
                previous_outputs += current_choice_str # Add current result to history for next turn
                previous_ai_choice = ai_choice # Update previous AI choice
        
                print(f"Outcome: Machine {ai_choice} {result}.")
                current_ratio = correct_ai_choices / total_ai_decisions if total_ai_decisions > 0 else 0
                print(f"AI 'Correct Choice' (picked Machine 2) Ratio: {correct_ai_choices}/{total_ai_decisions} = {current_ratio:.2f}")
                # For brevity, don't print full history each time
                # print(f"Updated History (last 5 lines):\n{''.join(previous_outputs.splitlines(True)[-5:])}")
            log_file.write(f"{j+1},{total_ai_decisions},{correct_ai_choices},{current_ratio_at_step:.4f}\n")

    print("\n--- Simulation Complete ---")
    final_ratio = correct_ai_choices / total_ai_decisions if total_ai_decisions > 0 else 0
    print(f"Final AI 'Correct Choice' (picked Machine 2) Ratio: {correct_ai_choices}/{total_ai_decisions} = {final_ratio:.2f}")
    # print("\nFull History:")
    # print(previous_outputs)

if __name__ == "__main__":
    main()

# Main execution loop
# def main():
#     previous_outputs = ""
#     correct, ratio, total, previous_choice = 0, 0.0, 0, 1
#     # Run for 10 iterations
#     while total < 100 or (ratio < 0.8 and total > 20):
#         prompt = f"""I am in a casino with two slot machines, 1 and 2.
# I will output either a 1 or a 2, based on the history of my choices and results, which are:
# {previous_outputs}
# I am in a casino with two slot machines, 1 and 2.
# I will output either a 1 or a 2, based on the history of my choices and results, which are:
# {previous_outputs}
# I will give my output in this format:
# Output: <number>

# Output:
# """
#         if previous_choice == 2:
#             correct += 1
#         total += 1
#         ratio = correct / total
#         print(f"------------- Iteration {total} -------------")
#         print(f"Correct: {correct} Ratio: {ratio} Total: {total}")
        
#         if total == 0:
#             choice = 1
#             previous_choice = 1
#             result = bandit_simulation(choice)
#             previous_outputs += f"Slot Machine {choice} {result}\n"
#             print(previous_outputs)
#         else:
#             ai_response = get_response(prompt)
            
#             try:
#                 print(ai_response)
#                 choice = int(re.search(r'Output:\s*(\d+)', ai_response).group(1))
#                 print(choice)
#                 previous_choice = choice
#                 if choice not in [1, 2]:
#                     print(f"Invalid choice '{choice}'")
#                     return
#             except ValueError:
#                 print(f"Invalid response '{ai_response}'")
#                 return
#             result = bandit_simulation(choice)
#             previous_outputs += f"Slot Machine {choice} {result}\n"
#             print(previous_outputs)

# if __name__ == "__main__":
#     main()
