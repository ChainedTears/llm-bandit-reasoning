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
    if choice == "PCR": # 55% success rate 
        return "success" if random_number < 55 else "lost"
    if choice == "Antigen": # 30% success rate 
        return "success" if random_number < 30 else "lost"
    if choice == "NAAT" # 70% success rate (objectively better)
        return "success" if random_number < 70 else "lost"
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
        prompt = f"""You are a decision-making agent. Your task is to choose the COVID test that fails the least.
Based on the history of success and fail, decide which COVID test to choose next.
Output ONLY the word 'PCR' or the word 'Antigen' or the word 'NAAT'. Do not include any other words, explanations, or formatting.

Example 1:
History:
Test PCR fail
Test Antigen fail
Test PCR success
Test PCR fail
Test PCR success
Test NAAT success
Your choice (PCR or Antigen or NAAT): PCR

Example 2:
History:
Test NAAT fail
Test Antigen fail
Test PCR success
Test Antigen 2 fail
Test PCR success
Test NAAT success
Your choice (PCR or Antigen or NAAT): NAAT

Current situation:
History:
{previous_outputs}
Your choice (PCR or Antigen or NAAT):""" # The final line cues the model

        ai_response_raw = get_response(prompt)
        print(f"Raw AI Response: {ai_response_raw}")

        ai_choice = None
        # Stricter parsing: expect '1' or '2' at the beginning of the response
        match = re.match(r'^\s*(PCR|Antigen|NAAT)\b', ai_response_raw, re.IGNORECASE)
        if match:
            ai_choice = match.group(1).capitalize()  # Normalize to consistent casing
        else:
            # Fallback search anywhere in the response
            match_fallback = re.search(r'\b(PCR|Antigen|NAAT)\b', ai_response_raw, re.IGNORECASE)
            if match_fallback:
                ai_choice = match_fallback.group(1).capitalize()
                print(f"Used fallback regex to find choice: {ai_choice}")


        if ai_choice not in ["PCR", "Antigen", "NAAT"]:
            print(f"AI did not output a clear PCR, Antigen, or NAAT. Asking again...")
            while ai_choice not in ["PCR", "Antigen", "NAAT"]:
                ai_response_raw = get_response(prompt)
                match = re.match(r'^\s*(PCR|Antigen|NAAT)\b', ai_response_raw, re.IGNORECASE)
                if match:
                    ai_choice = match.group(1).capitalize()



        print(f"AI chose: {ai_choice} test")

        total_ai_decisions += 1

        # Count as "correct" if AI selects NAAT (assumed to be the best-performing test overall)
        if ai_choice == "NAAT":
            correct_ai_choices += 1

        result = bandit_simulation(ai_choice)
        if result == "success":
            cumulative_reward += 1
        elif result == "fail":
            cumulative_reward -= 1
            
        current_choice_str = f"Test {ai_choice} {result}\n"
        previous_outputs += current_choice_str # Add current result to history for next turn
        previous_ai_choice = ai_choice # Update previous AI choice
        global correct_counter
        correct_counter += 1 if result == "success" else 0

        print(f"Outcome: Test {ai_choice} {result}.")
        current_ratio = correct_ai_choices / total_ai_decisions if total_ai_decisions > 0 else 0
        print(f"AI 'Correct Choice' (picked NAAT) Ratio: {correct_ai_choices}/{total_ai_decisions} = {current_ratio:.2f}")
    
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
