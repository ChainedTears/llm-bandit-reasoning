import os
import sys
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
from huggingface_hub import login
import re
import numpy as np

# Ensure Hugging Face token is available if downloading new models.
# Login via `huggingface-cli login` or ensure token is in environment.
# login(token="YOUR_HF_TOKEN_HERE") # Uncomment and replace if providing token directly

model_dict = {
    '1': "Qwen/Qwen2-1.5B-Instruct",
    '2': "Qwen/Qwen2-7B-Instruct",
    '3': "meta-llama/Meta-Llama-3-8B-Instruct",
    '4': "mistralai/Mistral-7B-Instruct-v0.2",
    '5': "microsoft/phi-2",
    '6': "google/gemma-2-9b-it",
}

print("Please select the model (using a number):")
for k, v_model_id in model_dict.items():
    model_name_simple = v_model_id.split('/')[-1]
    print(f" ({k}) {model_name_simple}")

receive = input("Select here: ")
while receive not in model_dict:
    receive = input(f"Invalid selection. Please choose from {list(model_dict.keys())}: ")
model_id = model_dict[receive]
safe_model_name_for_filename = model_id.replace("/", "_")

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

tokenizer = None
model = None
CACHE_DIR = "./model_cache" # Cache directory for models and tokenizers
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer for {model_id}: {e}")
    sys.exit(1)

if tokenizer:
    try:
        dtype = torch.float32
        if device.type == "mps":
            dtype = torch.float16
        elif device.type == "cuda":
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        
        print(f"Loading model {model_id} with dtype: {dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
            # trust_remote_code=True, # Uncomment if this specific model requires it
        ).to(device)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        sys.exit(1)

if tokenizer and tokenizer.pad_token_id is None:
    print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
    tokenizer.pad_token_id = tokenizer.eos_token_id

def get_response(prompt_text, current_tokenizer, current_model, current_device):
    if current_model is None or current_tokenizer is None:
        return "Model or tokenizer not loaded."

    max_prompt_len = getattr(current_tokenizer, 'model_max_length', 2048) // 2
    
    inputs = current_tokenizer(
        prompt_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_prompt_len 
    ).to(current_device)
    
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = current_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=5,       # CRITICAL: Keep low for single digit '1' or '2'
            do_sample=False,        # Greedy decoding for more deterministic output
            pad_token_id=current_tokenizer.pad_token_id,
            eos_token_id=current_tokenizer.eos_token_id
        )
    
    newly_generated_tokens = outputs[0, input_length:]
    generated_text = current_tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
    return generated_text

def bandit_simulation(choice):
    random_number = secrets.randbelow(100)
    if choice == 1: # Slot Machine 1: 30% win rate
        if random_number < 30:
            return 1, "won"
        else:
            return 0, "lost"
    elif choice == 2: # Slot Machine 2: 65% win rate
        if random_number < 65:
            return 1, "won"
        else:
            return 0, "lost"
    else:
        print(f"Error: Invalid choice '{choice}' in bandit_simulation. Defaulting to loss.")
        return 0, "error"

def run_simulation(current_model_id, current_tokenizer, current_model, current_device, num_iterations=25, run_id=1):
    previous_outputs_history_str = ""

    all_ai_choices_this_run = []
    rewards_obtained_this_run = []
    instantaneous_regrets_this_run = []
    optimal_arm_selections_this_run = [] 

    P_OPTIMAL_ARM_EXPECTED_REWARD = 0.65
    P_MACHINE_1_EXPECTED_REWARD = 0.30
    P_MACHINE_2_EXPECTED_REWARD = 0.65

    print(f"\n--- Starting Simulation Run {run_id} for {current_model_id} ({num_iterations} iterations) ---")

    for i in range(num_iterations):
        iteration_num = i + 1
        print(f"\n----- Iteration {iteration_num}/{num_iterations} -----")

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
{previous_outputs_history_str}Your choice (1 or 2):"""

        ai_choice = None
        attempts = 0
        max_parsing_attempts = 3

        while attempts < max_parsing_attempts and ai_choice is None:
            if attempts > 0:
                print(f"Retrying AI response generation/parsing (attempt {attempts + 1})...")
            
            ai_response_raw = get_response(prompt, current_tokenizer, current_model, current_device)
            print(f"Raw AI Response: '{ai_response_raw}'")

            match = re.match(r'^\s*([12])\b', ai_response_raw) # Strict parsing
            if match:
                ai_choice = int(match.group(1))
            else: 
                match_fallback = re.search(r'\b([12])\b', ai_response_raw) # Fallback parsing
                if match_fallback:
                    ai_choice = int(match_fallback.group(1))
                    print(f"Used fallback regex to find choice: {ai_choice}")
                else:
                    print(f"AI did not output a clear '1' or '2' in '{ai_response_raw}'.")
            
            if ai_choice not in [1, 2]:
                ai_choice = None 
            attempts += 1

        if ai_choice is None:
            print(f"Failed to get a valid choice from AI. Defaulting to random for this iteration.")
            ai_choice = secrets.choice([1, 2]) 

        print(f"AI chose: Machine {ai_choice}")

        reward_value, outcome_str = bandit_simulation(ai_choice)
        print(f"Outcome: Machine {ai_choice} {outcome_str} (Reward: {reward_value})")

        chosen_arm_expected_reward = P_MACHINE_1_EXPECTED_REWARD if ai_choice == 1 else P_MACHINE_2_EXPECTED_REWARD
        instantaneous_regret = P_OPTIMAL_ARM_EXPECTED_REWARD - chosen_arm_expected_reward
        
        all_ai_choices_this_run.append(ai_choice)
        rewards_obtained_this_run.append(reward_value)
        instantaneous_regrets_this_run.append(instantaneous_regret)
        optimal_arm_selections_this_run.append(1 if ai_choice == 2 else 0)

        current_choice_str_for_history = f"Slot Machine {ai_choice} {outcome_str}\n"
        previous_outputs_history_str += current_choice_str_for_history

    cumulative_rewards_this_run = np.cumsum(rewards_obtained_this_run).tolist() if rewards_obtained_this_run else []
    cumulative_regrets_this_run = np.cumsum(instantaneous_regrets_this_run).tolist() if instantaneous_regrets_this_run else []
    
    cumulative_optimal_selections = np.cumsum(optimal_arm_selections_this_run).tolist() if optimal_arm_selections_this_run else []
    optimal_arm_selection_ratio_this_run = [
        cumulative_optimal_selections[i] / (i + 1) for i in range(len(cumulative_optimal_selections))
    ] if cumulative_optimal_selections else []

    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"bandit_results_{safe_model_name_for_filename}_run{run_id}_{current_time_str}.json"
    
    results_data = {
        "model_id": current_model_id,
        "run_id": run_id,
        "timestamp": current_time_str,
        "num_iterations": num_iterations,
        "choices_per_iteration": all_ai_choices_this_run,
        "rewards_obtained_per_iteration": rewards_obtained_this_run,
        "instantaneous_regrets_per_iteration": instantaneous_regrets_this_run,
        "cumulative_rewards_per_iteration": cumulative_rewards_this_run,
        "cumulative_regrets_per_iteration": cumulative_regrets_this_run,
        "optimal_arm_selection_ratio_per_iteration": optimal_arm_selection_ratio_this_run,
    }

    results_output_dir = "simulation_results"
    os.makedirs(results_output_dir, exist_ok=True)
    full_output_path = os.path.join(results_output_dir, output_filename)
    
    try:
        with open(full_output_path, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"\n--- Simulation Run {run_id} Complete ---")
        print(f"Results saved to: {full_output_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        return None

    final_total_reward = cumulative_rewards_this_run[-1] if cumulative_rewards_this_run else 0
    final_total_regret = cumulative_regrets_this_run[-1] if cumulative_regrets_this_run else 0
    final_optimal_ratio = optimal_arm_selection_ratio_this_run[-1] if optimal_arm_selection_ratio_this_run else 0
    
    print(f"Final Total Reward: {final_total_reward}")
    print(f"Final Total Regret: {final_total_regret}")
    print(f"Final Optimal Arm Selection Ratio: {final_optimal_ratio:.2f}")

    return full_output_path

if __name__ == "__main__":
    if model is None or tokenizer is None:
        print("Exiting due to model or tokenizer loading failure.")
        sys.exit(1)
        
    TOTAL_SIMULATION_RUNS = 1 # Number of independent simulation runs
    ITERATIONS_PER_RUN = 25   # Number of decisions AI makes in one run

    all_run_json_paths = []
    for run_num in range(1, TOTAL_SIMULATION_RUNS + 1):
        json_file_path = run_simulation(
            current_model_id=model_id,
            current_tokenizer=tokenizer,
            current_model=model,
            current_device=device,
            num_iterations=ITERATIONS_PER_RUN,
            run_id=run_num
        )
        if json_file_path:
            all_run_json_paths.append(json_file_path)
        
        if TOTAL_SIMULATION_RUNS > 1 and run_num < TOTAL_SIMULATION_RUNS:
            print(f"Pausing before next run...")
            time.sleep(5) 

    print("\n--- All Simulation Runs Complete ---")
    if all_run_json_paths:
        print("Generated JSON file(s):")
        for p in all_run_json_paths:
            print(p)
    else:
        print("No JSON files were generated.")
