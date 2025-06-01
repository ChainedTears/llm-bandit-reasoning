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
# login(token="YOUR_HF_TOKEN_HERE")

# User's original model dictionary
model_dict = {
    '1': "Qwen/Qwen3-4B",
    '2': "Qwen/Qwen3-8B",
    '3': "meta-llama/Llama-3.1-8B",
    '4': "deepseek-ai/DeepSeek-R1",
    '5': "microsoft/phi-2",
    '6': "google/gemma-3-12b-it",
    '7': "openai/whisper-large-v3"
}

print("Please select the model (using a number from 1-7):")
print(" (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n (7) Whisper Large V3")
receive = input("Select here: ")

while receive not in model_dict:
    print("\nInvalid selection. Please select the model (using a number from 1-7):")
    print(" (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n (7) Whisper Large V3")
    receive = input("Select here: ")

model_id = model_dict[receive]
safe_model_name_for_filename = model_id.replace("/", "_")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

tokenizer = None
model = None
CACHE_DIR = "./model_cache"
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
        if device.type == "mps": dtype = torch.float16
        elif device.type == "cuda":
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported(): dtype = torch.bfloat16
            else: dtype = torch.float16
        print(f"Loading model {model_id} with dtype: {dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=CACHE_DIR,
            # trust_remote_code=True,
        ).to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        sys.exit(1)

if tokenizer and tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def get_response(prompt_text, current_tokenizer, current_model, current_device):
    if current_model is None or current_tokenizer is None: return "Model or tokenizer not loaded."
    if not hasattr(current_model, 'generate'):
        print(f"Error: Model {model_id} lacks 'generate' method (not a Causal LM?).")
        return "MODEL_ERROR_NO_GENERATE"
    max_prompt_len = getattr(current_tokenizer, 'model_max_length', 2048) // 2
    inputs = current_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = current_model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=5, do_sample=False,
            pad_token_id=current_tokenizer.pad_token_id, eos_token_id=current_tokenizer.eos_token_id
        )
    return current_tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()

def bandit_simulation(choice):
    r = secrets.randbelow(100)
    if choice == 1: return (1, "won") if r < 30 else (0, "lost")
    if choice == 2: return (1, "won") if r < 65 else (0, "lost")
    return 0, "error"

def run_simulation(current_model_id, current_tokenizer, current_model, current_device, num_iterations=25, run_id=1):
    # This function now prepares and returns the data for one run, instead of saving it.
    history_str = ""
    choices, rewards, regrets, optimal_selections = [], [], [], []
    P_OPT, P_M1, P_M2 = 0.65, 0.30, 0.65

    # Reduced print verbosity for many runs, but run_id helps track progress
    if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: # Print for first, every 20th, or if few iterations
        print(f"\n--- Running Simulation: Run {run_id}, Model {current_model_id}, Iterations {num_iterations} ---")

    for i in range(num_iterations):
        iteration_num = i + 1
        prompt = f"""You are a decision-making agent. Your task is to choose between slot machine 1 or 2.
Based on the history of wins and losses, decide which machine to play next.
Output ONLY the number '1' or the number '2'. Do not include any other words, explanations, or formatting.
Example 1: History:\nSlot Machine 1 lost\nSlot Machine 2 won\nSlot Machine 2 won\nYour choice (1 or 2): 2
Example 2: History:\nSlot Machine 1 won\nSlot Machine 1 won\nSlot Machine 2 lost\nYour choice (1 or 2): 1
Current situation:\nHistory:\n{history_str}Your choice (1 or 2):"""

        ai_choice, attempts = None, 0
        while attempts < 3 and ai_choice is None:
            raw_resp = get_response(prompt, current_tokenizer, current_model, current_device)
            if raw_resp == "MODEL_ERROR_NO_GENERATE": 
                ai_choice = secrets.choice([1,2]); 
                if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: print(f"Run {run_id}, Iter {iteration_num}: Model error, random choice: {ai_choice}"); 
                break
            match = re.match(r'^\s*([12])\b', raw_resp) or re.search(r'\b([12])\b', raw_resp)
            if match: ai_choice = int(match.group(1))
            if ai_choice not in [1,2]: ai_choice = None
            attempts +=1
        if ai_choice is None: 
            ai_choice = secrets.choice([1,2]); 
            if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: print(f"Run {run_id}, Iter {iteration_num}: AI failed, random choice.")
        
        reward_val, outcome = bandit_simulation(ai_choice)
        
        chosen_exp_reward = P_M1 if ai_choice == 1 else P_M2
        regret = P_OPT - chosen_exp_reward
        
        choices.append(ai_choice); rewards.append(reward_val); regrets.append(regret)
        optimal_selections.append(1 if ai_choice == 2 else 0)
        history_str += f"Slot Machine {ai_choice} {outcome}\n"

    cum_rewards = np.cumsum(rewards).tolist() if rewards else []
    cum_regrets = np.cumsum(regrets).tolist() if regrets else []
    cum_opt_sel = np.cumsum(optimal_selections).tolist() if optimal_selections else []
    opt_sel_ratio = [cum_opt_sel[k]/(k+1) for k in range(len(cum_opt_sel))] if cum_opt_sel else []

    final_optimal_ratio = opt_sel_ratio[-1] if opt_sel_ratio else 0
    total_actual_reward = cum_rewards[-1] if cum_rewards else 0
    
    run_data = { # This dictionary contains all data for this single run
        "run_id": run_id,
        # "timestamp_run_data_generated": time.strftime("%Y%m%d-%H%M%S"), # Can be added if needed per run
        "num_iterations_in_run": num_iterations, # Clarify this is for this specific run segment
        "choices_per_iteration": choices,
        "rewards_obtained_per_iteration": rewards,
        "instantaneous_regrets_per_iteration": regrets,
        "cumulative_rewards_per_iteration": cum_rewards,
        "cumulative_regrets_per_iteration": cum_regrets,
        "optimal_arm_selection_ratio_per_iteration": opt_sel_ratio,
        "summary_final_optimal_ratio_this_run": final_optimal_ratio,
        "summary_total_actual_reward_this_run": total_actual_reward
    }
    
    if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: # Print summary for some runs
        print(f"Run {run_id} Completed. Final Optimal Ratio: {final_optimal_ratio:.2f}, Total Reward: {total_actual_reward}")
    
    return run_data # Return the entire data structure for this run

if __name__ == "__main__":
    if model is None or tokenizer is None:
        print("Model/tokenizer failed to load. Exiting.")
        sys.exit(1)
        
    TOTAL_SIMULATION_RUNS = 500 # Run the 25-iteration simulation 500 times
    ITERATIONS_PER_RUN = 25     # Each simulation run consists of 25 iterations/choices
    
    all_runs_detailed_data = [] # List to hold the detailed data dict from each run
    
    # For aggregating overall summary statistics
    collected_final_optimal_ratios = []
    collected_total_rewards = []

    experiment_start_time = time.time()
    print(f"Starting experiment with {TOTAL_SIMULATION_RUNS} runs of {ITERATIONS_PER_RUN} iterations each...")

    for run_num in range(1, TOTAL_SIMULATION_RUNS + 1):
        # run_simulation now returns a dictionary of its detailed data
        single_run_data = run_simulation(
            model_id, tokenizer, model, device,
            num_iterations=ITERATIONS_PER_RUN,
            run_id=run_num
        )
        
        all_runs_detailed_data.append(single_run_data)
        
        # Collect summary stats for overall average calculation
        collected_final_optimal_ratios.append(single_run_data["summary_final_optimal_ratio_this_run"])
        collected_total_rewards.append(single_run_data["summary_total_actual_reward_this_run"])
        
        if TOTAL_SIMULATION_RUNS > 1 and run_num < TOTAL_SIMULATION_RUNS:
            if run_num % 20 == 0: # Progress update
                 elapsed = time.time() - experiment_start_time
                 runs_left = TOTAL_SIMULATION_RUNS - run_num
                 avg_time = elapsed / run_num
                 eta = avg_time * runs_left
                 print(f"PROGRESS: Completed {run_num}/{TOTAL_SIMULATION_RUNS} runs. ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")
    
    experiment_end_time = time.time()
    total_time_taken = experiment_end_time - experiment_start_time
    print(f"\nAll {TOTAL_SIMULATION_RUNS} simulation runs completed in {time.strftime('%H:%M:%S', time.gmtime(total_time_taken))}.")

    # Prepare the master dictionary for the single large JSON file
    master_output_data = {
        "experiment_metadata": {
            "model_id": model_id,
            "total_simulation_runs": TOTAL_SIMULATION_RUNS,
            "iterations_per_run": ITERATIONS_PER_RUN,
            "experiment_timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "total_duration_seconds": total_time_taken
        },
        "overall_summary_statistics": {
            "average_final_optimal_ratio": np.mean(collected_final_optimal_ratios) if collected_final_optimal_ratios else 0,
            "stdev_final_optimal_ratio": np.std(collected_final_optimal_ratios) if collected_final_optimal_ratios else 0,
            "average_total_reward_per_run": np.mean(collected_total_rewards) if collected_total_rewards else 0,
            "stdev_total_reward_per_run": np.std(collected_total_rewards) if collected_total_rewards else 0,
        },
        "all_runs_data": all_runs_detailed_data # List of dictionaries, each dict is one run's full data
    }

    # Save the single large JSON file
    results_output_dir = "simulation_results"
    os.makedirs(results_output_dir, exist_ok=True)
    master_filename = f"MASTER_bandit_results_{safe_model_name_for_filename}_{TOTAL_SIMULATION_RUNS}runs_{master_output_data['experiment_metadata']['experiment_timestamp']}.json"
    master_filepath = os.path.join(results_output_dir, master_filename)

    try:
        with open(master_filepath, "w") as f:
            json.dump(master_output_data, f, indent=4) # indent for readability, remove for smaller file size
        print(f"\nAll simulation data saved to ONE LARGE JSON FILE: {master_filepath}")
    except Exception as e:
        print(f"Error saving master JSON file: {e}")

    # Print overall summary statistics
    print("\n\n--- Overall Experiment Summary (from collected data) ---")
    if collected_final_optimal_ratios:
        avg_opt_ratio = master_output_data['overall_summary_statistics']['average_final_optimal_ratio']
        std_opt_ratio = master_output_data['overall_summary_statistics']['stdev_final_optimal_ratio']
        print(f"Average Final Optimal Arm Selection Ratio: {avg_opt_ratio:.4f} (StdDev: {std_opt_ratio:.4f})")
    
    if collected_total_rewards:
        avg_total_rew = master_output_data['overall_summary_statistics']['average_total_reward_per_run']
        std_total_rew = master_output_data['overall_summary_statistics']['stdev_total_reward_per_run']
        max_possible_per_run = ITERATIONS_PER_RUN * 0.65
        print(f"Average Total Reward per Run: {avg_total_rew:.3f} (StdDev: {std_total_rew:.3f}) / max optimal ~{max_possible_per_run:.2f}")
