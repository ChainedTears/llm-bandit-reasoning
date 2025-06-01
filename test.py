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

# Using the curated model list that generally works better with the detailed prompt
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
    print("\nInvalid selection. Please select the model (using a number):")
    for k, v_model_id in model_dict.items():
        model_name_simple = v_model_id.split('/')[-1]
        print(f" ({k}) {model_name_simple}")
    receive = input(f"Select here from {list(model_dict.keys())}: ")
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
    if not hasattr(current_model, 'generate'): return "MODEL_ERROR_NO_GENERATE"
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
    # This function now returns detailed lists for one run.
    history_str = ""
    choices, rewards, regrets, optimal_selections = [], [], [], []
    P_OPT, P_M1, P_M2 = 0.65, 0.30, 0.65

    if run_id % 50 == 0 or run_id == 1:
        print(f"\n--- Starting Simulation Run {run_id}/{TOTAL_SIMULATION_RUNS} ---")

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
                ai_choice = secrets.choice([1,2]); break 
            match = re.match(r'^\s*([12])\b', raw_resp) or re.search(r'\b([12])\b', raw_resp)
            if match: ai_choice = int(match.group(1))
            if ai_choice not in [1,2]: ai_choice = None
            attempts +=1
        if ai_choice is None: ai_choice = secrets.choice([1,2])
        
        reward_val, outcome = bandit_simulation(ai_choice)
        chosen_exp_reward = P_M1 if ai_choice == 1 else P_M2
        regret = P_OPT - chosen_exp_reward
        
        choices.append(ai_choice); rewards.append(reward_val); regrets.append(regret)
        optimal_selections.append(1 if ai_choice == 2 else 0)
        history_str += f"Slot Machine {ai_choice} {outcome}\n"

    cum_rewards = np.cumsum(rewards).tolist() if rewards else [0]*num_iterations
    cum_regrets = np.cumsum(regrets).tolist() if regrets else [0]*num_iterations
    cum_opt_sel = np.cumsum(optimal_selections).tolist() if optimal_selections else [0]*num_iterations
    opt_sel_ratio = [cum_opt_sel[k]/(k+1) if (k+1)>0 else 0 for k in range(len(cum_opt_sel))] if cum_opt_sel else [0]*num_iterations
    
    # Ensure all lists have the full 'num_iterations' length, padding with last value or 0 if empty initially
    # This is important for consistent averaging later if a run somehow has short lists.
    # However, the logic above should already produce lists of length num_iterations.
    # We'll rely on the np.cumsum behavior and list comprehensions to handle this.
    # If lists are empty due to num_iterations=0, they'll be initialized with zeros matching num_iterations.

    return {
        "choices_per_iteration": choices, # Still useful for some analyses if needed
        "cumulative_rewards_per_iteration": cum_rewards,
        "cumulative_regrets_per_iteration": cum_regrets,
        "optimal_arm_selection_ratio_per_iteration": opt_sel_ratio,
        "summary_final_optimal_ratio_this_run": opt_sel_ratio[-1] if opt_sel_ratio else 0,
        "summary_total_actual_reward_this_run": cum_rewards[-1] if cum_rewards else 0
    }

if __name__ == "__main__":
    if model is None or tokenizer is None:
        print("Model/tokenizer failed to load. Exiting.")
        sys.exit(1)
        
    TOTAL_SIMULATION_RUNS = 500 
    ITERATIONS_PER_RUN = 25     
    
    # Lists to store the per-iteration lists from ALL runs
    all_runs_cum_rewards = [] # List of 500 lists (each sublist has 25 cum_reward values)
    all_runs_cum_regrets = []
    all_runs_opt_arm_ratios = []
    # We can also store choices if we want to average P(choosing arm 2) at each step
    all_runs_choices_arm2_prob = []


    # For overall summary statistics
    collected_final_optimal_ratios = []
    collected_total_rewards = []

    experiment_start_time = time.time()
    print(f"Starting experiment: {TOTAL_SIMULATION_RUNS} runs of {ITERATIONS_PER_RUN} iterations each for model {model_id}...")

    for run_num in range(1, TOTAL_SIMULATION_RUNS + 1):
        run_data = run_simulation(
            model_id, tokenizer, model, device,
            num_iterations=ITERATIONS_PER_RUN, run_id=run_num
        )
        
        # Store the full per-iteration lists
        all_runs_cum_rewards.append(run_data["cumulative_rewards_per_iteration"])
        all_runs_cum_regrets.append(run_data["cumulative_regrets_per_iteration"])
        all_runs_opt_arm_ratios.append(run_data["optimal_arm_selection_ratio_per_iteration"])
        
        # Calculate P(choosing arm 2) for this run and store it
        prob_arm2_this_run = [1 if choice == 2 else 0 for choice in run_data["choices_per_iteration"]]
        all_runs_choices_arm2_prob.append(prob_arm2_this_run)

        # Collect final summary stats for overall averages
        collected_final_optimal_ratios.append(run_data["summary_final_optimal_ratio_this_run"])
        collected_total_rewards.append(run_data["summary_total_actual_reward_this_run"])
        
        if run_num % 20 == 0: # Progress update
            elapsed = time.time() - experiment_start_time; runs_left = TOTAL_SIMULATION_RUNS - run_num
            avg_time = elapsed / run_num if run_num > 0 else 0; eta = avg_time * runs_left
            print(f"PROGRESS: Completed {run_num}/{TOTAL_SIMULATION_RUNS} runs. ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")
    
    total_time_taken = time.time() - experiment_start_time
    print(f"\nAll {TOTAL_SIMULATION_RUNS} simulation runs completed in {time.strftime('%H:%M:%S', time.gmtime(total_time_taken))}.")

    # --- Averaging the per-iteration data across all runs ---
    # Convert lists of lists to numpy arrays for easy column-wise mean calculation
    # Ensure all inner lists are of the same length (ITERATIONS_PER_RUN)
    # The run_simulation function should ensure this.
    
    # Check if all lists are non-empty before proceeding with np.array and mean
    if all_runs_cum_rewards and len(all_runs_cum_rewards[0]) == ITERATIONS_PER_RUN:
        avg_cum_rewards_per_iteration = np.mean(np.array(all_runs_cum_rewards), axis=0).tolist()
        std_cum_rewards_per_iteration = np.std(np.array(all_runs_cum_rewards), axis=0).tolist()
    else:
        avg_cum_rewards_per_iteration = [0]*ITERATIONS_PER_RUN
        std_cum_rewards_per_iteration = [0]*ITERATIONS_PER_RUN
        print("Warning: Could not compute average cumulative rewards due to inconsistent data.")

    if all_runs_cum_regrets and len(all_runs_cum_regrets[0]) == ITERATIONS_PER_RUN:
        avg_cum_regrets_per_iteration = np.mean(np.array(all_runs_cum_regrets), axis=0).tolist()
        std_cum_regrets_per_iteration = np.std(np.array(all_runs_cum_regrets), axis=0).tolist()
    else:
        avg_cum_regrets_per_iteration = [0]*ITERATIONS_PER_RUN
        std_cum_regrets_per_iteration = [0]*ITERATIONS_PER_RUN
        print("Warning: Could not compute average cumulative regrets due to inconsistent data.")

    if all_runs_opt_arm_ratios and len(all_runs_opt_arm_ratios[0]) == ITERATIONS_PER_RUN:
        avg_opt_arm_ratios_per_iteration = np.mean(np.array(all_runs_opt_arm_ratios), axis=0).tolist()
        std_opt_arm_ratios_per_iteration = np.std(np.array(all_runs_opt_arm_ratios), axis=0).tolist()
    else:
        avg_opt_arm_ratios_per_iteration = [0]*ITERATIONS_PER_RUN
        std_opt_arm_ratios_per_iteration = [0]*ITERATIONS_PER_RUN
        print("Warning: Could not compute average optimal arm ratios due to inconsistent data.")

    if all_runs_choices_arm2_prob and len(all_runs_choices_arm2_prob[0]) == ITERATIONS_PER_RUN:
        avg_prob_choice_arm2_per_iteration = np.mean(np.array(all_runs_choices_arm2_prob), axis=0).tolist()
        std_prob_choice_arm2_per_iteration = np.std(np.array(all_runs_choices_arm2_prob), axis=0).tolist()
    else:
        avg_prob_choice_arm2_per_iteration = [0]*ITERATIONS_PER_RUN
        std_prob_choice_arm2_per_iteration = [0]*ITERATIONS_PER_RUN
        print("Warning: Could not compute average P(choice=Arm2) due to inconsistent data.")


    # Prepare the master dictionary for the single large JSON file
    master_output_data = {
        "experiment_metadata": {
            "model_id": model_id,
            "total_simulation_runs": TOTAL_SIMULATION_RUNS,
            "iterations_per_run": ITERATIONS_PER_RUN,
            "experiment_timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "total_duration_seconds": total_time_taken
        },
        "overall_summary_statistics": { # Final averages across all runs
            "average_final_optimal_ratio": np.mean(collected_final_optimal_ratios) if collected_final_optimal_ratios else 0,
            "stdev_final_optimal_ratio": np.std(collected_final_optimal_ratios) if collected_final_optimal_ratios else 0,
            "average_total_reward_per_run": np.mean(collected_total_rewards) if collected_total_rewards else 0,
            "stdev_total_reward_per_run": np.std(collected_total_rewards) if collected_total_rewards else 0,
        },
        "averaged_learning_curves": { # Averaged data per iteration step
            "avg_cumulative_rewards_per_iteration": avg_cum_rewards_per_iteration,
            "std_cumulative_rewards_per_iteration": std_cum_rewards_per_iteration,
            "avg_cumulative_regrets_per_iteration": avg_cum_regrets_per_iteration,
            "std_cumulative_regrets_per_iteration": std_cum_regrets_per_iteration,
            "avg_optimal_arm_selection_ratio_per_iteration": avg_opt_arm_ratios_per_iteration,
            "std_optimal_arm_selection_ratio_per_iteration": std_opt_arm_ratios_per_iteration,
            "avg_prob_choice_arm2_per_iteration": avg_prob_choice_arm2_per_iteration,
            "std_prob_choice_arm2_per_iteration": std_prob_choice_arm2_per_iteration
        }
        # If you also need ALL raw data from every run in this file, you could add:
        # "all_individual_runs_raw_data": all_runs_detailed_data_collected_previously (but this would make the file huge)
    }

    results_output_dir = "simulation_results"
    os.makedirs(results_output_dir, exist_ok=True)
    master_filename = f"AVERAGED_bandit_results_{safe_model_name_for_filename}_{TOTAL_SIMULATION_RUNS}runs_{master_output_data['experiment_metadata']['experiment_timestamp']}.json"
    master_filepath = os.path.join(results_output_dir, master_filename)

    try:
        with open(master_filepath, "w") as f:
            json.dump(master_output_data, f, indent=2) 
        print(f"\nAveraged simulation data for {TOTAL_SIMULATION_RUNS} runs saved to: {master_filepath}")
    except Exception as e:
        print(f"Error saving averaged JSON file: {e}")

    print("\n\n--- Overall Experiment Summary (from final averages) ---")
    print(f"Average Final Optimal Arm Selection Ratio: {master_output_data['overall_summary_statistics']['average_final_optimal_ratio']:.4f} (StdDev: {master_output_data['overall_summary_statistics']['stdev_final_optimal_ratio']:.4f})")
    print(f"Average Total Reward per Run: {master_output_data['overall_summary_statistics']['average_total_reward_per_run']:.3f} (StdDev: {master_output_data['overall_summary_statistics']['stdev_total_reward_per_run']:.3f})")
