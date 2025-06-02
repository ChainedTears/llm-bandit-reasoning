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
import logging # Import the logging library

# --- Suppress Transformers Library Warnings ---
# Add this near the top, after imports
logging.getLogger("transformers").setLevel(logging.ERROR)
# You could use logging.CRITICAL for even fewer messages if ERROR isn't enough

# --- Hugging Face Login ---
# Ensure Hugging Face token is available if downloading new models.
# login(token="YOUR_HF_TOKEN_HERE") # Uncomment and replace if providing token directly

# --- User's original model dictionary ---
model_dict = {
    '1': "Qwen/Qwen3-4B",
    '2': "Qwen/Qwen3-8B",
    '3': "meta-llama/Llama-3.1-8B",
    '4': "deepseek-ai/DeepSeek-R1", # User selection prompt refers to this as "Mistral 7B"
    '5': "microsoft/phi-2",
    '6': "google/gemma-3-12b-it",
    '7': "openai/whisper-large-v3"  # Note: This is a speech-to-text model
}

print("Please select the model (using a number from 1-7):")
# User's original input prompt text for selection
print(" (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n (7) Whisper Large V3")
receive = input("Select here: ")

while receive not in model_dict:
    print("\nInvalid selection. Please select the model (using a number from 1-7):")
    print(" (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n (7) Whisper Large V3")
    receive = input("Select here: ")

model_id = model_dict[receive]
safe_model_name_for_filename = model_id.replace("/", "_")

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Tokenizer and Model Loading ---
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
        if device.type == "mps": dtype = torch.float16
        elif device.type == "cuda":
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported(): dtype = torch.bfloat16
            else: dtype = torch.float16
        print(f"Loading model {model_id} with dtype: {dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=CACHE_DIR,
            # trust_remote_code=True, # Uncomment if model requires
        ).to(device)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        print("This might be because the model is not a Causal LM (e.g., Whisper is for speech), or requires `trust_remote_code=True`.")
        sys.exit(1)

if tokenizer and tokenizer.pad_token_id is None:
    print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- Get AI Response Function (using original prompt style) ---
def get_response(prompt_text, current_tokenizer, current_model, current_device):
    if current_model is None or current_tokenizer is None: return "Model or tokenizer not loaded."
    if not hasattr(current_model, 'generate'):
        # This error message will be printed by run_simulation if this special value is returned
        return "MODEL_ERROR_NO_GENERATE" 

    max_new_tokens_for_original_prompt = 15 # For "Output: <number>" format
    model_max_len = getattr(current_tokenizer, 'model_max_length', 2048)
    if hasattr(current_model, 'config') and hasattr(current_model.config, 'max_position_embeddings'):
        model_max_len = current_model.config.max_position_embeddings
    input_max_len = model_max_len - max_new_tokens_for_original_prompt - 50 
    if input_max_len <= 0: input_max_len = min(512, model_max_len // 2)

    inputs = current_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=input_max_len).to(current_device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = current_model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens_for_original_prompt, 
            do_sample=True, # As per original script's implication, can be set to False for determinism
            temperature=0.1, # Low temperature for more focused sampling
            top_p=1.0,
            pad_token_id=current_tokenizer.pad_token_id, 
            eos_token_id=current_tokenizer.eos_token_id
        )
    return current_tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()

# --- Bandit Simulation (returns numerical reward) ---
def bandit_simulation(choice):
    r = secrets.randbelow(100)
    if choice == 1: return (1, "won") if r < 30 else (0, "lost")
    if choice == 2: return (1, "won") if r < 65 else (0, "lost")
    return 0, "error" # Should ideally not be reached

# --- Run Single Simulation (returns detailed lists) ---
def run_one_simulation(p_model_id, p_tokenizer, p_model, p_device, num_iterations=25, run_id_for_log=0):
    history_str = ""
    choices_this_run, rewards_this_run, regrets_this_run, optimal_arm_choices_this_run = [], [], [], []
    P_OPTIMAL_ARM_EXPECTED_REWARD, P_MACHINE_1_EXPECTED_REWARD, P_MACHINE_2_EXPECTED_REWARD = 0.65, 0.30, 0.65

    if run_id_for_log % 50 == 0 or run_id_for_log == 1:
        print(f"--- Running sub-run {run_id_for_log}/{TOTAL_SIMULATION_RUNS} for model {p_model_id} ---")

    for i in range(num_iterations):
        iteration_num = i + 1
        # User's original prompt structure
        prompt = f"""I am in a casino with two slot machines, 1 and 2.
I will output either a 1 or a 2, based on the history of my choices and results, which are:
{history_str}
I am in a casino with two slot machines, 1 and 2.
I will output either a 1 or a 2, based on the history of my choices and results, which are:
{history_str}
I will give my output in this format:
Output: <number>

Output:"""

        ai_choice, attempts = None, 0
        max_attempts = 3
        while attempts < max_attempts and ai_choice is None:
            ai_response_raw = get_response(prompt, p_tokenizer, p_model, p_device)
            if ai_response_raw == "MODEL_ERROR_NO_GENERATE":
                print(f"Run {run_id_for_log}, Iter {iteration_num}: Critical model error (no generate method). This run will be invalid.")
                return {"run_error": True, "num_iterations": num_iterations}

            if run_id_for_log % 100 == 0 and i % 5 == 0: # Very sparse logging
                 print(f"Run {run_id_for_log}, Iter {iteration_num}, Raw AI Resp: '{ai_response_raw}'")
            
            try: # Parsing for "Output: <number>"
                match = re.search(r'Output:\s*([12])\b', ai_response_raw)
                if match:
                    parsed_val = int(match.group(1))
                    if parsed_val in [1, 2]: ai_choice = parsed_val
            except Exception: pass # Stay None if parsing fails
            attempts += 1
        
        if ai_choice is None: ai_choice = secrets.choice([1, 2]) # Default to random

        numerical_reward, result_str = bandit_simulation(ai_choice)
        choices_this_run.append(ai_choice)
        rewards_this_run.append(numerical_reward)
        optimal_arm_choices_this_run.append(1 if ai_choice == 2 else 0)
        
        chosen_arm_expected_reward = P_MACHINE_1_EXPECTED_REWARD if ai_choice == 1 else P_MACHINE_2_EXPECTED_REWARD
        instantaneous_regret = P_OPTIMAL_ARM_EXPECTED_REWARD - chosen_arm_expected_reward
        regrets_this_run.append(instantaneous_regret)
        history_str += f"Slot Machine {ai_choice} {result_str}\n"

    cum_rewards = np.cumsum(rewards_this_run).tolist() if rewards_this_run else [0]*num_iterations
    cum_regrets = np.cumsum(regrets_this_run).tolist() if regrets_this_run else [0]*num_iterations
    cum_optimal_choices = np.cumsum(optimal_arm_choices_this_run).tolist() if optimal_arm_choices_this_run else [0]*num_iterations
    optimal_choice_ratio_over_time = [cum_optimal_choices[k]/(k+1) if (k+1) > 0 else 0 for k in range(len(cum_optimal_choices))] if cum_optimal_choices else [0]*num_iterations
    
    return {
        "choices_per_iteration": choices_this_run,
        "cumulative_rewards_per_iteration": cum_rewards,
        "cumulative_regrets_per_iteration": cum_regrets,
        "optimal_arm_selection_ratio_per_iteration": optimal_choice_ratio_over_time,
        "summary_final_optimal_ratio_this_run": optimal_choice_ratio_over_time[-1] if optimal_choice_ratio_over_time else 0,
        "summary_total_actual_reward_this_run": cum_rewards[-1] if cum_rewards else 0,
        "run_error": False, "num_iterations": num_iterations
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    if model is None or tokenizer is None:
        print("Model or tokenizer failed to load globally. Exiting.")
        sys.exit(1)

    TOTAL_SIMULATION_RUNS = 500
    ITERATIONS_PER_RUN = 25

    all_runs_cum_rewards_series, all_runs_cum_regrets_series = [], []
    all_runs_optimal_ratio_series, all_runs_prob_choosing_optimal_series = [], []
    collected_final_optimal_ratios, collected_total_rewards = [], []
    successful_run_count = 0

    experiment_start_time = time.time()
    print(f"Starting experiment: {TOTAL_SIMULATION_RUNS} runs of {ITERATIONS_PER_RUN} iterations each for model {model_id}...")

    for run_idx in range(1, TOTAL_SIMULATION_RUNS + 1):
        single_run_data = run_one_simulation(
            model_id, tokenizer, model, device, ITERATIONS_PER_RUN, run_idx
        )

        if single_run_data.get("run_error"):
            print(f"Run {run_idx} for model {model_id} encountered a critical error and was excluded from averages.")
            continue 

        successful_run_count += 1
        all_runs_cum_rewards_series.append(single_run_data["cumulative_rewards_per_iteration"])
        all_runs_cum_regrets_series.append(single_run_data["cumulative_regrets_per_iteration"])
        all_runs_optimal_ratio_series.append(single_run_data["optimal_arm_selection_ratio_per_iteration"])
        all_runs_prob_choosing_optimal_series.append([1 if choice == 2 else 0 for choice in single_run_data["choices_per_iteration"]])
        collected_final_optimal_ratios.append(single_run_data["summary_final_optimal_ratio_this_run"])
        collected_total_rewards.append(single_run_data["summary_total_actual_reward_this_run"])

        if run_idx % 20 == 0:
            elapsed = time.time() - experiment_start_time
            runs_left_count = TOTAL_SIMULATION_RUNS - run_idx
            avg_time_per_run_val = elapsed / run_idx if run_idx > 0 else 0
            eta_seconds = avg_time_per_run_val * runs_left_count
            print(f"PROGRESS: Completed {run_idx}/{TOTAL_SIMULATION_RUNS} runs. ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}")

    total_experiment_time = time.time() - experiment_start_time
    print(f"\nAll {TOTAL_SIMULATION_RUNS} simulation runs attempted in {time.strftime('%H:%M:%S', time.gmtime(total_experiment_time))}.")
    print(f"Number of successful runs processed for averaging: {successful_run_count}")

    if successful_run_count == 0:
        print("No successful runs to average or save. Exiting.")
        sys.exit(1)

    avg_cum_rewards = np.mean(np.array(all_runs_cum_rewards_series), axis=0).tolist()
    std_cum_rewards = np.std(np.array(all_runs_cum_rewards_series), axis=0).tolist()
    avg_cum_regrets = np.mean(np.array(all_runs_cum_regrets_series), axis=0).tolist()
    std_cum_regrets = np.std(np.array(all_runs_cum_regrets_series), axis=0).tolist()
    avg_optimal_ratio = np.mean(np.array(all_runs_optimal_ratio_series), axis=0).tolist()
    std_optimal_ratio = np.std(np.array(all_runs_optimal_ratio_series), axis=0).tolist()
    avg_prob_choosing_optimal = np.mean(np.array(all_runs_prob_choosing_optimal_series), axis=0).tolist()
    std_prob_choosing_optimal = np.std(np.array(all_runs_prob_choosing_optimal_series), axis=0).tolist()

    output_data = {
        "experiment_metadata": {
            "model_id": model_id, "total_runs_attempted": TOTAL_SIMULATION_RUNS, 
            "successful_runs_averaged": successful_run_count, "iterations_per_run": ITERATIONS_PER_RUN, 
            "experiment_timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "prompt_style": "User Original (Output: <number>)" # Documenting the prompt style used
        },
        "overall_summary_stats_from_successful_runs": {
            "avg_final_optimal_ratio": np.mean(collected_final_optimal_ratios) if collected_final_optimal_ratios else 0,
            "std_final_optimal_ratio": np.std(collected_final_optimal_ratios) if collected_final_optimal_ratios else 0,
            "avg_total_reward": np.mean(collected_total_rewards) if collected_total_rewards else 0,
            "std_total_reward": np.std(collected_total_rewards) if collected_total_rewards else 0,
        },
        "averaged_learning_curves": {
            "iterations": list(range(1, ITERATIONS_PER_RUN + 1)),
            "avg_cumulative_reward_per_iteration": avg_cum_rewards,
            "std_cumulative_reward_per_iteration": std_cum_rewards,
            "avg_cumulative_regret_per_iteration": avg_cum_regrets,
            "std_cumulative_regret_per_iteration": std_cum_regrets,
            "avg_optimal_arm_selection_ratio_per_iteration": avg_optimal_ratio,
            "std_optimal_arm_selection_ratio_per_iteration": std_optimal_ratio,
            "avg_prob_choosing_optimal_arm_per_iteration": avg_prob_choosing_optimal,
            "std_prob_choosing_optimal_arm_per_iteration": std_prob_choosing_optimal
        }
    }

    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp_for_file = output_data["experiment_metadata"]["experiment_timestamp"]
    filename = f"AVERAGED_RESULTS_{safe_model_name_for_filename}_{TOTAL_SIMULATION_RUNS}runs_OriginalPrompt_{timestamp_for_file}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, "w") as f: json.dump(output_data, f, indent=2)
        print(f"\nAveraged results for {successful_run_count} successful runs saved to: {filepath}")
    except Exception as e: print(f"Error saving averaged JSON results: {e}")

    print("\n--- Overall Summary (based on successful runs) ---")
    summary = output_data["overall_summary_stats_from_successful_runs"]
    print(f"Avg. Final Optimal Ratio: {summary['avg_final_optimal_ratio']:.4f} (StdDev: {summary['std_final_optimal_ratio']:.4f})")
    print(f"Avg. Total Reward: {summary['avg_total_reward']:.3f} (StdDev: {summary['std_total_reward']:.3f})")
