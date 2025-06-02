import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
from huggingface_hub import login
import re
import numpy as np # For averaging
import os # For path operations and makedirs
import sys # For sys.exit

# login(token="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") # Assuming you handle login if needed

model_dict = {
    '1': "Qwen/Qwen3-4B",
    '2': "Qwen/Qwen3-8B",
    '3': "meta-llama/Llama-3.1-8B",
    '4': "deepseek-ai/DeepSeek-R1", # Note: Your input prompt lists "Mistral 7B" for option 4
    '5': "microsoft/phi-2",
    '6': "google/gemma-3-12b-it",
    '7': "openai/whisper-large-v3"  # Note: Whisper is a speech-to-text model
}

receive = input("Please select the model (using a number from 1-7): \n (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n (7) Whisper Large V3 \n Select here: ")
while receive not in model_dict:
    receive = input("Invalid selection. Please select model: \n (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n (7) Whisper Large V3 \n Select here: ")
model_id = model_dict[receive]
safe_model_name_for_filename = model_id.replace("/", "_") # For creating safe filenames

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

tokenizer = None
model = None
# Using a sub-directory for cache is generally cleaner
CACHE_DIR = "./model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

try:
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
        print(f"Loading model {model_id} with dtype: {dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=CACHE_DIR,
            # trust_remote_code=True, # Uncomment if model requires
        ).to(device)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        print("This could be due to an incorrect model ID, the model not being a CausalLM (e.g., Whisper), or needing `trust_remote_code=True`.")
        sys.exit(1)

if tokenizer and tokenizer.pad_token_id is None:
    print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Using the get_response from your provided code, with do_sample=False for determinism
def get_response_from_original(prompt_text_param, current_tokenizer, current_model, current_device):
    if current_model is None or current_tokenizer is None: return "Model or tokenizer not loaded."
    if not hasattr(current_model, 'generate'):
        print(f"FATAL: Model {model_id} selected is not a text generation model. It lacks a 'generate' method.")
        return "MODEL_ERROR_NO_GENERATE"

    # Using a slightly more robust max_length calculation
    max_new_tokens_val = 10 # Allow a bit more for potential verbosity with some models/prompts
    # Max length for tokenizer should be model's max length minus generation tokens and some buffer
    # Using model.config.max_position_embeddings or tokenizer.model_max_length
    model_max_len = getattr(current_tokenizer, 'model_max_length', 2048) # Default if not found
    if hasattr(current_model, 'config') and hasattr(current_model.config, 'max_position_embeddings'):
        model_max_len = current_model.config.max_position_embeddings

    input_max_len = model_max_len - max_new_tokens_val - 50 # 50 as buffer
    if input_max_len <= 0: input_max_len = min(512, model_max_len // 2)


    inputs = current_tokenizer(prompt_text_param, return_tensors="pt", truncation=True, max_length=input_max_len).to(current_device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = current_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens_val,
            do_sample=False, # Changed from your original True for more deterministic AI choices
            # temperature=0.7, # Not used if do_sample=False
            # top_p=1.0,       # Not used if do_sample=False
            pad_token_id=current_tokenizer.pad_token_id,
            eos_token_id=current_tokenizer.eos_token_id
        )
    return current_tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()

# Your second bandit_simulation definition, modified to return numerical reward
def bandit_simulation_numerical(choice):
    random_number = secrets.randbelow(100)
    if choice == 1: # 30% win rate
        return 1 if random_number < 30 else 0, "won" if random_number < 30 else "lost"
    if choice == 2: # 65% win rate (objectively better)
        return 1 if random_number < 65 else 0, "won" if random_number < 65 else "lost"
    print(f"Error in bandit_simulation_numerical with choice: {choice}")
    return 0, "error"


# This function will run one 25-iteration simulation and return detailed lists
def run_one_simulation(p_model_id, p_tokenizer, p_model, p_device, num_iterations=25, run_id_for_log=0):
    previous_outputs = ""
    
    # Data for this single run
    choices_this_run = []
    rewards_this_run = [] # Numerical rewards
    regrets_this_run = []
    optimal_arm_choices_this_run = [] # 1 if optimal (machine 2) was chosen, 0 otherwise

    P_OPTIMAL_ARM_EXPECTED_REWARD = 0.65
    P_MACHINE_1_EXPECTED_REWARD = 0.30
    P_MACHINE_2_EXPECTED_REWARD = 0.65

    if run_id_for_log % 50 == 0 or run_id_for_log == 1: # Reduce console output for many runs
        print(f"--- Starting sub-run {run_id_for_log} for model {p_model_id} ---")

    for i in range(num_iterations):
        iteration_num = i + 1
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
{previous_outputs}Your choice (1 or 2):"""

        ai_choice = None
        attempts = 0
        max_attempts = 3 # Max attempts to parse AI response

        while attempts < max_attempts and ai_choice is None:
            ai_response_raw = get_response_from_original(prompt, p_tokenizer, p_model, p_device)
            if ai_response_raw == "MODEL_ERROR_NO_GENERATE":
                print(f"Run {run_id_for_log}, Iter {iteration_num}: Critical model error. This run will be invalid.")
                return {"run_error": True, "num_iterations": num_iterations} # Signal error for this run

            if run_id_for_log % 100 == 0 and i % 5 == 0: # Very sparse logging of raw responses
                 print(f"Run {run_id_for_log}, Iter {iteration_num}, Raw AI Resp: '{ai_response_raw}'")

            match = re.match(r'^\s*([12])\b', ai_response_raw)
            if not match: # Fallback if no direct 1 or 2 found at the start
                match = re.search(r'\b([12])\b', ai_response_raw)
            
            if match:
                try:
                    ai_choice = int(match.group(1))
                    if ai_choice not in [1, 2]: ai_choice = None # Invalid choice
                except ValueError:
                    ai_choice = None # Parsing failed
            attempts += 1
        
        if ai_choice is None: # If still None after attempts
            if run_id_for_log % 50 == 0 or run_id_for_log == 1:
                print(f"Run {run_id_for_log}, Iter {iteration_num}: AI did not output clear 1 or 2. Using random choice.")
            ai_choice = secrets.choice([1, 2])

        numerical_reward, result_str = bandit_simulation_numerical(ai_choice)
        
        choices_this_run.append(ai_choice)
        rewards_this_run.append(numerical_reward)
        optimal_arm_choices_this_run.append(1 if ai_choice == 2 else 0)
        
        chosen_arm_expected_reward = P_MACHINE_1_EXPECTED_REWARD if ai_choice == 1 else P_MACHINE_2_EXPECTED_REWARD
        instantaneous_regret = P_OPTIMAL_ARM_EXPECTED_REWARD - chosen_arm_expected_reward
        regrets_this_run.append(instantaneous_regret)

        previous_outputs += f"Slot Machine {ai_choice} {result_str}\n"

    # Calculate per-iteration cumulative stats for this run
    cum_rewards = np.cumsum(rewards_this_run).tolist() if rewards_this_run else [0]*num_iterations
    cum_regrets = np.cumsum(regrets_this_run).tolist() if regrets_this_run else [0]*num_iterations
    cum_optimal_choices = np.cumsum(optimal_arm_choices_this_run).tolist() if optimal_arm_choices_this_run else [0]*num_iterations
    optimal_choice_ratio_over_time = [cum_optimal_choices[k]/(k+1) if (k+1) > 0 else 0 for k in range(len(cum_optimal_choices))] if cum_optimal_choices else [0]*num_iterations
    
    return {
        "choices_per_iteration": choices_this_run, # Raw choices
        "cumulative_rewards_per_iteration": cum_rewards,
        "cumulative_regrets_per_iteration": cum_regrets,
        "optimal_arm_selection_ratio_per_iteration": optimal_choice_ratio_over_time,
        "summary_final_optimal_ratio_this_run": optimal_choice_ratio_over_time[-1] if optimal_choice_ratio_over_time else 0,
        "summary_total_actual_reward_this_run": cum_rewards[-1] if cum_rewards else 0,
        "run_error": False,
        "num_iterations": num_iterations
    }


if __name__ == "__main__":
    if model is None or tokenizer is None: # Ensure model loaded before starting
        print("Model or tokenizer failed to load globally. Exiting.")
        sys.exit(1)

    TOTAL_SIMULATION_RUNS = 500 # Number of full 25-iteration simulations
    ITERATIONS_PER_RUN = 25

    # Lists to store the per-iteration series from ALL runs
    # Each element in these lists will be another list of length ITERATIONS_PER_RUN
    all_runs_cum_rewards_series = []
    all_runs_cum_regrets_series = []
    all_runs_optimal_ratio_series = []
    all_runs_prob_choosing_optimal_series = [] # For P(choose arm 2) at each step

    # For overall summary stats (averages of final values from each run)
    collected_final_optimal_ratios = []
    collected_total_rewards = []
    successful_run_count = 0

    experiment_start_time = time.time()
    print(f"Starting experiment: {TOTAL_SIMULATION_RUNS} runs of {ITERATIONS_PER_RUN} iterations each for model {model_id}...")

    for run_idx in range(1, TOTAL_SIMULATION_RUNS + 1):
        single_run_data = run_one_simulation(
            model_id, tokenizer, model, device,
            num_iterations=ITERATIONS_PER_RUN,
            run_id_for_log=run_idx
        )

        if single_run_data.get("run_error"):
            print(f"Run {run_idx} for model {model_id} failed and will be excluded from averages.")
            # To keep array dimensions consistent for averaging, we could append NaNs or skip.
            # For simplicity here, we skip appending if a run has a critical error.
            # Or, ensure `run_one_simulation` returns lists of correct length even on error.
            # The current `run_one_simulation` returns a dict with "run_error": True.
            # We will only process data from successful runs for averaging the curves.
            continue # Skip this run's data for averaging curves

        successful_run_count += 1
        all_runs_cum_rewards_series.append(single_run_data["cumulative_rewards_per_iteration"])
        all_runs_cum_regrets_series.append(single_run_data["cumulative_regrets_per_iteration"])
        all_runs_optimal_ratio_series.append(single_run_data["optimal_arm_selection_ratio_per_iteration"])
        
        prob_optimal_choice_this_run = [1 if choice == 2 else 0 for choice in single_run_data["choices_per_iteration"]]
        all_runs_prob_choosing_optimal_series.append(prob_optimal_choice_this_run)

        collected_final_optimal_ratios.append(single_run_data["summary_final_optimal_ratio_this_run"])
        collected_total_rewards.append(single_run_data["summary_total_actual_reward_this_run"])

        if run_idx % 20 == 0: # Progress update
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

    # --- Averaging the per-iteration data across all successful runs ---
    avg_cum_rewards = np.mean(np.array(all_runs_cum_rewards_series), axis=0).tolist()
    std_cum_rewards = np.std(np.array(all_runs_cum_rewards_series), axis=0).tolist()
    avg_cum_regrets = np.mean(np.array(all_runs_cum_regrets_series), axis=0).tolist()
    std_cum_regrets = np.std(np.array(all_runs_cum_regrets_series), axis=0).tolist()
    avg_optimal_ratio = np.mean(np.array(all_runs_optimal_ratio_series), axis=0).tolist()
    std_optimal_ratio = np.std(np.array(all_runs_optimal_ratio_series), axis=0).tolist()
    avg_prob_choosing_optimal = np.mean(np.array(all_runs_prob_choosing_optimal_series), axis=0).tolist()
    std_prob_choosing_optimal = np.std(np.array(all_runs_prob_choosing_optimal_series), axis=0).tolist()

    # --- Prepare the single large JSON output ---
    output_data = {
        "experiment_metadata": {
            "model_id": model_id,
            "total_runs_attempted": TOTAL_SIMULATION_RUNS,
            "successful_runs_averaged": successful_run_count,
            "iterations_per_run": ITERATIONS_PER_RUN,
            "experiment_timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "prompt_style": "User Original (few-shot, direct '1' or '2')" # Clarify prompt used
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
    filename = f"AVERAGED_RESULTS_{safe_model_name_for_filename}_{TOTAL_SIMULATION_RUNS}runs_{timestamp_for_file}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2) # Using indent=2 for readability
        print(f"\nAveraged results for {successful_run_count} successful runs saved to: {filepath}")
    except Exception as e:
        print(f"Error saving averaged JSON results: {e}")

    # Print overall summary
    print("\n--- Overall Summary (based on successful runs) ---")
    summary = output_data["overall_summary_stats_from_successful_runs"]
    print(f"Avg. Final Optimal Ratio: {summary['avg_final_optimal_ratio']:.4f} (StdDev: {summary['std_final_optimal_ratio']:.4f})")
    print(f"Avg. Total Reward: {summary['avg_total_reward']:.3f} (StdDev: {summary['std_total_reward']:.3f})")
