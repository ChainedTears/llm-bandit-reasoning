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
    '4': "deepseek-ai/DeepSeek-R1", # User selection prompt refers to this as "Mistral 7B"
    '5': "microsoft/phi-2",
    '6': "google/gemma-3-12b-it",
    '7': "openai/whisper-large-v3"  # This is a speech-to-text model
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
            # trust_remote_code=True, # May be needed for some models
        ).to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        print("This might be because the model is not a Causal LM (e.g., Whisper is for speech), or requires `trust_remote_code=True`.")
        sys.exit(1)

if tokenizer and tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def get_response(prompt_text, current_tokenizer, current_model, current_device):
    if current_model is None or current_tokenizer is None: return "Model or tokenizer not loaded."
    if not hasattr(current_model, 'generate'):
        print(f"FATAL: Model {model_id} selected is not a text generation model (e.g. Whisper). It lacks a 'generate' method.")
        return "MODEL_ERROR_NO_GENERATE" # Signal critical error

    # For "Output: <number>" format, allow a bit more tokens
    max_new_tokens_for_original_prompt = 15
    max_prompt_len = getattr(current_tokenizer, 'model_max_length', 2048) - max_new_tokens_for_original_prompt - 50
    if max_prompt_len <= 0: max_prompt_len = 512


    inputs = current_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = current_model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens_for_original_prompt, # Adjusted for "Output: <number>"
            do_sample=True, # Your original script implied sampling might be used
            temperature=0.1, # Low temperature for more deterministic sampling
            top_p=1.0,
            pad_token_id=current_tokenizer.pad_token_id, 
            eos_token_id=current_tokenizer.eos_token_id
        )
    return current_tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()

def bandit_simulation(choice):
    r = secrets.randbelow(100)
    if choice == 1: return (1, "won") if r < 30 else (0, "lost")
    if choice == 2: return (1, "won") if r < 65 else (0, "lost")
    return 0, "error"

def run_simulation(current_model_id, current_tokenizer, current_model, current_device, num_iterations=25, run_id=1):
    history_str = ""
    choices, rewards, regrets, optimal_selections = [], [], [], []
    P_OPT, P_M1, P_M2 = 0.65, 0.30, 0.65

    if run_id % 50 == 0 or run_id == 1:
        print(f"\n--- Starting Simulation Run {run_id}/{TOTAL_SIMULATION_RUNS} ---")

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
        while attempts < 3 and ai_choice is None:
            raw_resp = get_response(prompt, current_tokenizer, current_model, current_device)
            if raw_resp == "MODEL_ERROR_NO_GENERATE": # Critical model error
                # This run cannot proceed with this model. Return empty/failure indicators.
                print(f"Run {run_id}, Iter {iteration_num}: Unrecoverable model error. Aborting this run.")
                return { # Return structure indicating failure for this run
                    "choices_per_iteration": [0]*num_iterations, "cumulative_rewards_per_iteration": [0]*num_iterations,
                    "cumulative_regrets_per_iteration": [P_OPT*k for k in range(1,num_iterations+1)], # Max regret
                    "optimal_arm_selection_ratio_per_iteration": [0]*num_iterations,
                    "summary_final_optimal_ratio_this_run": 0, "summary_total_actual_reward_this_run": 0,
                    "run_error": True
                }

            if run_id % 100 == 0 and i % 5 == 0 : # Less verbose logging
                 print(f"Run {run_id}, Iter {iteration_num}, Raw AI Resp: '{raw_resp}'")
            
            try:
                match = re.search(r'Output:\s*([12])\b', raw_resp) # Expects "Output: 1" or "Output: 2"
                if match:
                    parsed_val = int(match.group(1))
                    if parsed_val in [1, 2]:
                        ai_choice = parsed_val
            except Exception:
                pass # Parsing failed, ai_choice remains None
            attempts +=1

        if ai_choice is None: ai_choice = secrets.choice([1,2]) # Default to random if parsing fails
        
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
    
    return {
        "choices_per_iteration": choices,
        "cumulative_rewards_per_iteration": cum_rewards,
        "cumulative_regrets_per_iteration": cum_regrets,
        "optimal_arm_selection_ratio_per_iteration": opt_sel_ratio,
        "summary_final_optimal_ratio_this_run": opt_sel_ratio[-1] if opt_sel_ratio else 0,
        "summary_total_actual_reward_this_run": cum_rewards[-1] if cum_rewards else 0,
        "run_error": False
    }

if __name__ == "__main__":
    if model is None or tokenizer is None: sys.exit(1)
        
    TOTAL_SIMULATION_RUNS = 500 
    ITERATIONS_PER_RUN = 25     
    
    all_runs_cum_rewards, all_runs_cum_regrets, all_runs_opt_arm_ratios, all_runs_choices_arm2_prob = [], [], [], []
    collected_final_optimal_ratios, collected_total_rewards, successful_runs = [], [], 0

    experiment_start_time = time.time()
    print(f"Starting experiment: {TOTAL_SIMULATION_RUNS} runs of {ITERATIONS_PER_RUN} iterations each for model {model_id}...")

    for run_num in range(1, TOTAL_SIMULATION_RUNS + 1):
        run_data = run_simulation(
            model_id, tokenizer, model, device, ITERATIONS_PER_RUN, run_num
        )
        
        if run_data.get("run_error"):
            print(f"Run {run_num} for model {model_id} failed and was skipped in averages.")
            # For errored runs, we still need to append lists of the correct length (ITERATIONS_PER_RUN)
            # filled with a value that indicates failure (e.g., NaN, or a placeholder if using mean later).
            # For simplicity, we'll append lists of zeros or max regret,
            # but ideally, these runs might be excluded from np.mean if too many fail.
            # Or, ensure run_simulation always returns lists of correct length.
            all_runs_cum_rewards.append([0] * ITERATIONS_PER_RUN) # Placeholder for error
            all_runs_cum_regrets.append([P_OPT * k for k in range(1,ITERATIONS_PER_RUN+1)]) # Max regret
            all_runs_opt_arm_ratios.append([0] * ITERATIONS_PER_RUN)
            all_runs_choices_arm2_prob.append([0] * ITERATIONS_PER_RUN)
            collected_final_optimal_ratios.append(0)
            collected_total_rewards.append(0)
            continue # Skip to next run

        successful_runs += 1
        all_runs_cum_rewards.append(run_data["cumulative_rewards_per_iteration"])
        all_runs_cum_regrets.append(run_data["cumulative_regrets_per_iteration"])
        all_runs_opt_arm_ratios.append(run_data["optimal_arm_selection_ratio_per_iteration"])
        all_runs_choices_arm2_prob.append([1 if choice == 2 else 0 for choice in run_data["choices_per_iteration"]])
        collected_final_optimal_ratios.append(run_data["summary_final_optimal_ratio_this_run"])
        collected_total_rewards.append(run_data["summary_total_actual_reward_this_run"])
        
        if run_num % 20 == 0:
            elapsed = time.time() - experiment_start_time; runs_left = TOTAL_SIMULATION_RUNS - run_num
            avg_time = elapsed / run_num if run_num > 0 else 0; eta = avg_time * runs_left
            print(f"PROGRESS: Completed {run_num}/{TOTAL_SIMULATION_RUNS} runs. ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")
    
    total_time_taken = time.time() - experiment_start_time
    print(f"\nAll {TOTAL_SIMULATION_RUNS} simulation runs attempted in {time.strftime('%H:%M:%S', time.gmtime(total_time_taken))}. Successful runs: {successful_runs}")

    if successful_runs < TOTAL_SIMULATION_RUNS:
        print(f"Warning: {TOTAL_SIMULATION_RUNS - successful_runs} runs encountered errors and were excluded or used placeholder data for averages.")
    
    # Averaging based on potentially fewer than TOTAL_SIMULATION_RUNS if there were errors
    # We filter out errored runs before averaging if we don't want placeholders to skew results
    # For now, the placeholders will be included in the average. A more robust solution
    # would be to filter the all_runs_* lists before np.array and np.mean.
    # However, if run_simulation consistently returns lists of the correct length (even if zeros for error),
    # np.mean will work but might be skewed if too many errors.

    avg_cum_rewards_per_iteration = np.mean(np.array(all_runs_cum_rewards), axis=0).tolist() if successful_runs > 0 else [0]*ITERATIONS_PER_RUN
    std_cum_rewards_per_iteration = np.std(np.array(all_runs_cum_rewards), axis=0).tolist() if successful_runs > 0 else [0]*ITERATIONS_PER_RUN
    avg_cum_regrets_per_iteration = np.mean(np.array(all_runs_cum_regrets), axis=0).tolist() if successful_runs > 0 else [0]*ITERATIONS_PER_RUN
    std_cum_regrets_per_iteration = np.std(np.array(all_runs_cum_regrets), axis=0).tolist() if successful_runs > 0 else [0]*ITERATIONS_PER_RUN
    avg_opt_arm_ratios_per_iteration = np.mean(np.array(all_runs_opt_arm_ratios), axis=0).tolist() if successful_runs > 0 else [0]*ITERATIONS_PER_RUN
    std_opt_arm_ratios_per_iteration = np.std(np.array(all_runs_opt_arm_ratios), axis=0).tolist() if successful_runs > 0 else [0]*ITERATIONS_PER_RUN
    avg_prob_choice_arm2_per_iteration = np.mean(np.array(all_runs_choices_arm2_prob), axis=0).tolist() if successful_runs > 0 else [0]*ITERATIONS_PER_RUN
    std_prob_choice_arm2_per_iteration = np.std(np.array(all_runs_choices_arm2_prob), axis=0).tolist() if successful_runs > 0 else [0]*ITERATIONS_PER_RUN

    master_output_data = {
        "experiment_metadata": {
            "model_id": model_id, "total_simulation_runs_attempted": TOTAL_SIMULATION_RUNS, "successful_runs": successful_runs,
            "iterations_per_run": ITERATIONS_PER_RUN, "experiment_timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "total_duration_seconds": total_time_taken, "prompt_style": "Original User Prompt (Output: <number>)"
        },
        "overall_summary_statistics": {
            "average_final_optimal_ratio": np.mean(collected_final_optimal_ratios) if successful_runs > 0 else 0,
            "stdev_final_optimal_ratio": np.std(collected_final_optimal_ratios) if successful_runs > 0 else 0,
            "average_total_reward_per_run": np.mean(collected_total_rewards) if successful_runs > 0 else 0,
            "stdev_total_reward_per_run": np.std(collected_total_rewards) if successful_runs > 0 else 0,
        },
        "averaged_learning_curves": {
            "avg_cumulative_rewards_per_iteration": avg_cum_rewards_per_iteration,
            "std_cumulative_rewards_per_iteration": std_cum_rewards_per_iteration,
            "avg_cumulative_regrets_per_iteration": avg_cum_regrets_per_iteration,
            "std_cumulative_regrets_per_iteration": std_cum_regrets_per_iteration,
            "avg_optimal_arm_selection_ratio_per_iteration": avg_opt_arm_ratios_per_iteration,
            "std_optimal_arm_selection_ratio_per_iteration": std_opt_arm_ratios_per_iteration,
            "avg_prob_choice_arm2_per_iteration": avg_prob_choice_arm2_per_iteration,
            "std_prob_choice_arm2_per_iteration": std_prob_choice_arm2_per_iteration
        }}
    results_output_dir = "simulation_results"; os.makedirs(results_output_dir, exist_ok=True)
    master_filename = f"AVERAGED_bandit_results_{safe_model_name_for_filename}_{TOTAL_SIMULATION_RUNS}runs-OriginalPrompt_{master_output_data['experiment_metadata']['experiment_timestamp']}.json"
    master_filepath = os.path.join(results_output_dir, master_filename)
    try:
        with open(master_filepath, "w") as f: json.dump(master_output_data, f, indent=2) 
        print(f"\nAveraged simulation data for {successful_runs}/{TOTAL_SIMULATION_RUNS} runs saved to: {master_filepath}")
    except Exception as e: print(f"Error saving averaged JSON file: {e}")

    print("\n\n--- Overall Experiment Summary (from final averages) ---") # Based on successful runs
    if successful_runs > 0:
        print(f"Average Final Optimal Arm Selection Ratio: {master_output_data['overall_summary_statistics']['average_final_optimal_ratio']:.4f} (StdDev: {master_output_data['overall_summary_statistics']['stdev_final_optimal_ratio']:.4f})")
        print(f"Average Total Reward per Run: {master_output_data['overall_summary_statistics']['average_total_reward_per_run']:.3f} (StdDev: {master_output_data['overall_summary_statistics']['stdev_total_reward_per_run']:.3f})")
    else:
        print("No successful runs to average.")
