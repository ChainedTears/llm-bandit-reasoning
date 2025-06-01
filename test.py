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
    
    # Increased max_new_tokens slightly as the old prompt format "Output: <number>" is longer
    max_new_tokens_for_old_prompt = 10 
    max_prompt_len = getattr(current_tokenizer, 'model_max_length', 2048) - max_new_tokens_for_old_prompt - 50 # Reserve space
    if max_prompt_len <=0 : max_prompt_len = 512 # A fallback if model_max_length is too small

    inputs = current_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(current_device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = current_model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens_for_old_prompt, 
            do_sample=False, # Kept do_sample=False, adjust if needed for old prompt style
            temperature=0.1, # Low temp if do_sample=True, but ignored if False
            top_p=1.0,       # Ignored if do_sample=False
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
    history_str = "" # Renamed from your original 'previous_outputs' for consistency
    choices, rewards, regrets, optimal_selections = [], [], [], []
    P_OPT, P_M1, P_M2 = 0.65, 0.30, 0.65

    if run_id % 20 == 0 or run_id == 1 or num_iterations < 50:
        print(f"\n--- Running Simulation: Run {run_id}, Model {current_model_id}, Iterations {num_iterations} ---")

    for i in range(num_iterations):
        iteration_num = i + 1
        # Your original prompt structure
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
        while attempts < 3 and ai_choice is None: # Max 3 parsing attempts
            raw_resp = get_response(prompt, current_tokenizer, current_model, current_device)
            if raw_resp == "MODEL_ERROR_NO_GENERATE": 
                ai_choice = secrets.choice([1,2]); 
                if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: print(f"Run {run_id}, Iter {iteration_num}: Model error, random choice: {ai_choice}"); 
                break
            
            if run_id % 20 == 0 or run_id == 1 or num_iterations < 50:
                 print(f"Run {run_id}, Iter {iteration_num}, Raw AI Resp: '{raw_resp}'") 
            
            # Your original parsing logic, made more robust
            try:
                match = re.search(r'Output:\s*([12])\b', raw_resp) # Look for "Output: 1" or "Output: 2"
                if match:
                    parsed_val = int(match.group(1))
                    if parsed_val in [1, 2]:
                        ai_choice = parsed_val
                    else:
                        if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: print(f"Parsed value {parsed_val} not 1 or 2.")
                else: # No "Output: <number>" pattern found
                    if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: print("AI response did not match 'Output: <number>' format.")
            except Exception as e: # Catch any error during parsing (e.g., if re.search is None)
                if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: print(f"Error parsing AI response: {e}")
            
            attempts +=1

        if ai_choice is None: 
            ai_choice = secrets.choice([1,2]); 
            if run_id % 20 == 0 or run_id == 1 or num_iterations < 50: print(f"Run {run_id}, Iter {iteration_num}: AI failed to provide valid choice, using random: {ai_choice}")
        
        reward_val, outcome = bandit_simulation(ai_choice)
        if run_id % 20 == 0 or run_id == 1 or num_iterations < 50:
            print(f"AI chose {ai_choice}, Outcome: {outcome}, Reward: {reward_val}")
        
        chosen_exp_reward = P_M1 if ai_choice == 1 else P_M2
        regret = P_OPT - chosen_exp_reward
        
        choices.append(ai_choice); rewards.append(reward_val); regrets.append(regret)
        optimal_selections.append(1 if ai_choice == 2 else 0)
        history_str += f"Slot Machine {ai_choice} {outcome}\n" # Use game outcome for history

    cum_rewards = np.cumsum(rewards).tolist() if rewards else []
    cum_regrets = np.cumsum(regrets).tolist() if regrets else []
    cum_opt_sel = np.cumsum(optimal_selections).tolist() if optimal_selections else []
    opt_sel_ratio = [cum_opt_sel[k]/(k+1) for k in range(len(cum_opt_sel))] if cum_opt_sel else []

    final_optimal_ratio = opt_sel_ratio[-1] if opt_sel_ratio else 0
    total_actual_reward = cum_rewards[-1] if cum_rewards else 0
    
    run_data = {
        "run_id": run_id,
        "num_iterations_in_run": num_iterations,
        "choices_per_iteration": choices,
        "rewards_obtained_per_iteration": rewards,
        "instantaneous_regrets_per_iteration": regrets,
        "cumulative_rewards_per_iteration": cum_rewards,
        "cumulative_regrets_per_iteration": cum_regrets,
        "optimal_arm_selection_ratio_per_iteration": opt_sel_ratio,
        "summary_final_optimal_ratio_this_run": final_optimal_ratio,
        "summary_total_actual_reward_this_run": total_actual_reward
    }
    
    if run_id % 20 == 0 or run_id == 1 or num_iterations < 50:
        print(f"Run {run_id} Completed. Final Optimal Ratio: {final_optimal_ratio:.2f}, Total Reward: {total_actual_reward}")
    
    return run_data

if __name__ == "__main__":
    if model is None or tokenizer is None:
        print("Model/tokenizer failed to load. Exiting.")
        sys.exit(1)
        
    TOTAL_SIMULATION_RUNS = 500
    ITERATIONS_PER_RUN = 25    
    all_runs_detailed_data = []
    collected_final_optimal_ratios = []
    collected_total_rewards = []
    experiment_start_time = time.time()
    print(f"Starting experiment with {TOTAL_SIMULATION_RUNS} runs of {ITERATIONS_PER_RUN} iterations each...")

    for run_num in range(1, TOTAL_SIMULATION_RUNS + 1):
        single_run_data = run_simulation(
            model_id, tokenizer, model, device,
            num_iterations=ITERATIONS_PER_RUN, run_id=run_num
        )
        all_runs_detailed_data.append(single_run_data)
        collected_final_optimal_ratios.append(single_run_data["summary_final_optimal_ratio_this_run"])
        collected_total_rewards.append(single_run_data["summary_total_actual_reward_this_run"])
        if TOTAL_SIMULATION_RUNS > 1 and run_num < TOTAL_SIMULATION_RUNS and run_num % 20 == 0:
            elapsed = time.time() - experiment_start_time; runs_left = TOTAL_SIMULATION_RUNS - run_num
            avg_time = elapsed / run_num; eta = avg_time * runs_left
            print(f"PROGRESS: Completed {run_num}/{TOTAL_SIMULATION_RUNS} runs. ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")
    
    total_time_taken = time.time() - experiment_start_time
    print(f"\nAll {TOTAL_SIMULATION_RUNS} simulation runs completed in {time.strftime('%H:%M:%S', time.gmtime(total_time_taken))}.")

    master_output_data = {
        "experiment_metadata": {
            "model_id": model_id, "total_simulation_runs": TOTAL_SIMULATION_RUNS,
            "iterations_per_run": ITERATIONS_PER_RUN, "experiment_timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "total_duration_seconds": total_time_taken
        },
        "overall_summary_statistics": {
            "average_final_optimal_ratio": np.mean(collected_final_optimal_ratios) if collected_final_optimal_ratios else 0,
            "stdev_final_optimal_ratio": np.std(collected_final_optimal_ratios) if collected_final_optimal_ratios else 0,
            "average_total_reward_per_run": np.mean(collected_total_rewards) if collected_total_rewards else 0,
            "stdev_total_reward_per_run": np.std(collected_total_rewards) if collected_total_rewards else 0,
        },
        "all_runs_data": all_runs_detailed_data
    }

    results_output_dir = "simulation_results"; os.makedirs(results_output_dir, exist_ok=True)
    master_filename = f"MASTER_bandit_results_{safe_model_name_for_filename}_{TOTAL_SIMULATION_RUNS}runs_{master_output_data['experiment_metadata']['experiment_timestamp']}.json"
    master_filepath = os.path.join(results_output_dir, master_filename)
    try:
        with open(master_filepath, "w") as f: json.dump(master_output_data, f, indent=2) # smaller indent for large file
        print(f"\nAll simulation data saved to ONE LARGE JSON FILE: {master_filepath}")
    except Exception as e: print(f"Error saving master JSON file: {e}")

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
