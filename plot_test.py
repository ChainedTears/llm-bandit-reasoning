import os
import sys
import json
import time
import matplotlib.pyplot as plt

def plot_single_run_from_master(master_data, run_index_to_plot, source_json_filename_base):
    plots_output_dir = "plot_results"
    os.makedirs(plots_output_dir, exist_ok=True)

    experiment_metadata = master_data.get("experiment_metadata", {})
    model_name_from_data = experiment_metadata.get("model_id", "unknown_model")
    safe_model_name_plots = model_name_from_data.replace("/", "_")
    
    all_runs_data_list = master_data.get("all_runs_data", [])

    if not all_runs_data_list:
        print("Error: 'all_runs_data' list is missing or empty in the master JSON.")
        return

    if run_index_to_plot < 0 or run_index_to_plot >= len(all_runs_data_list):
        print(f"Error: Run index {run_index_to_plot} is out of bounds. Available runs: 0 to {len(all_runs_data_list) - 1}.")
        return
        
    data_for_this_run = all_runs_data_list[run_index_to_plot]
    run_id_from_data = data_for_this_run.get("run_id", run_index_to_plot + 1) 

    num_iterations_from_data = data_for_this_run.get("num_iterations_in_run", 0)
    plotting_run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    if num_iterations_from_data == 0:
        print(f"Error: 'num_iterations_in_run' is 0 or missing for run_id {run_id_from_data}. Cannot generate plots.")
        return

    iteration_numbers = list(range(1, num_iterations_from_data + 1))

    # Plot 1: AI Choices Over Iterations
    choices_data = data_for_this_run.get("choices_per_iteration", [])
    if choices_data:
        plt.figure(figsize=(12, 6))
        plt.step(iteration_numbers, choices_data, where='post', linestyle='-', marker='o', markersize=4, label=f'AI Choice (Machine #)')
        plt.xlabel('Iteration Number'); plt.ylabel('AI Choice (Slot Machine)')
        plt.title(f'AI Slot Machine Choices (Run {run_id_from_data})\nModel: {model_name_from_data}\nSource: {source_json_filename_base}.json')
        unique_choices_made = sorted(list(set(choices_data))); y_ticks_choices = [1, 2] 
        if unique_choices_made:
            if not (1 in unique_choices_made and 2 in unique_choices_made): y_ticks_choices = [1,2]
            else: y_ticks_choices = sorted(list(set(unique_choices_made + [1,2])))
        plt.yticks(y_ticks_choices); plt.grid(True, linestyle='--', alpha=0.7); plt.legend(loc='best'); plt.tight_layout()
        choice_plot_path = os.path.join(plots_output_dir, f"choices_run{run_id_from_data}_{safe_model_name_plots}_{source_json_filename_base}_{plotting_run_timestamp}.png")
        plt.savefig(choice_plot_path); plt.close()
        print(f"Saved choices plot for run {run_id_from_data} to: {choice_plot_path}")
    else:
        print(f"Warning: 'choices_per_iteration' data missing for run_id {run_id_from_data}. Skipping choices plot.")

    # Plot 2: Cumulative Reward
    cumulative_rewards_data = data_for_this_run.get("cumulative_rewards_per_iteration", [])
    if cumulative_rewards_data:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, cumulative_rewards_data, marker='o', linestyle='-', color='green', label='Cumulative Reward')
        plt.xlabel('Iteration Number'); plt.ylabel('Cumulative Reward')
        plt.title(f"AI's Cumulative Reward (Run {run_id_from_data})\nModel: {model_name_from_data}\nSource: {source_json_filename_base}.json")
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend(loc='best'); plt.tight_layout()
        reward_plot_path = os.path.join(plots_output_dir, f"cum_reward_run{run_id_from_data}_{safe_model_name_plots}_{source_json_filename_base}_{plotting_run_timestamp}.png")
        plt.savefig(reward_plot_path); plt.close()
        print(f"Saved cumulative reward plot for run {run_id_from_data} to: {reward_plot_path}")
    else:
        print(f"Warning: 'cumulative_rewards_per_iteration' data missing for run_id {run_id_from_data}. Skipping plot.")

    # Plot 3: Cumulative Regret
    cumulative_regrets_data = data_for_this_run.get("cumulative_regrets_per_iteration", [])
    if cumulative_regrets_data:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, cumulative_regrets_data, marker='s', linestyle='-', color='red', label='Cumulative Regret')
        plt.xlabel('Iteration Number'); plt.ylabel('Cumulative Regret')
        plt.title(f"AI's Cumulative Regret (Run {run_id_from_data})\nModel: {model_name_from_data}\nSource: {source_json_filename_base}.json")
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend(loc='best'); plt.tight_layout()
        regret_plot_path = os.path.join(plots_output_dir, f"cum_regret_run{run_id_from_data}_{safe_model_name_plots}_{source_json_filename_base}_{plotting_run_timestamp}.png")
        plt.savefig(regret_plot_path); plt.close()
        print(f"Saved cumulative regret plot for run {run_id_from_data} to: {regret_plot_path}")
    else:
        print(f"Warning: 'cumulative_regrets_per_iteration' data missing for run_id {run_id_from_data}. Skipping plot.")

    # Plot 4: Optimal Arm Selection Ratio
    optimal_ratio_data = data_for_this_run.get("optimal_arm_selection_ratio_per_iteration", [])
    if optimal_ratio_data:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, optimal_ratio_data, marker='^', linestyle='-', color='purple', label='Optimal Arm Selection Ratio')
        plt.xlabel('Iteration Number'); plt.ylabel('Cumulative Ratio of Selecting Optimal Arm')
        plt.title(f"AI's Optimal Arm Selection Ratio (Run {run_id_from_data})\nModel: {model_name_from_data}\nSource: {source_json_filename_base}.json")
        plt.ylim(0, 1.05); plt.grid(True, linestyle='--', alpha=0.7); plt.legend(loc='best'); plt.tight_layout()
        optimal_ratio_plot_path = os.path.join(plots_output_dir, f"opt_arm_ratio_run{run_id_from_data}_{safe_model_name_plots}_{source_json_filename_base}_{plotting_run_timestamp}.png")
        plt.savefig(optimal_ratio_plot_path); plt.close()
        print(f"Saved optimal arm selection ratio plot for run {run_id_from_data} to: {optimal_ratio_plot_path}")
    else:
        print(f"Warning: 'optimal_arm_selection_ratio_per_iteration' data missing for run_id {run_id_from_data}. Skipping plot.")

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_test.py <path_to_master_results_file.json> <run_index_to_plot>")
        print("Example: python plot_test.py simulation_results/MASTER_bandit_results_MODEL_500runs_TIMESTAMP.json 0")
        sys.exit(1)

    input_json_file_path = sys.argv[1]
    try:
        run_index_to_plot = int(sys.argv[2])
    except ValueError:
        print("Error: <run_index_to_plot> must be an integer.")
        sys.exit(1)

    if not os.path.isfile(input_json_file_path):
        print(f"Error: File not found: {input_json_file_path}")
        sys.exit(1)

    source_json_filename_base = os.path.splitext(os.path.basename(input_json_file_path))[0]
    try:
        with open(input_json_file_path, "r") as f: master_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_file_path}."); sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading {input_json_file_path}: {e}"); sys.exit(1)
        
    required_top_level_keys = ["experiment_metadata", "all_runs_data"] 
    if not all(key in master_data for key in required_top_level_keys):
        print(f"Error: Master JSON file {input_json_file_path} is missing required keys."); sys.exit(1)

    plot_single_run_from_master(master_data, run_index_to_plot, source_json_filename_base)

if __name__ == "__main__":
    main()
