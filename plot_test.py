import os
import sys
import json
import time
import matplotlib.pyplot as plt
import numpy as np 

def plot_averaged_results(averaged_data_json, source_json_filename_base):
    plots_output_dir = "plot_results"
    os.makedirs(plots_output_dir, exist_ok=True)

    metadata = averaged_data_json.get("experiment_metadata", {})
    model_name = metadata.get("model_id", "unknown_model")
    safe_model_name_plots = model_name.replace("/", "_")
    num_iterations = metadata.get("iterations_per_run", 0)
    total_runs = metadata.get("successful_runs", metadata.get("total_simulation_runs_attempted", "N/A")) # Use successful_runs

    learning_curves = averaged_data_json.get("averaged_learning_curves", {})
    plotting_run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    if num_iterations == 0:
        print("Error: 'iterations_per_run' is 0 or missing. Cannot generate plots.")
        return
    if not learning_curves:
        print("Error: 'averaged_learning_curves' data is missing. Cannot generate plots.")
        return

    iteration_numbers = list(range(1, num_iterations + 1))
    title_suffix = f"\nModel: {model_name} (Avg. over {total_runs} successful runs)\nSource: {source_json_filename_base}.json"

    # --- Plot 1: Averaged Cumulative Reward ---
    avg_cum_rewards = learning_curves.get("avg_cumulative_rewards_per_iteration", [])
    std_cum_rewards = learning_curves.get("std_cumulative_rewards_per_iteration", [])
    if avg_cum_rewards:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, avg_cum_rewards, marker='o', linestyle='-', color='green', label='Avg. Cumulative Reward')
        if std_cum_rewards and len(std_cum_rewards) == len(avg_cum_rewards):
            plt.fill_between(iteration_numbers, 
                             np.array(avg_cum_rewards) - np.array(std_cum_rewards),
                             np.array(avg_cum_rewards) + np.array(std_cum_rewards),
                             color='green', alpha=0.2, label='Std. Dev.')
        plt.xlabel('Iteration Number'); plt.ylabel('Avg. Cumulative Reward')
        plt.title(f"Average Cumulative Reward{title_suffix}")
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend(loc='best'); plt.tight_layout()
        reward_plot_path = os.path.join(plots_output_dir, f"avg_cum_reward_{safe_model_name_plots}_{plotting_run_timestamp}.png")
        plt.savefig(reward_plot_path); plt.close()
        print(f"Saved avg cumulative reward plot to: {reward_plot_path}")

    # --- Plot 2: Averaged Cumulative Regret ---
    avg_cum_regrets = learning_curves.get("avg_cumulative_regrets_per_iteration", [])
    std_cum_regrets = learning_curves.get("std_cumulative_regrets_per_iteration", [])
    if avg_cum_regrets:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, avg_cum_regrets, marker='s', linestyle='-', color='red', label='Avg. Cumulative Regret')
        if std_cum_regrets and len(std_cum_regrets) == len(avg_cum_regrets):
            plt.fill_between(iteration_numbers,
                             np.array(avg_cum_regrets) - np.array(std_cum_regrets),
                             np.array(avg_cum_regrets) + np.array(std_cum_regrets),
                             color='red', alpha=0.2, label='Std. Dev.')
        plt.xlabel('Iteration Number'); plt.ylabel('Avg. Cumulative Regret')
        plt.title(f"Average Cumulative Regret{title_suffix}")
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend(loc='best'); plt.tight_layout()
        regret_plot_path = os.path.join(plots_output_dir, f"avg_cum_regret_{safe_model_name_plots}_{plotting_run_timestamp}.png")
        plt.savefig(regret_plot_path); plt.close()
        print(f"Saved avg cumulative regret plot to: {regret_plot_path}")

    # --- Plot 3: Averaged Optimal Arm Selection Ratio ---
    avg_opt_arm_ratio = learning_curves.get("avg_optimal_arm_selection_ratio_per_iteration", [])
    std_opt_arm_ratio = learning_curves.get("std_optimal_arm_selection_ratio_per_iteration", [])
    if avg_opt_arm_ratio:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, avg_opt_arm_ratio, marker='^', linestyle='-', color='purple', label='Avg. Optimal Arm Selection Ratio')
        if std_opt_arm_ratio and len(std_opt_arm_ratio) == len(avg_opt_arm_ratio):
            plt.fill_between(iteration_numbers,
                             np.array(avg_opt_arm_ratio) - np.array(std_opt_arm_ratio),
                             np.array(avg_opt_arm_ratio) + np.array(std_opt_arm_ratio),
                             color='purple', alpha=0.2, label='Std. Dev.')
        plt.xlabel('Iteration Number'); plt.ylabel('Avg. Ratio of Selecting Optimal Arm')
        plt.title(f"Average Optimal Arm Selection Ratio{title_suffix}")
        plt.ylim(0, 1.05); plt.grid(True, linestyle='--', alpha=0.7); plt.legend(loc='best'); plt.tight_layout()
        opt_ratio_plot_path = os.path.join(plots_output_dir, f"avg_opt_arm_ratio_{safe_model_name_plots}_{plotting_run_timestamp}.png")
        plt.savefig(opt_ratio_plot_path); plt.close()
        print(f"Saved avg optimal arm selection ratio plot to: {opt_ratio_plot_path}")

    # --- Plot 4: Averaged Probability of Choosing Arm 2 ---
    avg_prob_choice2 = learning_curves.get("avg_prob_choice_arm2_per_iteration", [])
    std_prob_choice2 = learning_curves.get("std_prob_choice_arm2_per_iteration", [])
    if avg_prob_choice2:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, avg_prob_choice2, marker='.', linestyle='-', color='blue', label='Avg. P(Choose Optimal Arm 2)')
        if std_prob_choice2 and len(std_prob_choice2) == len(avg_prob_choice2):
             plt.fill_between(iteration_numbers,
                             np.array(avg_prob_choice2) - np.array(std_prob_choice2),
                             np.array(avg_prob_choice2) + np.array(std_prob_choice2),
                             color='blue', alpha=0.2, label='Std. Dev.')
        plt.xlabel('Iteration Number'); plt.ylabel('Avg. Probability of Choosing Optimal Arm (Arm 2)')
        plt.title(f"Average Probability of Choosing Optimal Arm per Iteration{title_suffix}")
        plt.ylim(0, 1.05); plt.grid(True, linestyle='--', alpha=0.7); plt.legend(loc='best'); plt.tight_layout()
        prob_choice_plot_path = os.path.join(plots_output_dir, f"avg_prob_choice2_{safe_model_name_plots}_{plotting_run_timestamp}.png")
        plt.savefig(prob_choice_plot_path); plt.close()
        print(f"Saved avg probability of choosing Arm 2 plot to: {prob_choice_plot_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_test.py <path_to_averaged_results_file.json>")
        sys.exit(1)
    input_json_file_path = sys.argv[1]
    if not os.path.isfile(input_json_file_path):
        print(f"Error: File not found: {input_json_file_path}"); sys.exit(1)
    source_json_filename_base = os.path.splitext(os.path.basename(input_json_file_path))[0]
    try:
        with open(input_json_file_path, "r") as f: loaded_data = json.load(f)
    except json.JSONDecodeError: print(f"Error: Could not decode JSON from {input_json_file_path}."); sys.exit(1)
    except Exception as e: print(f"An error occurred: {e}"); sys.exit(1)
    required_keys = ["experiment_metadata", "averaged_learning_curves"] 
    if not all(key in loaded_data for key in required_keys):
        print(f"Error: JSON file is missing required keys for averaged data plotting."); sys.exit(1)
    plot_averaged_results(loaded_data, source_json_filename_base)

if __name__ == "__main__":
    main()
