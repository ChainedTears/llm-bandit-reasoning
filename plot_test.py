import os
import sys
import json
import time
import matplotlib.pyplot as plt
import numpy as np 

def plot_averaged_results(averaged_data_json, source_json_basename):
    plots_output_dir = "plot_results"
    os.makedirs(plots_output_dir, exist_ok=True)

    metadata = averaged_data_json.get("experiment_metadata", {})
    model_name = metadata.get("model_id", "unknown_model")
    safe_model_name_plots = model_name.replace("/", "_")
    num_iterations = metadata.get("iterations_per_run", 0)
    # Use successful_runs for the title if available, otherwise fall back
    num_successful_runs = metadata.get("successful_runs_averaged", metadata.get("total_runs_attempted", "N/A")) 

    learning_curves = averaged_data_json.get("averaged_learning_curves", {})
    plotting_run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    if num_iterations == 0 or not learning_curves:
        print("Error: Insufficient data in JSON for plotting averaged curves.")
        return

    iterations_axis = learning_curves.get("iterations", list(range(1, num_iterations + 1)))
    if not learning_curves.get("iterations") and num_iterations > 0: # Fallback
        iterations_axis = list(range(1, num_iterations + 1))


    title_suffix = f"\nModel: {model_name} (Avg. over {num_successful_runs} successful runs)\nSource: {source_json_basename}.json"

    # Plot 1: Averaged Cumulative Reward
    avg_cum_reward = learning_curves.get("avg_cumulative_reward_per_iteration")
    std_cum_reward = learning_curves.get("std_cumulative_reward_per_iteration")
    if avg_cum_reward:
        plt.figure(figsize=(10, 5))
        plt.plot(iterations_axis, avg_cum_reward, marker='o', linestyle='-', markersize=4, color='green', label='Avg. Cumulative Reward')
        if std_cum_reward and len(std_cum_reward) == len(avg_cum_reward):
            plt.fill_between(iterations_axis, 
                             np.array(avg_cum_reward) - np.array(std_cum_reward),
                             np.array(avg_cum_reward) + np.array(std_cum_reward),
                             color='green', alpha=0.2, label='Std. Dev.')
        plt.xlabel('Iteration'); plt.ylabel('Avg. Cumulative Reward')
        plt.title(f"Average Cumulative Reward{title_suffix}")
        plt.grid(True, alpha=0.6); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"AVG_CumReward_{safe_model_name_plots}_{plotting_run_timestamp}.png"))
        plt.close(); print(f"Saved Avg. Cumulative Reward plot.")
    else: print("Skipping Avg. Cumulative Reward plot: data not found.")

    # Plot 2: Averaged Cumulative Regret
    avg_cum_regret = learning_curves.get("avg_cumulative_regret_per_iteration")
    std_cum_regret = learning_curves.get("std_cumulative_regret_per_iteration")
    if avg_cum_regret:
        plt.figure(figsize=(10, 5))
        plt.plot(iterations_axis, avg_cum_regret, marker='s', linestyle='-', markersize=4, color='red', label='Avg. Cumulative Regret')
        if std_cum_regret and len(std_cum_regret) == len(avg_cum_regret):
            plt.fill_between(iterations_axis,
                             np.array(avg_cum_regret) - np.array(std_cum_regret),
                             np.array(avg_cum_regret) + np.array(std_cum_regret),
                             color='red', alpha=0.2, label='Std. Dev.')
        plt.xlabel('Iteration'); plt.ylabel('Avg. Cumulative Regret')
        plt.title(f"Average Cumulative Regret{title_suffix}")
        plt.grid(True, alpha=0.6); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"AVG_CumRegret_{safe_model_name_plots}_{plotting_run_timestamp}.png"))
        plt.close(); print(f"Saved Avg. Cumulative Regret plot.")
    else: print("Skipping Avg. Cumulative Regret plot: data not found.")

    # Plot 3: Averaged Optimal Arm Selection Ratio
    avg_opt_ratio = learning_curves.get("avg_optimal_arm_selection_ratio_per_iteration")
    std_opt_ratio = learning_curves.get("std_optimal_arm_selection_ratio_per_iteration")
    if avg_opt_ratio:
        plt.figure(figsize=(10, 5))
        plt.plot(iterations_axis, avg_opt_ratio, marker='^', linestyle='-', markersize=4, color='purple', label='Avg. Optimal Arm Selection Ratio')
        if std_opt_ratio and len(std_opt_ratio) == len(avg_opt_ratio):
            plt.fill_between(iterations_axis,
                             np.array(avg_opt_ratio) - np.array(std_opt_ratio),
                             np.array(avg_opt_ratio) + np.array(std_opt_ratio),
                             color='purple', alpha=0.2, label='Std. Dev.')
        plt.xlabel('Iteration'); plt.ylabel('Avg. Optimal Arm Selection Ratio')
        plt.title(f"Average Optimal Arm Selection Ratio{title_suffix}")
        plt.ylim(0, 1.05); plt.grid(True, alpha=0.6); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"AVG_OptimalRatio_{safe_model_name_plots}_{plotting_run_timestamp}.png"))
        plt.close(); print(f"Saved Avg. Optimal Arm Selection Ratio plot.")
    else: print("Skipping Avg. Optimal Arm Selection Ratio plot: data not found.")

    # Plot 4: Averaged Probability of Choosing Optimal Arm (Arm 2)
    avg_prob_opt_choice = avg_curves.get("avg_prob_choosing_optimal_arm_per_iteration")
    std_prob_opt_choice = avg_curves.get("std_prob_choosing_optimal_arm_per_iteration")
    if avg_prob_opt_choice:
        plt.figure(figsize=(10, 5))
        plt.plot(iterations_axis, avg_prob_opt_choice, marker='.', linestyle='-', markersize=4, color='blue', label='Avg. P(Choose Optimal Arm 2)')
        if std_prob_opt_choice and len(std_prob_opt_choice) == len(avg_prob_opt_choice):
            plt.fill_between(iterations_axis,
                             np.array(avg_prob_opt_choice) - np.array(std_prob_opt_choice),
                             np.array(avg_prob_opt_choice) + np.array(std_prob_opt_choice),
                             color='blue', alpha=0.2, label='Std. Dev.')
        plt.xlabel('Iteration'); plt.ylabel('Avg. P(Choose Optimal Arm 2)')
        plt.title(f"Average Prob. of Choosing Optimal Arm (Arm 2){title_suffix}")
        plt.ylim(0, 1.05); plt.grid(True, alpha=0.6); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"AVG_ProbOptimalChoice_{safe_model_name_plots}_{plotting_run_timestamp}.png"))
        plt.close(); print(f"Saved Avg. Prob. Optimal Choice plot.")
    else: print("Skipping Avg. Prob. Optimal Choice plot: data not found.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_test.py <path_to_averaged_results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.isfile(results_file):
        print(f"File not found: {results_file}")
        sys.exit(1)
    
    results_basename = os.path.splitext(os.path.basename(results_file))[0]

    with open(results_file, "r") as f:
        data = json.load(f)

    if "experiment_metadata" not in data or "averaged_learning_curves" not in data:
        print("Error: JSON file does not appear to contain averaged results data in the expected format.")
        sys.exit(1)

    plot_averaged_results(data, results_basename)

if __name__ == "__main__":
    main()
