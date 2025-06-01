import os
import sys
import json
import time
import matplotlib.pyplot as plt

def plot_all_results(data_from_json, source_json_filename_base):
    plots_output_dir = "plot_results"
    os.makedirs(plots_output_dir, exist_ok=True)

    model_name_from_data = data_from_json.get("model_id", "unknown_model")
    safe_model_name_plots = model_name_from_data.replace("/", "_")
    num_iterations_from_data = data_from_json.get("num_iterations", 0)
    
    plotting_run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    if num_iterations_from_data == 0:
        print("Error: 'num_iterations' is 0 or missing. Cannot generate plots.")
        return

    iteration_numbers = list(range(1, num_iterations_from_data + 1))

    # Plot 1: AI Choices Over Iterations
    choices_data = data_from_json.get("choices_per_iteration", [])
    if choices_data:
        plt.figure(figsize=(12, 6))
        plt.step(iteration_numbers, choices_data, where='post', linestyle='-', marker='o', markersize=4, label=f'AI Choice (Machine #)')
        plt.xlabel('Iteration Number')
        plt.ylabel('AI Choice (Slot Machine)')
        plt.title(f'AI Slot Machine Choices\nModel: {model_name_from_data}\nSource: {source_json_filename_base}.json')
        
        unique_choices_made = sorted(list(set(choices_data)))
        y_ticks_choices = [1, 2] 
        if unique_choices_made:
            if not (1 in unique_choices_made and 2 in unique_choices_made): # Ensure both 1 and 2 are options if not all present
                 y_ticks_choices = [1,2]
            else: # if both 1 and 2 are present or more
                 y_ticks_choices = sorted(list(set(unique_choices_made + [1,2])))


        plt.yticks(y_ticks_choices)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        choice_plot_path = os.path.join(plots_output_dir, f"choices_{safe_model_name_plots}_{source_json_filename_base}_{plotting_run_timestamp}.png")
        plt.savefig(choice_plot_path)
        plt.close()
        print(f"Saved choices plot to: {choice_plot_path}")
    else:
        print("Warning: 'choices_per_iteration' data missing. Skipping choices plot.")

    # Plot 2: Cumulative Reward Over Iterations
    cumulative_rewards_data = data_from_json.get("cumulative_rewards_per_iteration", [])
    if cumulative_rewards_data:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, cumulative_rewards_data, marker='o', linestyle='-', color='green', label='Cumulative Reward')
        plt.xlabel('Iteration Number')
        plt.ylabel('Cumulative Reward')
        plt.title(f"AI's Cumulative Reward Over Iterations\nModel: {model_name_from_data}\nSource: {source_json_filename_base}.json")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        reward_plot_path = os.path.join(plots_output_dir, f"cumulative_reward_{safe_model_name_plots}_{source_json_filename_base}_{plotting_run_timestamp}.png")
        plt.savefig(reward_plot_path)
        plt.close()
        print(f"Saved cumulative reward plot to: {reward_plot_path}")
    else:
        print("Warning: 'cumulative_rewards_per_iteration' data missing. Skipping cumulative reward plot.")

    # Plot 3: Cumulative Regret Over Iterations
    cumulative_regrets_data = data_from_json.get("cumulative_regrets_per_iteration", [])
    if cumulative_regrets_data:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, cumulative_regrets_data, marker='s', linestyle='-', color='red', label='Cumulative Regret')
        plt.xlabel('Iteration Number')
        plt.ylabel('Cumulative Regret')
        plt.title(f"AI's Cumulative Regret Over Iterations\nModel: {model_name_from_data}\nSource: {source_json_filename_base}.json")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        regret_plot_path = os.path.join(plots_output_dir, f"cumulative_regret_{safe_model_name_plots}_{source_json_filename_base}_{plotting_run_timestamp}.png")
        plt.savefig(regret_plot_path)
        plt.close()
        print(f"Saved cumulative regret plot to: {regret_plot_path}")
    else:
        print("Warning: 'cumulative_regrets_per_iteration' data missing. Skipping cumulative regret plot.")

    # Plot 4: Optimal Arm Selection Ratio Over Iterations
    optimal_ratio_data = data_from_json.get("optimal_arm_selection_ratio_per_iteration", [])
    if optimal_ratio_data:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_numbers, optimal_ratio_data, marker='^', linestyle='-', color='purple', label='Optimal Arm (Machine 2) Selection Ratio')
        plt.xlabel('Iteration Number')
        plt.ylabel('Cumulative Ratio of Selecting Optimal Arm')
        plt.title(f"AI's Optimal Arm Selection Ratio Over Iterations\nModel: {model_name_from_data}\nSource: {source_json_filename_base}.json")
        plt.ylim(0, 1.05) 
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        optimal_ratio_plot_path = os.path.join(plots_output_dir, f"optimal_arm_ratio_{safe_model_name_plots}_{source_json_filename_base}_{plotting_run_timestamp}.png")
        plt.savefig(optimal_ratio_plot_path)
        plt.close()
        print(f"Saved optimal arm selection ratio plot to: {optimal_ratio_plot_path}")
    else:
        print("Warning: 'optimal_arm_selection_ratio_per_iteration' data missing. Skipping optimal arm ratio plot.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_test.py <path_to_results_file.json>")
        sys.exit(1)

    input_json_file_path = sys.argv[1]
    if not os.path.isfile(input_json_file_path):
        print(f"Error: File not found: {input_json_file_path}")
        sys.exit(1)

    source_json_filename_base = os.path.splitext(os.path.basename(input_json_file_path))[0]

    try:
        with open(input_json_file_path, "r") as f:
            loaded_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_file_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading {input_json_file_path}: {e}")
        sys.exit(1)
        
    required_top_level_keys = ["model_id", "num_iterations"] 
    if not all(key in loaded_data for key in required_top_level_keys):
        print(f"Error: JSON file {input_json_file_path} is missing required keys: {required_top_level_keys}")
        sys.exit(1)

    plot_all_results(loaded_data, source_json_filename_base)

if __name__ == "__main__":
    main()
