import os
import sys
import json
import time
import matplotlib.pyplot as plt
import numpy as np

def plot_scenario_run_results(data_from_json_file, output_dir_for_plots):
    """
    Plots results for a single scenario run (one model, one scenario, many episodes).
    """
    model_name = data_from_json_file.get("model_name", "unknown_model")
    scenario_display_name = data_from_json_file.get("scenario_name", "unknown_scenario")
    scenario_id = data_from_json_file.get("scenario_id", "unknown_id") # For filename
    
    # This is your "Accuracy Ratio" data per episode
    optimal_ratios_all_episodes = data_from_json_file.get("optimal_choice_ratios_per_episode", [])
    # This is your "Cumulative Reward" (total score per episode) data
    total_scores_all_episodes = data_from_json_file.get("total_scores_per_episode", [])

    if not optimal_ratios_all_episodes or not total_scores_all_episodes:
        print(f"Error: Data for optimal_choice_ratios_per_episode or total_scores_per_episode missing/empty for {scenario_display_name} with {model_name}.")
        return

    num_episodes = len(optimal_ratios_all_episodes)
    episode_indices = list(range(1, num_episodes + 1))

    safe_model_name = model_name.replace("/", "_").replace("-","_")
    safe_scenario_id = scenario_id.replace(" ", "_").replace("/", "_")
    plot_timestamp = data_from_json_file.get("timestamp", time.strftime("%Y%m%d-%H%M%S")) # Use timestamp from data if available
    
    # --- Plot 1: Optimal Choice Ratio Per Episode Over Time ---
    plt.figure(figsize=(12, 6))
    plt.plot(episode_indices, optimal_ratios_all_episodes, linestyle='-', color='blue', linewidth=1, label='Optimal Choice Ratio')
    # Optional: Add a moving average to see trends
    if num_episodes >= 20: # Only plot moving average if enough data points
        moving_avg_window = min(50, num_episodes // 5) # Window size for moving average
        moving_avg = np.convolve(optimal_ratios_all_episodes, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
        # Adjust x-axis for moving average
        moving_avg_x = episode_indices[moving_avg_window-1:]
        if len(moving_avg_x) == len(moving_avg): # Ensure lengths match
             plt.plot(moving_avg_x, moving_avg, color='orange', linestyle='--', linewidth=2, label=f'{moving_avg_window}-Episode Moving Avg.')
        else:
             print(f"Warning: Length mismatch for moving average plot. Expected {len(moving_avg_x)}, got {len(moving_avg)}. Skipping moving average.")


    plt.xlabel(f'Episode Index')
    plt.ylabel('Optimal Choice Ratio (per Episode)')
    plt.title(f'Optimal Choice Ratio per Episode: {scenario_display_name}\nModel: {model_name} ({num_episodes} Episodes)')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    optimal_ratio_path = os.path.join(output_dir_for_plots, f"OptimalRatio_{safe_model_name}_{safe_scenario_id}_{plot_timestamp}.png")
    plt.savefig(optimal_ratio_path)
    plt.close()
    print(f"Saved optimal choice ratio per episode plot to: {optimal_ratio_path}")

    # --- Plot 2: Total Score Per Episode Over Time ---
    plt.figure(figsize=(12, 6))
    plt.plot(episode_indices, total_scores_all_episodes, linestyle='-', color='red', linewidth=1, label='Total Score per Episode')
    # Optional: Add a moving average
    if num_episodes >= 20:
        moving_avg_window = min(50, num_episodes // 5)
        moving_avg_scores = np.convolve(total_scores_all_episodes, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
        moving_avg_scores_x = episode_indices[moving_avg_window-1:]
        if len(moving_avg_scores_x) == len(moving_avg_scores):
             plt.plot(moving_avg_scores_x, moving_avg_scores, color='green', linestyle='--', linewidth=2, label=f'{moving_avg_window}-Episode Moving Avg.')
        else:
            print(f"Warning: Length mismatch for moving average score plot. Expected {len(moving_avg_scores_x)}, got {len(moving_avg_scores)}. Skipping.")


    plt.xlabel(f'Episode Index')
    plt.ylabel('Total Score (per Episode)')
    plt.title(f'Total Score per Episode: {scenario_display_name}\nModel: {model_name} ({num_episodes} Episodes)')
    plt.grid(True)
    plt.legend()
    total_score_path = os.path.join(output_dir_for_plots, f"TotalScore_{safe_model_name}_{safe_scenario_id}_{plot_timestamp}.png")
    plt.savefig(total_score_path)
    plt.close()
    print(f"Saved total score per episode plot to: {total_score_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_test.py <path_to_results_data_JSON_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.isfile(results_file):
        print(f"File not found: {results_file}")
        sys.exit(1)

    # Create a general output directory for all plots if it doesn't exist
    # The plot_scenario_run_results function will save into this with specific names
    base_plot_output_dir = "plots_from_experiments" 
    os.makedirs(base_plot_output_dir, exist_ok=True)

    with open(results_file, "r") as f:
        data_for_one_run = json.load(f) # Expects a single dict from one scenario/model run
    
    plot_scenario_run_results(data_for_one_run, base_plot_output_dir)

if __name__ == "__main__":
    main()
