import os
import sys
import json
import time
import matplotlib.pyplot as plt
import numpy as np # For calculating cumulative means

def plot_episode_results(episode_data, output_dir):
    model_name = episode_data.get("model_name", "unknown_model")
    scenario_id = episode_data.get("scenario_id", "unknown_scenario")
    episode_num = episode_data.get("episode_number", 0)
    
    choices = episode_data.get("choices_per_iteration", [])
    scores = episode_data.get("scores_per_iteration", [])
    optimal_choices_flags = episode_data.get("optimal_choices_per_iteration", []) # List of 0s and 1s
    # valid_responses_flags = episode_data.get("valid_responses_per_iteration", [])

    if not choices:
        print(f"No choice data for {scenario_id} ep {episode_num}. Skipping plots.")
        return

    safe_model_name = model_name.replace("/", "_")
    timestamp = episode_data.get("timestamp", time.strftime("%Y%m%d-%H%M%S")) # Get timestamp from data or generate
    
    iterations = list(range(1, len(choices) + 1))

    # --- Plot 1: Choices Over Iterations (Potentially tricky with string labels) ---
    # This plot might be less useful if option names are very different across scenarios
    # or if you have too many options. Consider if this is still needed or how to best represent it.
    # For now, we'll attempt it but it might need finessing.
    plt.figure(figsize=(12, 5))
    # Create a mapping from string choices to integers for plotting if they are strings
    unique_choices = sorted(list(set(choices)))
    choice_to_int = {choice: i for i, choice in enumerate(unique_choices)}
    numeric_choices = [choice_to_int.get(choice, -1) for choice in choices] # -1 for unmapped/error

    plt.step(iterations, numeric_choices, where='post')
    plt.xlabel('Iteration')
    plt.ylabel('AI Choice (Mapped to Int)')
    plt.title(f'AI Choices: {scenario_id} (Ep. {episode_num})\nModel: {model_name}')
    if unique_choices:
        plt.yticks(list(choice_to_int.values()), unique_choices)
    plt.grid(True)
    choice_path = os.path.join(output_dir, f"choices_{safe_model_name}_{scenario_id}_ep{episode_num}_{timestamp}.png")
    plt.savefig(choice_path)
    plt.close()
    print(f"Saved choices plot to: {choice_path}")

    # --- Plot 2: Cumulative Average Score Over Iterations ---
    if scores:
        cumulative_scores = np.cumsum(scores)
        cumulative_avg_scores = [cs / (i + 1) for i, cs in enumerate(cumulative_scores)]
        plt.figure(figsize=(12, 5))
        plt.plot(iterations, cumulative_avg_scores, marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Average Score')
        plt.title(f'Cumulative Avg Score: {scenario_id} (Ep. {episode_num})\nModel: {model_name}')
        plt.grid(True)
        avg_score_path = os.path.join(output_dir, f"avg_score_{safe_model_name}_{scenario_id}_ep{episode_num}_{timestamp}.png")
        plt.savefig(avg_score_path)
        plt.close()
        print(f"Saved cumulative average score plot to: {avg_score_path}")

    # --- Plot 3: Cumulative EV-Optimal Choice Ratio Over Iterations ---
    if optimal_choices_flags:
        cumulative_optimal_choices = np.cumsum(optimal_choices_flags)
        cumulative_optimal_ratio = [coc / (i + 1) for i, coc in enumerate(cumulative_optimal_choices)]
        plt.figure(figsize=(12, 5))
        plt.plot(iterations, cumulative_optimal_ratio, marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative EV-Optimal Choice Ratio')
        plt.title(f'EV-Optimal Choice Ratio: {scenario_id} (Ep. {episode_num})\nModel: {model_name}')
        plt.ylim(0, 1.05) # Keep Y-axis between 0 and 1
        plt.grid(True)
        optimal_ratio_path = os.path.join(output_dir, f"optimal_ratio_{safe_model_name}_{scenario_id}_ep{episode_num}_{timestamp}.png")
        plt.savefig(optimal_ratio_path)
        plt.close()
        print(f"Saved EV-optimal choice ratio plot to: {optimal_ratio_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_test.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.isfile(results_file):
        print(f"File not found: {results_file}")
        sys.exit(1)

    output_dir = "plot_results_per_episode" # Changed directory name
    os.makedirs(output_dir, exist_ok=True)

    with open(results_file, "r") as f:
        # Assuming the JSON file is a list of episode data dictionaries
        all_episodes_data = json.load(f) 
    
    if not isinstance(all_episodes_data, list):
        print("Error: JSON file should contain a list of episode results.")
        # If your JSON is structured differently (e.g., dict of scenarios, then list of episodes)
        # you'll need to adjust the loading and iteration here.
        # For example, if it's a dict like your `overall_results`:
        # loaded_data = json.load(f)
        # if "experiment_details" in loaded_data and "episode_data" in loaded_data:
        #    all_episodes_data = loaded_data["episode_data"] # Assuming episode data is in a sub-key
        # else: ... error ...
        sys.exit(1)


    for episode_data in all_episodes_data:
        plot_episode_results(episode_data, output_dir)

if __name__ == "__main__":
    main()
