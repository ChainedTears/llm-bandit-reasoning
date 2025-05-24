import os
import sys
import json
import time
import matplotlib.pyplot as plt

def plot_results(choices, correctness, model_name="model"):
    output_dir = "plot_results"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_model_name = model_name.replace("/", "_")

    iterations = list(range(1, len(choices) + 1))

    plt.figure(figsize=(10, 4))
    plt.step(iterations, choices, where='post')
    plt.xlabel('Iteration')
    plt.ylabel('AI Choice (Slot Machine)')
    plt.title(f'AI Slot Machine Choices Over Iterations\nModel: {model_name}')
    plt.yticks([1, 2])
    plt.grid(True)
    choice_path = os.path.join(output_dir, f"choices_plot_{safe_model_name}_{timestamp}.png")
    plt.savefig(choice_path)
    plt.close()
    print(f"Saved choices plot to: {choice_path}")

    plt.figure(figsize=(10, 4))
    plt.plot(iterations, correctness, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Correct Choice Ratio')
    plt.title(f"AI's Correct Choice Ratio Over Iterations\nModel: {model_name}")
    plt.ylim(0, 1)
    plt.grid(True)
    correctness_path = os.path.join(output_dir, f"correctness_plot_{safe_model_name}_{timestamp}.png")
    plt.savefig(correctness_path)
    plt.close()
    print(f"Saved correctness plot to: {correctness_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_test.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.isfile(results_file):
        print(f"File not found: {results_file}")
        sys.exit(1)

    with open(results_file, "r") as f:
        data = json.load(f)

    model_name = data.get("model", "unknown_model")
    choices = data.get("choices", [])
    correctness = data.get("correctness", [])

    plot_results(choices, correctness, model_name)

if __name__ == "__main__":
    main()
