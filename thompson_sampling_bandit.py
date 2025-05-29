import random
import numpy as np

random.seed(0)
np.random.seed(0)

def bandit_simulation(choice):
    """
    Simulates the slot machine bandit.
    Slot Machine 1: 30% win rate
    Slot Machine 2: 65% win rate
    """
    r = random.random()
    if choice == 1:
        return "won" if r < 0.30 else "lost"
    elif choice == 2:
        return "won" if r < 0.65 else "lost"

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = [1] * n_arms  # Start with Beta(1,1)
        self.failures = [1] * n_arms

    def select_arm(self):
        samples = [np.random.beta(self.successes[i], self.failures[i]) for i in range(self.n_arms)]
        return samples.index(max(samples))

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

def main():
    ts = ThompsonSampling(2)
    results_log = []
    max_iterations = 100
    total_reward = 0

    for i in range(max_iterations):
        chosen_arm = ts.select_arm()
        machine = chosen_arm + 1
        result = bandit_simulation(machine)
        reward = 1 if result == "won" else 0
        ts.update(chosen_arm, reward)
        total_reward += reward

        results_log.append((machine, result))
        print(f"Iteration {i+1}: Chose Slot Machine {machine} -> {result}")
        print(f"Successes: {ts.successes}, Failures: {ts.failures}\n")

    print(f"Total wins: {total_reward} out of {max_iterations} plays.")
    print(f"Win rate: {total_reward / max_iterations:.2f}")

if __name__ == "__main__":
    main()
