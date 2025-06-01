import random

random.seed(0)

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
        self.successes = [1] * n_arms  # Beta(1,1) prior
        self.failures = [1] * n_arms

    def select_arm(self):
        samples = [random.betavariate(self.successes[i], self.failures[i]) for i in range(self.n_arms)]
        return samples.index(max(samples))

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

def main():
    simulations = 500
    iterations_per_simulation = 25

    all_simulation_rewards = []

    for sim in range(simulations):
        ts = ThompsonSampling(2)
        results_log = []
        total_reward = 0

        # Pull both arms once initially to avoid tie bias
        for arm in range(ts.n_arms):
            machine = arm + 1
            result = bandit_simulation(machine)
            reward = 1 if result == "won" else 0
            ts.update(arm, reward)
            total_reward += reward
            results_log.append((machine, result))
            print(f"Simulation {sim+1}, Initial pull: Slot Machine {machine} -> {result}")
            print(f"Successes: {ts.successes}, Failures: {ts.failures}\n")

        # Main loop
        for i in range(iterations_per_simulation):
            chosen_arm = ts.select_arm()
            machine = chosen_arm + 1
            result = bandit_simulation(machine)
            reward = 1 if result == "won" else 0
            ts.update(chosen_arm, reward)
            total_reward += reward

            results_log.append((machine, result))
            print(f"Simulation {sim+1}, Iteration {i+1}: Chose Slot Machine {machine} -> {result}")
            print(f"Successes: {ts.successes}, Failures: {ts.failures}\n")

        all_simulation_rewards.append(total_reward)

    # Final summary across all simulations
    average_reward = sum(all_simulation_rewards) / simulations
    print(f"\n==== Summary after {simulations} simulations of {iterations_per_simulation} iterations each ====")
    print(f"Average Total Reward: {average_reward:.2f}")
    print(f"Average Win Rate: {average_reward / (iterations_per_simulation + ts.n_arms):.2f}")

if __name__ == "__main__":
    main()
