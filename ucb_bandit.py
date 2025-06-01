import math
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

class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0] * n_arms   # Number of times each arm played
        self.values = [0.0] * n_arms # Average reward of each arm
        self.total_counts = 0
    
    def select_arm(self):
        # Play each arm once before applying UCB
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            bonus = math.sqrt((2 * math.log(self.total_counts)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        return ucb_values.index(max(ucb_values))
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update average reward
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

def main():
    simulations = 500
    iterations_per_simulation = 25

    all_simulation_rewards = []
    all_simulation_regrets = []

    for sim in range(simulations):
        ucb = UCB(2)
        results_log = []
        cumulative_rewards = []
        regrets = []
        total_reward = 0
        best_possible_reward_rate = 0.65  # Since machine 2 is best

        for i in range(iterations_per_simulation):
            chosen_arm = ucb.select_arm()
            machine = chosen_arm + 1
            result = bandit_simulation(machine)
            reward = 1 if result == "won" else 0
            ucb.update(chosen_arm, reward)

            total_reward += reward
            cumulative_rewards.append(total_reward)

            # Regret = missed opportunity by not playing best machine
            optimal_total_reward = (i + 1) * best_possible_reward_rate
            regret = optimal_total_reward - total_reward
            regrets.append(regret)

            # Print every 10 iterations
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Simulation {sim+1}, Iteration {i+1}: Chose Slot Machine {machine} -> {result}")
                print(f"Counts: {ucb.counts}, Avg rewards: {ucb.values}")
                print(f"Cumulative Reward: {total_reward}, Regret: {regret:.2f}\n")

        all_simulation_rewards.append(total_reward)
        all_simulation_regrets.append(regrets[-1])

    # Final summary across all simulations
    average_reward = sum(all_simulation_rewards) / simulations
    average_regret = sum(all_simulation_regrets) / simulations
    print(f"\n==== Summary after {simulations} simulations of {iterations_per_simulation} iterations each ====")
    print(f"Average Total Reward: {average_reward:.2f}")
    print(f"Average Final Regret: {average_regret:.2f}")

if __name__ == "__main__":
    main()
    
