import math
import secrets

def bandit_simulation(choice):
    """
    Simulates the slot machine bandit.
    Slot Machine 1: 30% win rate
    Slot Machine 2: 65% win rate
    """
    random_number = secrets.randbelow(100)
    if choice == 1:
        return "won" if random_number < 30 else "lost"
    elif choice == 2:
        return "won" if random_number < 65 else "lost"

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
    ucb = UCB(2)
    results_log = []
    max_iterations = 100

    for i in range(max_iterations):
        chosen_arm = ucb.select_arm()  # 0 or 1
        machine = chosen_arm + 1       # Slot Machine numbering starts at 1
        result = bandit_simulation(machine)
        reward = 1 if result == "won" else 0
        ucb.update(chosen_arm, reward)
        results_log.append((machine, result))
        print(f"Iteration {i+1}: Chose Slot Machine {machine} -> {result}")
        print(f"Counts: {ucb.counts}, Average rewards: {ucb.values}\n")

    # Summary
    total_wins = sum(1 for _, r in results_log if r == "won")
    print(f"Total wins: {total_wins} out of {max_iterations} plays.")
    print(f"Win rate: {total_wins / max_iterations:.2f}")

if __name__ == "__main__":
    main()
