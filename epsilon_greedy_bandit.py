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

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = [0] * n_arms    # Number of times each arm played
        self.values = [0.0] * n_arms  # Average reward of each arm
    
    def select_arm(self):
        import random
        # Explore with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(self.n_arms)
        # Exploit: choose best known arm
        else:
            return self.values.index(max(self.values))
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update average reward for chosen arm
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

def main():
    epsilon = 0.1  # Set exploration rate to 10%
    simulations = 500
    iterations_per_simulation = 25
    all_simulation_wins = []

    for sim in range(simulations):
        eg = EpsilonGreedy(2, epsilon)  # Initialize strategy for 2 slot machines
        results_log = []                # Keep track of all results for analysis

        for i in range(iterations_per_simulation):
            chosen_arm = eg.select_arm()  # Choose arm to play (0 or 1)
            machine = chosen_arm + 1      # Slot machines are labeled 1 and 2 for clarity

            # Simulate the slot machine pull and get result
            result = bandit_simulation(machine)

            # Convert "won"/"lost" to numeric reward (1 or 0)
            reward = 1 if result == "won" else 0

            # Update strategy with the reward received
            eg.update(chosen_arm, reward)

            # Log the result of this iteration
            results_log.append((machine, result))

            # Print detailed iteration results and current strategy state
            print(f"Simulation {sim+1}, Iteration {i+1}: Chose Slot Machine {machine} -> {result}")
            print(f"Counts: {eg.counts}, Average rewards: {eg.values}\n")

        # After each simulation, calculate total wins
        total_wins = sum(1 for _, r in results_log if r == "won")
        all_simulation_wins.append(total_wins)

    # After all simulations, calculate average wins and overall win rate
    average_wins = sum(all_simulation_wins) / simulations
    average_win_rate = average_wins / iterations_per_simulation
    print(f"\n==== Summary after {simulations} simulations of {iterations_per_simulation} iterations each ====")
    print(f"Average Total Wins: {average_wins:.2f}")
    print(f"Average Win Rate: {average_win_rate:.2f}")

if __name__ == "__main__":
    main()
