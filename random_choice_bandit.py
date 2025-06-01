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

class RandomChoice:
    def __init__(self, n_arms):
        self.n_arms = n_arms
    
    def select_arm(self):
        return random.randrange(self.n_arms)

    def update(self, chosen_arm, reward):
        # Random choice does not update anything
        pass

def main():
    simulations = 500
    iterations_per_simulation = 25
    all_simulation_wins = []

    for sim in range(simulations):
        random_agent = RandomChoice(2)
        results_log = []

        for i in range(iterations_per_simulation):
            chosen_arm = random_agent.select_arm()  # 0 or 1
            machine = chosen_arm + 1
            result = bandit_simulation(machine)
            reward = 1 if result == "won" else 0
            random_agent.update(chosen_arm, reward)
            results_log.append((machine, result))
            print(f"Simulation {sim+1}, Iteration {i+1}: Chose Slot Machine {machine} -> {result}")

        total_wins = sum(1 for _, r in results_log if r == "won")
        all_simulation_wins.append(total_wins)

    average_wins = sum(all_simulation_wins) / simulations
    average_win_rate = average_wins / iterations_per_simulation
    print(f"\n==== Summary after {simulations} simulations of {iterations_per_simulation} iterations each ====")
    print(f"Average Total Wins: {average_wins:.2f}")
    print(f"Average Win Rate: {average_win_rate:.2f}")

if __name__ == "__main__":
    main()
