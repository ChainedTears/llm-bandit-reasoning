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

class RandomChoice:
    def __init__(self, n_arms):
        self.n_arms = n_arms
    
    def select_arm(self):
        return secrets.randbelow(self.n_arms)

    def update(self, chosen_arm, reward):
        # Random choice does not update anything
        pass

def main():
    random_agent = RandomChoice(2)
    results_log = []
    max_iterations = 100

    for i in range(max_iterations):
        chosen_arm = random_agent.select_arm()  # 0 or 1
        machine = chosen_arm + 1
        result = bandit_simulation(machine)
        reward = 1 if result == "won" else 0
        random_agent.update(chosen_arm, reward)
        results_log.append((machine, result))
        print(f"Iteration {i+1}: Chose Slot Machine {machine} -> {result}")

    total_wins = sum(1 for _, r in results_log if r == "won")
    print(f"Total wins: {total_wins} out of {max_iterations} plays.")
    print(f"Win rate: {total_wins / max_iterations:.2f}")

if __name__ == "__main__":
    main()
