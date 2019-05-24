import random
from typing import List


class ArmBandit:
    def __init__(self, p: float):
        self.p: float = p

    def run_trial(self) -> float:
        if random.random() < self.p:
            return 1.0
        else:
            return 0.0


class EGreedy:
    def __init__(self, epsilon: float = 0.1, bandits: List[ArmBandit] = None):
        self.epsilon: float = epsilon
        self.bandits: List[ArmBandit] = bandits
        self.bandits_count: int = len(self.bandits)
        self.bandit_pulls: List[int] = [0 for _ in range(self.bandits_count)]
        self.bandit_successes: List[int] = [0 for _ in range(self.bandits_count)]
        self.bandit_percentage_success: List[float] = [0 for _ in range(self.bandits_count)]

    def pull_bandit(self):
        if random.random() > self.epsilon:
            index: int = self.bandit_percentage_success.index(max(self.bandit_percentage_success))
        else:
            index: int = random.randrange(start=0, stop=self.bandits_count)

        reward: float = self.bandits[index].run_trial()
        self.bandit_percentage_success[index] = (self.bandit_successes[index] / (self.bandit_pulls[index] + 1)) + \
                                                (reward / (self.bandit_pulls[index] + 1))
        self.bandit_pulls[index] += 1
        self.bandit_successes[index] += reward


def main():
    # define constants
    simulations: int = 400
    epsilon: float = 0.05
    p1: float = 0.1
    p2: float = 0.15

    bandits = [ArmBandit(p1), ArmBandit(p2)]
    egreedy = EGreedy(epsilon=epsilon, bandits=bandits)
    for _ in range(simulations):
        egreedy.pull_bandit()
    print("[DONE] Executed {simulations} simulations.".format(simulations=simulations))
    print("\t[Bandits]: " + " ".join(["Bandit(p=%s)" % b.p for b in bandits]))
    print("\t\t[Bandit Pulls]:\t\t" + " ".join(["%s\t" % b for b in egreedy.bandit_pulls]))
    print("\t\t[Bandit Successes]:\t\t" + " ".join(["%s\t" % b for b in egreedy.bandit_successes]))
    print("\t\t[Bandit Percentage Successes]:\t\t" + " ".join(["%s\t" % b for b in egreedy.bandit_percentage_success]))


if __name__ == "__main__":
    main()
