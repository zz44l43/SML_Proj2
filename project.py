import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# pyplot = plt.pyplot


class MAB(ABC):
    """
    Abstract class that represents a multi-armed bandit (MAB)
    """

    def __init__(self, narms, epsilon, Q0, reward_distribution="Gaussian"):
        self.narms = narms
        self.epsilon = epsilon
        self.arm_values = np.random.normal(0, 1, narms) if reward_distribution is "Gaussian" else np.random.rand(narms)
        self.best_action = np.argmax(self.arm_values)
        self.pulls = np.zeros(narms)
        self.est_values = np.empty(narms)
        self.est_values.fill(Q0)
        self.reward_distribution = reward_distribution

    def get_reward(self, action):
        if self.reward_distribution is "Gaussian":
            return self.get_Gaussian(action)
        return self.get_Bernoulli(action)

    def get_Gaussian(self, action):
        return np.random.normal(self.arm_values[action], 1)

    def get_Bernoulli(self, action):
        return np.random.binomial(1, self.arm_values[action])

    @abstractmethod
    def play(self, tround, context):
        """
        Play a round

        Arguments
        =========
        tround : int
            positive integer identifying the round

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to the arms

        Returns
        =======
        arm : int
            the positive integer arm id for this round
        """

    @abstractmethod
    def update(self, arm, reward, context):
        """
        Updates the internal state of the MAB after a play

        Arguments
        =========
        arm : int
            a positive integer arm id in {1, ..., self.narms}

        reward : float
            reward received from arm

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to arms
        """


class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    epsilon : float
        explore probability

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, epsilon, Q0=np.inf, reward_distribution="Gaussian"):
        super().__init__(narms, epsilon, Q0, reward_distribution)

    #     def get_reward(self, action):
    #         return
    # #         return 1 if np.random.normal(0,1) > 0.2 else 0
    # #         return np.random.normal(self.arm_values[action],1)
    #         return self.arm_values[action] + np.random.normal(0,1)

    def play(self, tround, context=None):
        num_random = np.random.random()
        if self.epsilon > num_random:
            return np.random.randint(self.narms)
        else:
            return np.argmax(self.est_values)
            choice = np.random.choice(np.where(self.est_values == self.est_values.max())[0])
            return choice

    def update(self, arm, reward, context=None):
        self.pulls[arm] += 1
        self.est_values[arm] += (reward - self.est_values[arm]) / self.pulls[arm]


def experiment(eps, epsilon, Q0):
    rewards_all_eps = np.zeros(pulls)
    count_optimal = 0
    for ep in range(eps):
        reward_all = []
        reward_all_avg = np.zeros(pulls)
        actions = []
        bandit_ep0 = EpsGreedy(narms, epsilon, Q0)
        cumulative_reward = 0
        cumulative_reward_all = np.zeros(pulls)
        for i in range(pulls):
            action = bandit_ep0.play(i)
            actions.append(action)
            reward = bandit_ep0.get_reward(action)
            cumulative_reward += reward
            bandit_ep0.update(action, reward)
            reward_all.append(reward)
            prev_reward = reward_all_avg[i - 1] if i > 0 else 0
            reward_all_avg[i] = prev_reward + (reward - prev_reward) / (i + 1)
            cumulative_reward_all[i] = cumulative_reward / (i + 1)
            count_optimal += 1 if bandit_ep0.best_action == action else 0
        rewards_all_eps += cumulative_reward_all
    return (rewards_all_eps / eps, count_optimal)


data_set = np.loadtxt("dataset.txt", dtype="int")
arm_played = data_set[:, :1]
rewards_received = data_set[:, 1:2]
context = data_set[:, 2:]

pulls = 500
narms = 10
eps = 1000

ep0, ep0_optimal_count = experiment(eps, 0, 0)
ep01, ep01_optimal_count = experiment(eps, 0.1, 0)
ep001, ep001_optimal_count = experiment(eps, 0.01, 0)

plt.plot(ep0, label="epsilon=0")
plt.plot(ep01, label="epsilon=0.1")
plt.plot(ep001, label="epsilon=0.01")
plt.ylim(0, 2.5)
plt.legend()
plt.show()

print("123")
