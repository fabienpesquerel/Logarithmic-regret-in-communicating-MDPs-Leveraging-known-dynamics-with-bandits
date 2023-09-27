import numpy as np
from learners.discreteMDPs.utils import *


def step_size(n):
    return np.log(n+1) / (n * np.log(2))


class MQlearning:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.Q = np.zeros((self.nS, self.nA))
        self.R = np.zeros((self.nS, self.nA))
        self.policy = np.zeros((self.nS, self.nA))
        self.alpha = np.zeros((self.nS, self.nA))
        self.nbr_pulls = np.zeros((self.nS, self.nA))
        self.gamma = 0.99  # Fairly close to 1

    def name(self):
        return "MQ-learning"

    def reset(self, inistate):
        self.t = 1
        self.R = np.zeros((self.nS, self.nA)) + 0.5
        self.nbr_pulls = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                self.Q[s, a] = 1. / (1 - self.gamma)  # Optimistic initialization: Crucial for good performances.
                self.policy[s, a] = 1. / self.nA
                self.alpha[s, a] = 1.

    def play(self, state):
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    def update(self, state, action, reward, observation):

        self.nbr_pulls[state, action] += 1
        n = self.nbr_pulls[state, action]

        self.R[state, action] = self.R[state, action] + (1./(np.sqrt(n)+1)) * (reward - self.R[state, action])
        self.Q[state, action] = self.Q[state, action] + 1./(np.sqrt(n)+1) * (
                    self.R[state, action] + self.gamma * max(self.Q[observation]) - self.Q[state, action])

        (u, arg) = allmax(self.Q[state])
        self.policy[state] = [1. / len(arg) if x in arg else 0 for x in range(self.nA)]
