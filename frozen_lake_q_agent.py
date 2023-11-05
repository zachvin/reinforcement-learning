# Zach Vincent
# based on code from this online course:
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/

import numpy as np

class Agent:
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_delta):
        self.lr         = lr
        self.gamma      = gamma
        self.n_actions  = n_actions
        self.n_states   = n_states
        self.eps_min    = eps_end
        self.eps        = eps_start
        self.eps_delta  = eps_delta

        self.Q = dict()
        self.init_Q()

    def init_Q(self):
        # initialize Q table with 0
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.Q[(s,a)] = 0

    def choose_action(self, s):
        # choose random action
        if np.random.random() < self.eps:
            a = np.random.choice([i for i in range(self.n_actions)])
        else:
            # make array of all possible actions
            actions = np.array([self.Q[(s,a)] for a in range(self.n_actions)])
            # action is index with largest value
            a = np.argmax(actions)
        
        return a

    def decrement_epsilon(self):
        self.eps = self.eps * self.eps_delta if self.eps > self.eps_min \
                   else self.eps_min

    def learn(self, s, a, r, s_):
        # find best action for state s
        actions = np.array([self.Q[(s, a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions)

        # Bellman eqn
        # update Q table with new value for state s
        self.Q[(s, a)] += self.lr*(r +
                                self.gamma*self.Q[(s_,a_max)] -
                                self.Q[(s,a)])

        # decrement epsilon
        self.decrement_epsilon()