# Zach Vincent
# based on code from this online course:
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/

import gymnasium as gym
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
#from util import plot_learning_curve

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        # initialize
        super(LinearDeepQNetwork, self).__init__()

        # build layers
        # input_dims is number of things we can observe about environment
        print(f'input dims: {input_dims}:128')
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # loss function
        self.loss = nn.MSELoss()

        # send data
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # propagate data forward
        print(state)
        print('separator')
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        # returns actions tensor
        return actions

class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-5, eps_min=0.01):

        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = gamma
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            # get numpy array out of returned tensor
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        # zero the gradient
        self.Q.optimizer.zero_grad()

        # turn data into tensors
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(states_).max()

        # corresponds to best action in next state (q_next)
        q_target = reward + self.gamma*q_next

        # we want q_pred to move to q_next, which is our target state
        # q_next reward is represented by q_target
        # thus our loss is measured between q_pred and q_target
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(lr=0.0001, input_dims=4,
                  n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()[0]

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_

        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'episode {i}\tscore {score:.1f}\tavg score {avg_score:.1f}\tepsilon {agent.epsilon:.2f}')

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, '')