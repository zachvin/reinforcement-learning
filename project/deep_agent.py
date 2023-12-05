# Zach Vincent
# based on code from this online course by Phil Tabor:
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/

import numpy as np
import torch as T
from deep_network import DeepQNetwork
from replay_buffer import ReplayBuffer
import time

class DQNAgent():
    def __init__(self, gamma:float, epsilon:float, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, name=None, dir='tmp/dqn', env=None):
                 
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.lr             = lr
        self.n_actions      = n_actions
        self.input_dims     = input_dims
        self.batch_size     = batch_size
        self.eps_min        = eps_min
        self.eps_dec        = eps_dec
        self.replace        = replace
        self.name           = name
        self.dir            = dir
        self.env            = env

        assert self.env is not None, 'No environment specified in DQNAgent instantiation'

        self.learn_counter  = 0

        # build memory buffer
        self.buffer = ReplayBuffer(mem_size, input_dims)

        # By creating two neural nets, we use one to evaluate the Q values
        # of state-action pairs and one to select the optimal value. Otherwise,
        # the NN will try to select optimal Q values while simultaneously
        # changing them.
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims   = self.input_dims[0],
                                   name         = self.name + '_deep_q_eval',
                                   dir          = self.dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims   = self.input_dims[0],
                                   name         = self.name + '_deep_q_next',
                                   dir          = self.dir)

    def choose_action(self, observation, evaluate):
        # choose learned action
        if evaluate or np.random.random() > self.epsilon:
            # state is tensor object with shape (3,)
            state = T.tensor(observation, dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)

            # argmax returns tensor
            action = T.argmax(actions).item()

        # choose random action
        else:
            #action = np.random.uniform(self.env.action_space.low, self.env.action_space.high)
            action = np.random.choice([_ for _ in range(10)])

        return action

    def store_memory(self, state, action, reward, state_, done):
        self.buffer.append_buffer(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, state_, done = \
                                self.buffer.sample_buffer(self.batch_size)

        # turn each item from memory buffer into a tensor
        # lowercase tensor preserves datatype of numpy array
        states  = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones   = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        # choose when to learn
        if self.learn_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
    
    def save_models(self):
        self.q_eval.save_checkpoint('q_eval')
        self.q_next.save_checkpoint('q_next')

    def load_models(self):
        self.q_eval.load_checkpoint('eval')
        self.q_next.load_checkpoint('next')

    def learn(self):
        # wait until batch is filled up
        if self.buffer.index < self.batch_size:
            return

        # always zero gradients on optimizer
        self.q_eval.optimizer.zero_grad()

        # choose actions and update Q values separately
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        # here we use indices to make sure output has shape batch_size
        indices = np.arange(self.batch_size)

        # gives predicted action values for batch of states
        # looking for value of actions taken in batch of states
        q_pred = self.q_eval.forward(states.reshape(self.batch_size,3))[indices, actions]

        # find value for maximal actions for set of states
        # what are max values of next states, and move action predictions toward
        # that value
        q_next = self.q_next.forward(states_.reshape(self.batch_size,3)).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        # calculate loss and back propagate
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        
        self.q_eval.optimizer.step()
        self.learn_counter += 1

        self.decrement_epsilon()