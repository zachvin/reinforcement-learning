# Zach Vincent
# based on code from this online course by Phil Tabor:
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/

# The purpose of the replay buffer is to randomize the data from which the
# agent learns. In doing so, the agent does not learn using correlations between
# consecutive frames. In addition, the agent will not get stuck in local minima
# as easily because it will not repeat the same decisions in each episode.
# This improvement is as seen in the "Human-level control through deep
# reinforcement learning" paper.

import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        '''
        Build a ReplayBuffer object.
        Params:
            max_size (int): maximum number of memories to be stored in buffer
            input_shape (int):
        '''
        # how big the replay buffer should be
        self.mem_size = max_size

        # pos of last stored memory
        self.index = 0

        # build all memories
        self.mem_state      = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.mem_new_state  = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.mem_action     = np.zeros(self.mem_size, dtype=np.int64)
        self.mem_reward     = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_terminal   = np.zeros(self.mem_size, dtype=np.bool_)

    # store observation into memory arrays
    def append_buffer(self, state, action, reward, state_, done):
        # get index, overwrite memories after mem_size reached
        index = self.index % self.mem_size

        # fill in each memory type
        self.mem_state[index]       = state
        self.mem_new_state[index]   = state_
        self.mem_action[index]      = action
        self.mem_reward[index]      = reward
        self.mem_terminal[index]    = done

        # increment index
        self.index += 1

    # get a random memory
    def sample_buffer(self, batch_size):
        # memory can be sampled before and after it has been filled
        max_mem = min(self.index, self.mem_size)

        # get batch_size number of indices
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # get memory as np array
        states      = self.mem_state[batch]
        states_     = self.mem_new_state[batch]
        actions     = self.mem_action[batch]
        rewards     = self.mem_reward[batch]
        terminals   = self.mem_terminal[batch]

        return states, actions, rewards, states_, terminals