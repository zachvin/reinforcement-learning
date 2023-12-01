import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        # pos of last stored memory
        self.mem_cntr = 0

        # build all memories
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                          dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=uint8)

    # store observation into memory arrays
    def store_transition(self, state, action, reward, state_, done):
        # get index, overwrite memories after mem_size reached
        index = self.mem_cntr % self.mem_size

        # fill in each memory type
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # increment memory counter
        self.mem_cntr += 1

    # get a random memory
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        # choose index up to max_mem, with shape batch_size, and remove
        # index once it's been chosen
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # get and return memory
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
