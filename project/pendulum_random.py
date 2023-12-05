# Zach Vincent

import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v1', render_mode='human')
observation, info = env.reset()

score = 0
n_games = 5
rewards = []
for i in range(n_games):
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()

env.close()