import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# make environment
env = gym.make('FrozenLake-v1')

n = 1000
win_pct = []
scores = []
for i in range(n):
    done = False
    # reset environment for each game
    obs = env.reset()
    score = 0
    while not done:
        # sample random action from action space
        action = env.action_space.sample()
        # take action
        obs, reward, done, truncated, info = env.step(action)
        score += reward

    scores.append(score)

    # record average score of last 10 games
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

# plot
plt.plot(win_pct)
plt.show()