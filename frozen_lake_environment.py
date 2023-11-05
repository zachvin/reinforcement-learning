# Zach Vincent
# based on code from this online course:
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_q_agent import Agent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01,
                   eps_delta=0.9999995, n_actions=4, n_states=16)

    scores = []
    win_pct_list = []
    n_games = 500000

    for i in range(n_games):
        # initialize environment
        done = False
        obs = env.reset()[0]
        score = 0

        # start episode
        while not done:
            # choose action based on epsilon-greedy
            action = agent.choose_action(obs)
            # get resulting state
            obs_, reward, done, truncated, info = env.step(action)
            # learn with resulting state
            agent.learn(obs, action, reward, obs_)
            score += reward

            obs = obs_

        # end episode
        # metrics and debug info
        scores.append(score)
        if i % 100 == 0:
            pct = np.mean(scores[-100:])
            win_pct_list.append(pct)
            if i % 1000 == 0:
                print(f'episode {i}\twin pct {pct:.2f}\tepsilon {agent.eps:.2f}')

    # plot results
    plt.plot(win_pct_list)
    plt.show()