import numpy as np
from deep_agent import DQNAgent
import gymnasium as gym
import utils

if __name__ == '__main__':
    # training parameters
    load_checkpoint = False
    n_games = 15000

    # object creation
    env = gym.make('Pendulum-v1')
    agent = DQNAgent(gamma      = 0.9,
                     epsilon    = 1.0,
                     lr         = 0.001,
                     input_dims = (3,), #env.observation_space.shape,
                     n_actions  = 10, #env.action_space.shape[0],
                     mem_size   = 50000,
                     eps_min    = 0.1,
                     batch_size = 32,
                     replace    = 1000,
                     eps_dec    = 1.5e-6,
                     dir        = 'models/',
                     name       = 'Pendulum-v1',
                     env        = env)
    
    # load checkpoint if evaluating pretrained model
    if load_checkpoint:
        agent.load_models()

    agent_name  = 'DQN_' + agent.name + '_lr_' + str(agent.lr)
    figure_name = 'plots/' + agent_name + '.png'

    best_score = -np.inf    # best total score achieved by an episode
    n_epochs = 0             # total number of steps taken in all episodes
    scores_array, epsilon_array, epochs_array = [], [], []
    avgs_array = []
    for i in range(n_games):
        # start new episode
        score = 0
        
        # observation is numpy.ndarray with shape (3,)
        observation, info = env.reset()
        terminated, truncated = False, False

        # begin training
        while not terminated and not truncated:
            action = agent.choose_action(observation)

            observation_, reward, terminated, truncated, info = env.step([(action-5)*.2])
            score += reward

            # learn if we're not just evaluating the net
            if not load_checkpoint:
                agent.store_memory(observation, action, reward,
                                       observation_, terminated or truncated)
                agent.learn()
            
            # update observation
            observation = observation_
        
        n_epochs += 1

        
        scores_array.append(score)
        avg_score = np.mean(scores_array[-100:])
        #print(f'episode {i} score: {score} average score {avg_score} best score {best_score} epsilon {agent.epsilon} steps {n_steps}')

        # if average score of last 100 episodes is better than our previous
        # best, save progress in file
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        avgs_array.append(avg_score)
        epochs_array.append(n_epochs)
        epsilon_array.append(agent.epsilon)

        # print updates
        print(f'GAME: {i},\tEpsilon: {agent.epsilon:7.2f}, Score: {score:7.2f} Avg. Score: {np.mean(scores_array[-50:]):7.2f}')


    utils.plot_model(epochs_array, scores_array, avgs_array, epsilon_array, figure_name)