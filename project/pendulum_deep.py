import numpy as np
from deep_agent import DQNAgent
import gymnasium as gym
import utils

if __name__ == '__main__':
    # training parameters
    load_checkpoint = False
    n_games = 10000

    # object creation
    env = gym.make('Pendulum-v1')
    agent = DQNAgent(gamma      = 0.90,
                     epsilon    = 1.0,
                     lr         = 0.001,
                     input_dims = (3,),
                     n_actions  = 1,
                     mem_size   = 50000,
                     eps_min    = 0.1,
                     batch_size = 64,
                     replace    = 1000,
                     eps_dec    = 1e-5,
                     dir        = 'models/',
                     name       = 'Pendulum-v1')
    
    # load checkpoint if evaluating pretrained model
    if load_checkpoint:
        agent.load_models()

    agent_name  = 'DQN_' + agent.name + '_lr_' + str(agent.lr)
    figure_name = 'plots/' + agent_name + '.png'

    best_score = -np.inf    # best total score achieved by an episode
    n_steps = 0             # total number of steps taken in all episodes
    scores_array, epsilon_array, steps_array = [], [], []
    for i in range(n_games):
        # start new episode
        score = 0
        observation, info = env.reset()
        terminated, truncated = False, False

        # begin training
        while not terminated and not truncated:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step([action])
            score += reward

            # learn if we're not just evaluating the net
            if not load_checkpoint:
                agent.store_memory(observation, action, reward,
                                       observation_, terminated or truncated)
                agent.learn()
            
            # update observation
            observation = observation_
            n_steps += 1

        scores_array.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores_array[-100:])
        #print(f'episode {i} score: {score} average score {avg_score} best score {best_score} epsilon {agent.epsilon} steps {n_steps}')

        # if average score of last 100 episodes is better than our previous
        # best, save progress in file
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        # print updates
        if i % 50 == 0:
            print(f'GAME {i}: Epsilon {agent.epsilon}, Avg. Score: {np.mean(scores_array[-50:])}')

        epsilon_array.append(agent.epsilon)

    utils.plot_model(steps_array, scores_array, epsilon_array, figure_name)