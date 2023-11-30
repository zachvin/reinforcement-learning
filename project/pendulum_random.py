import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v1')
observation, info = env.reset()

score = 0
rewards = []
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward
    rewards.append(score)

    print(f'Reward: {reward}')
    if terminated or truncated:
        observation, info = env.reset()

env.close()

plt.plot(range(100), rewards)
plt.show()