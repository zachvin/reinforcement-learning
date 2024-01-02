# reinforcement-learning
### Introduction
This repository houses a series of files meant to introduce the topic of reinforcement learning and apply it to a continuous state space to control a pendulum in OpenAI's Gymnasium.

### Setup
Python 3.10 is used, but Python 3.8 also appears to work as intended.

Install libraries:

`pip install torch matplotlib numpy gymnasium gymnasium[classic-control]`

Navigate to `project/` and run:

`python3 pendulum_deep.py`

Any changes to hyperparameters or the number of games to be run can be edited directly in `pendulum_deep.py`. By default, as the average score improves, the program saves both networks to the `models/` folder, and upon completion, plots the training data in `plots/`.
