# reinforcement-learning
### Introduction
This repository houses a series of files meant to introduce the topic of reinforcement learning and apply it to a continuous state space to control a pendulum in OpenAI's Gymnasium.
See the project demonstration [here](https://youtu.be/0mDYZusld8Y) and learn more about my findings [here](https://zach-vincent.com/projects/pendulum.html).

| File name | File type | Description |
| --- | --- | --- |
| `pendulum_deep.py` | Main file | Main loop; constructs agent object, observes and manipulates environment |
| `deep_agent.py` | Agent class | Constructs network object, contains functions for choosing actions, storing memories, and learning |
| `deep_network.py` | Network class | Builds Torch neural network and contains functions for saving network weights for future evaluation |
| `replay_buffer.py` | Memory buffer class | Handles memory control, contains functions for appending to and sampling from memory |
| `utils.py` | Plotting tools | Modularizes plotting learning rates |



### Setup
Python 3.10 is used, but Python 3.8 also appears to work as intended.

Install libraries:

`pip install torch matplotlib numpy gymnasium gymnasium[classic-control]`

Navigate to `project/` and run:

`python3 pendulum_deep.py`

Any changes to hyperparameters or the number of games to be run can be edited directly in `pendulum_deep.py`. By default, as the average score improves, the program saves both networks to the `models/` folder, and upon completion, plots the training data in `plots/`.
