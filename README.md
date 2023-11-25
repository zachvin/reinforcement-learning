# reinforcement-learning
### Introduction
This repository houses a series of files meant to introduce the topic of reinforcement learning and eventually apply it to a continuous state space for the purpose of flying a simulated plane with vision input.

### Setup
There is an included `requirements.txt` file that outlines all the required packages. Python 3.8 is used for its compatibility with Gymnasium and PyTorch.

### Usage with Gazebo
Load the Gazebo example world `zephyr_runway.sdf` by running `gz sim -v4 -r zephyr_runway.sdf` in a terminal window.

In another terminal window, start up the ArduPilot SITL integration with `sim_vehicle.py -v ArduPlane -f gazebo-zephyr --model JSON --map --console`. In this window, run the following:

`mode fbwa`

`FBWA> arm throttle`

`FBWA> rc 3 1800`

`FBWA> mode circle`

This sets the zephyr model to take off and fly in a straight line. In the future, the deep learning agent will send `rc <channel> <pwm_value>` instructions in response to the camera and position inputs.


### Progress
Included are three main agents: one random, one using a Q table, and one using a simple deep Q neural net. These agents act as players in two different environments: Frozen Lake and CartPole. Frozen Lake has the player attempt to navigate holes to get to the goal location while occasionally slipping on the ice. The CartPole simulation has the player move a cart left and right in order to balance a vertical pole on top.

The random agent performs poorly with ~10% win rate. (Frozen Lake)

The Q table agent performs moderately better with ~40% win rate. (Frozen Lake)

The deep Q neural net agent performs moderately well with a relatively high average score, but fails to move past local minima as epsilon decreases to a minimum. This shows that effective learning cannot be achieved by tacking a neural net onto a Q agent. (CartPole)
