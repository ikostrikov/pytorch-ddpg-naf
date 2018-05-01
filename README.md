### Description
Reimplementation of [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748v1.pdf) and [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf).

Contributions are welcome. If you know how to make it more stable, don't hesitate to send a pull request.

### Run
Use the default hyperparameters.

#### For NAF:

```
python main.py --algo NAF --env-name HalfCheetah-v2
```
#### For DDPG

```
python main.py --algo DDPG --env-name HalfCheetah-v2
```
