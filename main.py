import argparse
import math
from collections import namedtuple
from itertools import count

import gym
import numpy as np
from gym import wrappers

import torch
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
args = parser.parse_args()

env_name = 'Pendulum-v0'
env = NormalizedActions(gym.make(env_name))

env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(env_name), force=True)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = NAF(args.gamma, args.tau, args.hidden_size,
            env.observation_space.shape[0], env.action_space)
memory = ReplayMemory(args.replay_size)
ounoise = OUNoise(env.action_space.shape[0])

rewards = []
for i_episode in range(args.num_episodes):
    if i_episode < args.num_episodes // 2:
        state = torch.Tensor([env.reset()])
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()
        episode_reward = 0
        for t in range(args.num_steps):
            action = agent.select_action(state, ounoise)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += reward

            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            if i_episode % 10 == 0:
                env.render()

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            if len(memory) > args.batch_size * 5:
                for _ in range(args.updates_per_step):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))

                    agent.update_parameters(batch)

            if done:
                break
        rewards.append(episode_reward)
    else:
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        for t in range(args.num_steps):
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += reward

            next_state = torch.Tensor([next_state])

            if i_episode % 10 == 0:
                env.render()

            state = next_state
            if done:
                break

        rewards.append(episode_reward)
    print("Episode: {}, noise: {}, reward: {}, average reward: {}".format(i_episode, ounoise.scale,
                                                                          rewards[-1], np.mean(rewards[-100:])))

env.close()
