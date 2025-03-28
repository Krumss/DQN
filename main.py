import gymnasium as gym
import ale_py
import DQN
import torch
import random

EPS_HIGHEST = 1
EPS_LOWEST = 0.01
eps = EPS_HIGHEST

gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', render_mode="human")
obs, info = env.reset()

policy_network = DQN.QNetwork(n_states=obs.shape, n_actions=env.action_space.n)
target_network = DQN.QNetwork(n_states=obs.shape, n_actions=env.action_space.n)

for episode in range(1000):
    state = env.reset()
    terminated = False
    while not terminated:
        if random.random() < eps:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        else:
            obs, reward, terminated, truncated, info = env.step(policy_network.forward(obs))


for _ in range(100):
    env.render()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

env.close()