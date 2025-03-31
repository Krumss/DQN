import gymnasium as gym
import ale_py
import torch
import numpy as np
import torch.nn.functional as F
import random
import DQN
import ReplayBuffer

EPS_HIGHEST = 1
EPS_LOWEST = 0.001
eps = EPS_HIGHEST
EPS_DECAY = 1e6
SAMPLE_BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 1000
UPDATE_TARGET_C_STEPS = 1000
GAMMA = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(state):
    #self.steps += 1
    if random.random() < eps:
        return env.action_space.sample()
    else:
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_network(state)
        return q_values.argmax().item()
    
def update_eps(steps):
    eps = EPS_LOWEST + (EPS_HIGHEST - EPS_LOWEST) * np.exp(-1. * steps / EPS_DECAY)

def update_target_network():
    target_network.load_state_dict(policy_network.state_dict())

def train(batch_size):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    
    current_q = policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q = target_network(next_states).max(1)[0].detach()
    target_q = rewards + (1 - dones) * GAMMA * next_q
    
    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5')
curr_state, info = env.reset()

policy_network = DQN.QNetwork(input_shape=curr_state.shape, n_actions=env.action_space.n)
target_network = DQN.QNetwork(input_shape=curr_state.shape, n_actions=env.action_space.n)
replay_buffer = ReplayBuffer(capacity=10000)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-4)

# Training loop
for episode in range(EPISODES):
    steps = 0
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = np.sign(reward)  # Clip reward
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        train(SAMPLE_BATCH_SIZE)
        update_eps(steps)
        
        if steps % UPDATE_TARGET_C_STEPS == 0:
            update_target_network()
    
    print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {eps:.4f}")