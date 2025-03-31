import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from gymnasium.wrappers import AtariPreprocessing, FrameStack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create and preprocess the environment
env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
env = FrameStack(env, 4)

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)
    
    def forward(self, x):
        x = x / 255.0  # Normalize pixel values
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_shape, action_size, replay_buffer, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=1e6):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = replay_buffer
        self.model = DQN(state_shape, action_size).to(device)
        self.target_model = DQN(state_shape, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.steps = 0
    
    def select_action(self, state):
        self.steps += 1
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps / self.epsilon_decay)
    
    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Initialize environment and agent
state_shape = env.observation_space.shape
action_size = env.action_space.n
replay_buffer = ReplayBuffer(capacity=100000)
agent = DQNAgent(state_shape, action_size, replay_buffer)

# Hyperparameters
batch_size = 32
update_target_every = 1000
episodes = 1000

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = np.sign(reward)  # Clip reward
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        agent.train(batch_size)
        agent.update_epsilon()
        
        if agent.steps % update_target_every == 0:
            agent.update_target_network()
    
    print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")