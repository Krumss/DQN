import torch
import DQN
import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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