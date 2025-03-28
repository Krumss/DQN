import torch
import torch.nn as nn

class QNetwork(nn.Module):
    #Breakout states: 3d numpy (210, 160, 3) 210 height, 160 width, 3 color channels RGB
    def __init__(self, n_states, n_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_states[2], 32, 8, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=0),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(n_states[1] * n_states[0] * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.conv(x)