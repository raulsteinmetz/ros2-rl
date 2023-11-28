import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from buffer import Buffer
from noise import OUActionNoise
import os


class Actor(nn.Module):
    def __init__(self, state_space, action_high, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_space)
        self.lin_vel = nn.Linear(256, 1)
        self.ang_vel = nn.Linear(256, 1)
        self.action_high = action_high

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, 0.1, train=self.training)
        x = torch.relu(self.fc2(x))
        x = torch.dropout(x, 0.1, train=self.training)
        x = torch.relu(self.fc3(x))
        lin_vel = torch.sigmoid(self.lin_vel(x))
        ang_vel = torch.tanh(self.ang_vel(x))
        return torch.cat((lin_vel, ang_vel), dim=1) * self.action_high
    
class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.state_fc = nn.Linear(state_space, 256)
        self.action_fc = nn.Linear(action_space, 256)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, state, action):
        state_out = torch.relu(self.state_fc(state))
        action_out = torch.relu(self.action_fc(action))
        concat = torch.cat([state_out, action_out], dims=1)
        x = torch.relu(self.fc1(concat))
        x = torch.relu(self.fc2(x))
        return self.output(x)
    
# Will continue the agent after this commit