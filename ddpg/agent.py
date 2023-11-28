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
    
class Agent:
    def __init__(self, state_space, action_space, action_high, action_low, gamma, tau, critic_lr, actor_lr, noise_std):
        self.mem = Buffer(state_space, action_space, 50000, 64)
        self.actor = Actor(state_space, action_high, action_space)
        self.critic = Critic(state_space, action_space)

        self.target_actor = Actor(state_space, action_high, action_space)
        self.target_critic = Critic(state_space, action_space)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_high = action_high
        self.action_low = action_low

        self.gamma = gamma
        self.tau = tau

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.02) * np.ones(1), theta=0.1)

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.float32)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_sate_batch = torch.tensor(next_state_batch, dtype=torch.float32)

        # Update critic
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            target_actions = self.target_actor(next_state_batch)
            target_values = self.target_critic(next_state_batch, target_actions)
            y = reward_batch + self.gamma * target_values
        critic_value = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(critic_value, y)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actions = self.actor(state_batch)
        critic_value = self.critic(state_batch, actions)
        actor_loss = -critic_value.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        def learn(self):
            record_range = min(self.mem.buffer_counter, self.mem.buffer_capacity)
            batch_indices = np.random.choice(record_range, self.mem.batch_size)

            state_batch = self.mem.state_buffer[batch_indices]
            action_batch = self.mem.action_buffer[batch_indices]
            reward_batch = self.mem.reward_buffer[batch_indices]
            next_state_batch = self.mem.next_state_buffer[batch_indices]

            self.update(state_batch, action_batch, reward_batch, next_state_batch)

        def policy(self, state):
            state = torch.tensor(state, dtype=torch.float32)
            sampled_actions = self.actor(state).detach().numpy()
            sample_actions += self.noise()
            legal_action = np.clip(sampled_actions, self.action_low, self.action_high)
            return np.squeeze(legal_action)
        
        def update_target(self):
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            # Save and load model weights
        def try_load_model_weights(self, model, file_path):
            if os.path.exists(file_path):
                model.load_state_dict(torch.load(file_path))

        def save_models(self, directory="./models"):
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(self.target_actor.state_dict(), os.path.join(directory, "target_actor.pth"))
            torch.save(self.target_critic.state_dict(), os.path.join(directory, "target_critic.pth"))

        def load_models(self, directory="./models"):
            self.try_load_model_weights(self.target_actor, os.path.join(directory, "target_actor.pth"))
            self.try_load_model_weights(self.target_critic, os.path.join(directory, "target_critic.pth"))