import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from buffer import Buffer
from noise import OUActionNoise
from utils import get_gpu
import os

device = get_gpu()


class Actor(nn.Module):
    def __init__(self, state_space, action_high, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.lin_vel = nn.Linear(256, 1)
        self.ang_vel = nn.Linear(256, 1)
        self.action_high = action_high
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = T.relu(self.fc1(x))
        x = T.dropout(x, 0.1, train=self.training)
        x = T.relu(self.fc2(x))
        x = T.dropout(x, 0.1, train=self.training)
        x = T.relu(self.fc3(x))
        lin_vel = T.sigmoid(self.lin_vel(x))
        ang_vel = T.tanh(self.ang_vel(x))
        return T.cat((lin_vel, ang_vel), dim=1) * self.action_high
    
class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.state_fc = nn.Linear(state_space, 256)
        self.action_fc = nn.Linear(action_space, 256)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


    def forward(self, state, action):
        state_out = T.relu(self.state_fc(state))
        action_out = T.relu(self.action_fc(action))
        concat = T.cat([state_out, action_out], dim=1)
        x = T.relu(self.fc1(concat))
        x = T.relu(self.fc2(x))
        return self.output(x)
    
class Agent:
    def __init__(self, state_space, action_space, action_high, action_low, gamma, tau, critic_lr, actor_lr, noise_std):
        self.mem = Buffer(state_space, action_space, 50000, 64)
        self.actor = Actor(state_space, action_high, action_space).to(device)
        self.critic = Critic(state_space, action_space).to(device)

        self.target_actor = Actor(state_space, action_high, action_space).to(device)
        self.target_critic = Critic(state_space, action_space).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_high = action_high
        self.action_low = action_low

        self.gamma = gamma
        self.tau = tau

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.02) * np.ones(1), theta=0.1)

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # state_batch = T.tensor(state_batch, dtype=T.float32)
        # action_batch = T.tensor(action_batch, dtype=T.float32)
        # reward_batch = T.tensor(reward_batch, dtype=T.float32)
        # next_sate_batch = T.tensor(next_state_batch, dtype=T.float32)

        # Update critic
        self.critic_optimizer.zero_grad()
        with T.no_grad():
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

        state_batch = T.tensor(self.mem.state_buffer[batch_indices], dtype=T.float32).to(device)
        action_batch = T.tensor(self.mem.action_buffer[batch_indices], dtype=T.float32).to(device)
        reward_batch = T.tensor(self.mem.reward_buffer[batch_indices], dtype=T.float32).to(device)
        next_state_batch = T.tensor(self.mem.next_state_buffer[batch_indices], dtype=T.float32).to(device)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def policy(self, state):
        state = state.clone().detach().to(device)
        # state = T.tensor(state, dtype=T.float32)
        noise = self.noise()
        sampled_actions = self.actor(state).detach().cpu().numpy() + noise
        legal_action = np.clip(sampled_actions, self.action_low, self.action_high)
        return (legal_action)
    
    def update_target(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # Save and load model weights
    def try_load_model_weights(self, model, file_path):
        if os.path.exists(file_path):
            model.load_state_dict(T.load(file_path))

    def save_models(self, directory="./models"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        T.save(self.target_actor.state_dict(), os.path.join(directory, "target_actor.pth"))
        T.save(self.target_critic.state_dict(), os.path.join(directory, "target_critic.pth"))

    def load_models(self, directory="./models"):
        self.try_load_model_weights(self.target_actor, os.path.join(directory, "target_actor.pth"))
        self.try_load_model_weights(self.target_critic, os.path.join(directory, "target_critic.pth"))