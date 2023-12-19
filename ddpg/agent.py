import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ddpg.buffer import Buffer
from ddpg.noise import OUActionNoise
from ddpg.utils import get_gpu, fanin_init
import os

ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 1. # rad/s
var_v = ACTION_V_MAX * 0.30
var_w = ACTION_W_MAX * 2 * 0.15

class Actor(nn.Module):
    def __init__(self, state_space, action_high, action_space, action_limit_v=0.22, action_limit_w=1.0):
        super(Actor, self).__init__()
        self.EPS = 0.003
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        self.fa1 = nn.Linear(state_space, 512)
        self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())

        self.fa2 = nn.Linear(512, 512)
        self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())

        self.fa3 = nn.Linear(512, 2)
        self.fa3.weight.data.uniform_(-self.EPS, self.EPS)

    def forward(self, state):
        x = T.relu(self.fa1(state))
        # x = T.dropout(x, 0.1, train=self.training)
        x = T.relu(self.fa2(x))
        # x = T.dropout(x, 0.1, train=self.training)
        # x = T.relu(self.fc3(x))
        # lin_vel = T.sigmoid(self.lin_vel(x))
        # ang_vel = T.tanh(self.ang_vel(x))
        # return T.cat((lin_vel, ang_vel), dim=1) * self.action_high
        action = self.fa3(x)
        if state.shape == T.Size([14]):
            action[0] = T.sigmoid(action[0]) * self.action_limit_v
            action[1] = T.tanh(action[1]) * self.action_limit_v
        else:
            action[:, 0] = T.sigmoid(action[:, 0]) * self.action_limit_w
        return action
    
class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.EPS = 0.003
        self.fc1 = nn.Linear(state_space, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fa1 = nn.Linear(action_space, 256)
        self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())

        self.fca1 = nn.Linear(512, 512)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fca2 = nn.Linear(512, 1)
        self.fca2.weight.data.uniform_(-self.EPS, self.EPS)


    def forward(self, state, action):
        state_out = T.relu(self.fc1(state))
        action_out = T.relu(self.fa1(action))
        concat = T.cat([state_out, action_out], dim=1)
        x = T.relu(self.fca1(concat))
        x = T.relu(self.fca2(x))
        return x
    
class Agent:
    def __init__(self, state_space, action_space, action_high, action_low, gamma, tau, critic_lr, actor_lr, noise_std, batch_size=128):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.memory = Buffer(state_space, action_space, 150000, 128)
        self.actor = Actor(state_space, action_high, action_space).to(self.device)
        self.critic = Critic(state_space, action_space).to(self.device)

        self.target_actor = Actor(state_space, action_high, action_space).to(self.device)
        self.target_critic = Critic(state_space, action_space).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_high = action_high
        self.action_low = action_low

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.01) * np.ones(1), theta=0.1)
        self.training_mode=True

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def get_exploitation_action(self, state):
        self.actor.eval()  # Put the actor in evaluation mode
        state = T.tensor(state, dtype=T.float32).to(self.device)
        action = self.actor(state).detach()
        self.actor.train()  # Return to training mode
        return action.cpu().numpy()
    
    def get_exploration_action(self, state):
        self.actor.eval()
        state = T.tensor(state, dtype=T.float32).to(self.device)
        action = self.actor(state).detach()
        self.actor.train()

        noise = self.noise()
        action = action.cpu().numpy() + noise
        return np.clip(action, self.action_low, self.action_high)

    # def update(self, state_batch, action_batch, reward_batch, next_state_batch):
    #     # state_batch = T.tensor(state_batch, dtype=T.float32)
    #     # action_batch = T.tensor(action_batch, dtype=T.float32)
    #     # reward_batch = T.tensor(reward_batch, dtype=T.float32)
    #     # next_sate_batch = T.tensor(next_state_batch, dtype=T.float32)

    #     # Update critic
    #     self.critic_optimizer.zero_grad()
    #     with T.no_grad():
    #         target_actions = self.target_actor(next_state_batch)
    #         target_values = self.target_critic(next_state_batch, target_actions)
    #         y = reward_batch + self.gamma * target_values
    #     critic_value = self.critic(state_batch, action_batch)
    #     critic_loss = nn.MSELoss()(critic_value, y)
    #     critic_loss.backward()
    #     self.critic_optimizer.step()

    #     # Update actor
    #     self.actor_optimizer.zero_grad()
    #     actions = self.actor(state_batch)
    #     critic_value = self.critic(state_batch, actions)
    #     actor_loss = -critic_value.mean()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()

    def learn(self):
        if self.memory.buffer_counter < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch = \
                self.memory.sample_buffer(self.batch_size)
        self.optimize(state_batch, action_batch, reward_batch, next_state_batch)

    def policy(self, state, add_noise=True):
        state = state.clone().detach().to(self.device)
        action = self.actor(state).detach()
        if add_noise:
            noise = self.noise()
            action = action.cpu() + noise
            # sampled_actions = self.actor(state).detach().cpu().numpy() + noise
        else:
            action = action.cpu().numpy()
        legal_action = np.clip(action.numpy(), self.action_low, self.action_high)
        return (legal_action)
    
    def choose_action(self, observation):
        # Normaliza a observação
        normalized_observation = observation / np.linalg.norm(observation)
        # Converte a observação para tensor e envia para o dispositivo
        state = T.tensor([normalized_observation], dtype=T.float32).to(self.device)

        # Obter ação
        if self.training_mode:
            action = self.get_exploration_action(state)
        else:
            action = self.get_exploitation_action(state)

        print(f"Choosen action: {action}")

        
        action = action.squeeze()

        action[0] = np.clip(np.random.normal(action[0], var_v), 0., ACTION_V_MAX)
        action[1] = np.clip(np.random.normal(action[1], var_w), -ACTION_W_MAX, ACTION_W_MAX)
        print(f"Choosen action (after clipping): {action[0]} - {action[1]}")

        return action
    
    def remember(self, state, action, reward, new_state, done):
        # guardar acoes e consequencias no buffer de memoria
        self.memory.store_transition(state, action, reward, new_state)

    def update_target(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def optimize(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Update critic network
        self.critic_optimizer.zero_grad()
        with T.no_grad():
            target_actions = self.target_actor(next_state_batch)
            target_values = self.target_critic(next_state_batch, target_actions)
            y = reward_batch + self.gamma * target_values
        critic_value = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(critic_value, y)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        actions = self.actor(state_batch)
        critic_value = self.critic(state_batch, actions)
        actor_loss = -T.mean(critic_value)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Atualizando as redes-alvo
        self.update_target()

        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)
        # self.update(state_batch, action_batch, reward_batch, next_state_batch)


    # Save and load model weights
    def try_load_model_weights(self, model, file_path):
        if os.path.exists(file_path):
            model.load_state_dict(T.load(file_path))

    def save_models(self, directory="./ddpg/models"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        T.save(self.target_actor.state_dict(), os.path.join(directory, "target_actor.pth"))
        T.save(self.target_critic.state_dict(), os.path.join(directory, "target_critic.pth"))

    def load_models(self, directory="./ddpg./models"):
        self.try_load_model_weights(self.target_actor, os.path.join(directory, "target_actor.pth"))
        self.try_load_model_weights(self.target_critic, os.path.join(directory, "target_critic.pth"))
