import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + np.sqrt(self.dt) * self.std_dev * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 n_atoms, v_min, v_max, name, chkpt_dir='tmp/d4pg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.n_atoms = n_atoms  # number of distribtuion bins
        self.v_min = v_min      # min distribution value
        self.v_max = v_max      # max distribution value
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_d4pg')

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, self.n_atoms)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=1e-5)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cuda:0')
        self.to(self.device)

    def forward(self, state, action):
        state = T.tensor(state, dtype=T.float).to(self.device) if isinstance(state, np.ndarray) else state
        action = T.tensor(action, dtype=T.float).to(self.device) if isinstance(action, np.ndarray) else action
        state_action = T.cat([state, action], dim=1)
        x = F.leaky_relu(self.fc1(state_action), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        q_values = self.q(x)

        # Applying softmax to q_values to get a probability distribution
        q_values = F.softmax(q_values, dim=1)

        return q_values

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
            n_actions, name, min_action, max_action, chkpt_dir='tmp/d4pg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.min_action = min_action
        self.max_action = max_action
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_d4pg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-5)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cuda:0')
        self.to(self.device)

        self.to(self.device)

    def forward(self, state):
        state_tensor = T.tensor(state, dtype=T.float32).to(self.device) if isinstance(state, np.ndarray) else state
        prob = self.fc1(state_tensor)
        prob = F.leaky_relu(prob)
        prob = self.fc2(prob)
        prob = F.leaky_relu(prob)

        mu = T.tanh(self.mu(prob))
        # mu = F.hardtanh(self.mu(prob), self.min_action, self.max_action)


        return mu

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_atoms, v_min, v_max,
                 max_action, min_action, gamma=0.99, update_actor_interval=2,
                 warmup=1000, n_actions=2, max_size=500000, layer1_size=256,
                 layer2_size=256, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.max_action = max_action
        self.min_action = min_action
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions, name='actor', 
                                  min_action=min_action, max_action=max_action)

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions, n_atoms=n_atoms, 
                                      v_min=v_min, v_max=v_max, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions, n_atoms=n_atoms, 
                                      v_min=v_min, v_max=v_max, name='critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions, name='target_actor',
                                         min_action=min_action, max_action=max_action)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions=n_actions, n_atoms=n_atoms, 
                                             v_min=v_min, v_max=v_max, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions=n_actions, n_atoms=n_atoms, 
                                             v_min=v_min, v_max=v_max, name='target_critic_2')

        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
        self.epsilon = 1.0  # 100% probability for exploration
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.01
        self.fixed_noise_scale = 0.2 
        self.fixed_epsilon = 0.1
        self.update_network_parameters(tau=1)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            # Durante o warmup, escolha uma ação aleatória
            action = np.random.uniform(low=self.min_action, high=self.max_action, size=(self.n_actions,))
        else:
            # Use a política com um ruído constante para exploração
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
            noise = self.noise() * self.fixed_noise_scale
            print(f"self.noise: {noise}")

            # Com probabilidade epsilon, escolha uma ação aleatória para explorar
            if np.random.rand() > self.epsilon:
                action = np.random.uniform(low=self.min_action, high=self.max_action, size=(self.n_actions,))
                print(f"action: {action},  random: True")
            else:
                # Caso contrário, escolha a ação da política com ruído adicionado
                action = mu.cpu().detach().numpy() + noise
                print(f"action: {action},  random: False")

        # Garanta que a ação esteja dentro dos limites
        action = np.clip(action, self.min_action, self.max_action)
        print(f"final action: {action} between {self.min_action} min and {self.max_action} max")
        self.time_step += 1
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        state_batch = T.tensor(state, dtype=T.float).to(self.device)
        new_state_batch = T.tensor(new_state, dtype=T.float).to(self.device)
        action_batch = T.tensor(action, dtype=T.float).to(self.device)
        reward_batch = T.tensor(reward, dtype=T.float).to(self.device)
        done_batch = T.tensor(done).to(self.device)
        
        # with T.no_grad():
        target_actions = self.target_actor.forward(new_state_batch)
        q1_ = self.target_critic_1.forward(new_state_batch, target_actions)
        q2_ = self.target_critic_2.forward(new_state_batch, target_actions)
        # q_target = T.min(q1_, q2_)
        q_target = (q1_ + q2_) / 2.0
        projected_dist = self.project_distribution(q_target, reward_batch, done_batch)

        q_pred = self.critic_1.forward(state_batch, action_batch)
        loss = F.mse_loss(q_pred, projected_dist)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        loss.backward()

        # Clipping the critic gradients
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1)

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()

        # Clipping the actor gradient
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)

        self.actor.optimizer.step()

        self.update_network_parameters()
        return loss

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def project_distribution(self, q_target, rewards, dones):
        batch_size = rewards.size(0)
        deltas = (self.v_max - self.v_min) / (self.n_atoms - 1)
        atoms = T.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)

        projected_distributions = T.zeros(batch_size, self.n_atoms).to(self.device)
        gamma_term = self.gamma ** (1 - dones.float()).unsqueeze(1)

        atom_values = rewards.unsqueeze(1) + gamma_term * atoms
        atom_values = T.clamp(atom_values, self.v_min, self.v_max)
        index = (atom_values - self.v_min) / deltas
        lower = index.floor().long()
        upper = index.ceil().long()

        q_target_expanded = q_target.unsqueeze(1).expand(-1, self.n_atoms, -1)

        lower_contrib = (upper.float() - index).unsqueeze(2) * q_target_expanded
        upper_contrib = (index - lower.float()).unsqueeze(2)  * q_target_expanded

        lower_index_mask = T.zeros(batch_size, self.n_atoms, self.n_atoms, dtype=T.bool).to(self.device)
        upper_index_mask = T.zeros_like(lower_index_mask).to(self.device)

        lower_index_mask.scatter_(1, lower.unsqueeze(1), True)
        upper_index_mask.scatter_(1, upper.unsqueeze(1), True)

        projected_distributions += T.sum(lower_contrib * lower_index_mask, dim=2)
        projected_distributions += T.sum(upper_contrib * upper_index_mask, dim=2)

        return projected_distributions

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()


