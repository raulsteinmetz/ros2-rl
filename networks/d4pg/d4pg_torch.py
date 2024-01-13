import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks.ddpg.networks import ActorNetwork, CriticNetwork
from networks.ddpg.noise import OUActionNoise
from networks.util.buffer import ReplayBuffer

class Agent:
    def __init__(self, alpha=5e-5, beta=0.001, tau=0.001, n_actions=0, input_dims=0,
                 gamma=0.99, max_size=1000000, fc1_dims=400, fc2_dims=300, V_MIN=-5, V_MAX=5, N_ATOMS=51,
                 batch_size=64, max_action=0, min_action=0, checkpoint_dir='tmp/d4pg'):
        """
        Initialize the Agent.

        Args:
            alpha (float): Learning rate for the actor.
            beta (float): Learning rate for the critic.
            tau (float): Soft update parameter.
            n_actions (int): Number of actions.
            input_dims (int): Input dimensions.
            gamma (float): Discount factor.
            max_size (int): Maximum size of the replay buffer.
            fc1_dims (int): Dimension of the first fully connected layer.
            fc2_dims (int): Dimension of the second fully connected layer.
            batch_size (int): Batch size for training.
            max_action (float): Maximum action value.
            min_action (float): Minimum action value.
            checkpoint_dir (str): Directory for saving checkpoints.
            V_MIN (float): Minimum value of the support for the value distribution.
            V_MAX (float): Maximum value of the support for the value distribution.
            N_ATOMS (int): Number of atoms in the value distribution.
        """
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.name = 'ddpg'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, 'actor', checkpoint_dir)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, 'critic', checkpoint_dir)
        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, 'target_actor', checkpoint_dir)
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, 'target_critic', checkpoint_dir)

        self.update_network_parameters(tau=1)

        self.N_ATOMS = N_ATOMS
        self.V_MIN = V_MIN
        self.V_MAX = V_MAX

    def choose_action(self, observation):
        """
        Choose an action based on the current observation using the actor network.

        Args:
            observation: The current state observation.

        Returns:
            The action chosen by the actor network with added noise for exploration.
        """
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device) 
        self.actor.train()

        print("Base Action:", mu.cpu().detach().numpy()[0])
        print("Noise:", T.tensor(self.noise(), dtype=T.float))
        print("Action with Noise:", mu_prime.cpu().detach().numpy()[0])
        
        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        """
        Store a transition in the replay buffer.

        Args:
            state: The starting state.
            action: The action taken.
            reward: The reward received.
            state_: The resulting state.
            done: Boolean indicating whether the episode is finished.
        """
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        """
        Save the current state of all networks (actor and critic) as checkpoints.
        """
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        """
        Load the saved states of all networks (actor and critic) from checkpoints.
        """
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        # Calculate the target distribution
        with T.no_grad():
            target_actions = self.target_actor.forward(states_)
            target_distr = self.target_critic.forward(states_, target_actions)
            target_atoms = T.linspace(self.V_MIN, self.V_MAX, self.N_ATOMS).to(self.actor.device)
            delta_z = (self.V_MAX - self.V_MIN) / (self.N_ATOMS - 1)
            
            Tz = rewards.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1).float()) * target_atoms.unsqueeze(0)
            Tz = Tz.clamp(min=self.V_MIN, max=self.V_MAX)
            b = (Tz - self.V_MIN) / delta_z
            lower = b.floor().long()
            upper = b.ceil().long()
            
            m = states.new_zeros(self.batch_size, self.N_ATOMS)
            offset = T.linspace(0, (self.batch_size - 1) * self.N_ATOMS, self.batch_size).unsqueeze(1).expand(self.batch_size, self.N_ATOMS).long()
            
            offset = offset.to(self.actor.device)
            lower = lower.to(self.actor.device)
            upper = upper.to(self.actor.device)

            m.view(-1).index_add_(0, (lower + offset).view(-1), (target_distr * (upper.float() - b)).view(-1))

        # Critic update
        critic_distr = self.critic.forward(states, actions)
        critic_loss = -T.sum(m * T.log(critic_distr + 1e-6), dim=1).mean()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Actor update
        actor_loss = -T.mean(T.sum(self.critic.forward(states, self.actor.forward(states)) * target_atoms, dim=1))

        self.actor.optimizer.zero_grad()
        actor_loss = -T.mean(self.critic.forward(states, self.actor.forward(states)))
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor.optimizer.step()

        print("Critic Loss:", critic_loss.item(), "Actor Loss:", actor_loss.item())

        # Update the target networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update the target networks
        with T.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)