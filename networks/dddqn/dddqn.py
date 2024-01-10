import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        """
        Initialize the Replay Buffer.

        :param max_size: Maximum size of the buffer.
        :param input_shape: Shape of the input states.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def remember(self, state, action, reward, state_, done):
        """
        Store a new memory in the buffer.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param state_: The next state.
        :param done: Boolean flag indicating if the episode is done.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Sample a batch of memories from the buffer.

        :param batch_size: The size of the batch to sample.
        :return: A batch of states, actions, rewards, next states, and dones.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        """
        Initialize the Dueling Deep Q-Network.

        :param lr: Learning rate.
        :param n_actions: Number of possible actions.
        :param name: Name of the model for checkpointing.
        :param input_dims: Input dimensions of the network.
        :param chkpt_dir: Directory to save checkpoints.
        """
        super(DuelingDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 512)
        self.V = nn.Linear(512, 1)  # Value stream
        self.A = nn.Linear(512, n_actions)  # Advantage stream

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_file = os.path.join(chkpt_dir, name)

    def forward(self, state):
        """
        Forward pass through the network.

        :param state: Input state.
        :return: Value and advantage streams.
        """
        flat1 = F.relu(self.fc1(state))
        V = self.V(flat1)  # Value stream
        A = self.A(flat1)  # Advantage stream

        return V, A

    def save_checkpoint(self):
        """
        Save the current state of the model as a checkpoint.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the model from a saved checkpoint.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, replace=1000, chkpt_dir='tmp/dueling_ddqn'):
        """
        Initialize the Agent with Dueling Deep Q-Networks.

        :param gamma: Discount factor for future rewards.
        :param epsilon: Initial value for the epsilon-greedy strategy.
        :param lr: Learning rate.
        :param n_actions: Number of actions.
        :param input_dims: Input dimensions.
        :param mem_size: Memory buffer size.
        :param batch_size: Batch size for learning.
        :param eps_min: Minimum value for epsilon.
        :param eps_dec: Epsilon decrement value.
        :param replace: Interval for updating the target network.
        :param chkpt_dir: Directory for saving checkpoints.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(lr, n_actions, input_dims=input_dims, name='dueling_ddqn_q_eval', chkpt_dir=chkpt_dir)
        self.q_next = DuelingDeepQNetwork(lr, n_actions, input_dims=input_dims, name='dueling_ddqn_q_next', chkpt_dir=chkpt_dir)

    def choose_action(self, observation):
        """
        Choose an action based on the current observation using epsilon-greedy strategy.

        :param observation: The current state observation.
        :return: The action chosen.
        """
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def remember(self, state, action, reward, state_, done):
        """
        Store a new memory in the buffer.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param state_: The next state.
        :param done: Boolean flag indicating if the episode is done.
        """
        self.memory.remember(state, action, reward, state_, done)


    def replace_target_network(self):
        """
        Replace the target network weights with the evaluation network weights.
        This occurs at regular intervals defined by 'replace_target_cnt'.
        """
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        """
        Decrement the epsilon value used in the epsilon-greedy strategy.
        Epsilon is decreased by 'eps_dec' until it reaches 'eps_min'.
        """
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save_models(self):
        """
        Save the models for both the evaluation and target networks.
        """
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        """
        Load the models for both the evaluation and target networks from saved checkpoints.
        """
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        """
        Conduct a learning step for the agent. This includes sampling a batch of memories,
        calculating loss, and updating network weights.

        :return: The loss value from the learning step.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval(states)
        V_s_, A_s_ = self.q_next(states_)

        V_s_eval, A_s_eval = self.q_eval(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        return loss.item()