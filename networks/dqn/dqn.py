import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """
        Initialize the Deep Q-Network.

        :param lr: Learning rate for the optimizer.
        :param input_dims: Dimensions of the input layer.
        :param fc1_dims: Number of neurons in the first fully connected layer.
        :param fc2_dims: Number of neurons in the second fully connected layer.
        :param n_actions: Number of actions, defining the output layer size.
        """
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the network.

        :param state: The input state for which the action values are to be predicted.
        :return: Action values predicted by the network.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=50000, eps_end=0.05, eps_dec=5e-4):
        """
        Initialize the Agent with a Deep Q-Network.

        :param gamma: Discount factor for future rewards.
        :param epsilon: Initial value for the epsilon-greedy strategy.
        :param lr: Learning rate for the network.
        :param input_dims: Input dimensions for the network.
        :param batch_size: Size of the batch used during training.
        :param n_actions: Number of possible actions.
        :param max_mem_size: Maximum size of the memory buffer.
        :param eps_end: Minimum value that epsilon can decay to.
        :param eps_dec: The amount by which epsilon is decreased in each iteration.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size

        self.Q_eval = DeepQNetwork(lr, input_dims=input_dims, fc1_dims=512, fc2_dims=512, n_actions=n_actions)
        
        # Initialize memory buffers
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.mem_cntr = 0
        self.iter_cntr = 0

    def remember(self, state, action, reward, state_, terminal):
        """
        Store a transition in the memory buffer.

        :param state: The state of the environment before taking the action.
        :param action: The action taken.
        :param reward: The reward received after taking the action.
        :param state_: The state of the environment after taking the action.
        :param terminal: Boolean indicating whether the state is terminal.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        """
        Choose an action based on the current observation using epsilon-greedy strategy.

        :param observation: The current state of the environment.
        :return: The action chosen.
        """
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        """
        Perform a learning step for the agent.

        :return: The loss value of the learning step.
        """
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)
        
        return loss.item()
        
    def save_models(self):
        """
        Save the current state of the model.
        """
        T.save(self.Q_eval.state_dict(), './tmp/dqn/model.pt')

    def load_models(self):
        """
        Load the model from the saved state.
        """
        self.Q_eval.load_state_dict(T.load('./tmp/dqn/model.pt'))
        self.Q_eval.eval()
