import numpy as np
import tensorflow as T

device = get_gpu()


class Buffer:
    def __init__(self, state_space, action_space, buffer_capacity=150000, batch_size=128):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_space))
        self.action_buffer = np.zeros((self.buffer_capacity, action_space))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_space))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def sample_buffer(self, batch_size):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state = T.tensor(self.state_buffer[batch_indices], dtype= T.float32).to(device)
        action = T.tensor(self.action_buffer[batch_indices], dtype= T.float32).to(device)
        reward = T.tensor(self.reward_buffer[batch_indices], dtype=T.float32).to(device)
        next_state = T.tensor(self.next_state_buffer[batch_indices], dtype= T.float32).to(device)

        return state, action, reward, next_state