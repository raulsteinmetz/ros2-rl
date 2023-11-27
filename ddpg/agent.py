import tensorflow as tf
from tensorflow import keras
import numpy as np
from buffer import Buffer
from noise import OUActionNoise
import os


def get_actor(state_space, action_high, action_space):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = keras.layers.Input(shape=(state_space,))
    # Regular dense layer with 512 units
    out = keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    out = keras.layers.Dropout(0.1)(out) 
    # Uncomment below line to add regularizers and dropout
    out = keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(out)
    out = keras.layers.Dropout(0.1)(out) 
    out = keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(out)
    out = keras.layers.Dropout(0.1)(out) 
    
    # Separate outputs for linear and angular velocities
    lin_vel = keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(0.01))(out) # Linear velocity with sigmoid activation
    ang_vel = keras.layers.Dense(1, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(0.01))(out) # Angular velocity with tanh activation
    outputs = keras.layers.Concatenate()([lin_vel, ang_vel])
    # outputs = keras.layers.Concatenate()([lin_vel, ang_vel], kernel_initializar=last_init)

    # Scale the outputs by the action high
    outputs = outputs * action_high
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(state_space, action_space):
    # State input
    state_input = keras.layers.Input(shape=(state_space))
    # Regular dense layer for state pathway
    state_out = keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(state_input)
    state_out = keras.layers.Dropout(0.1)(state_out)
    # Uncomment below lines to add regularizers and dropout for state pathway
    # state_out = keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(state_input)
    # state_out = keras.layers.Dropout(0.1)(state_out) # Dropout layer
    
    action_input = keras.layers.Input(shape=(action_space))
    # Regular dense layer for action pathway
    action_out = keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(action_input)
    action_out = keras.layers.Dropout(0.1)(action_out)
    # Uncomment below lines to add regularizers and dropout for action pathway
    # action_out = keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(action_input)
    # action_out = keras.layers.Dropout(0.1)(action_out) # Dropout layer

    # Concatenate state and action pathways
    concat = keras.layers.Concatenate()([state_out, action_out])
    
    # Regular dense layers after concatenation
    out = keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(concat)
    out = keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(out)
    outputs = keras.layers.Dense(1, activation="linear", kernel_regularizer=tf.keras.regularizers.l2(0.01))(out) # Output Q value with linear activation

    # Outputs a single value for a given state-action
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

class Agent:
    def __init__(self, state_space, action_space, action_high,
                 action_low, gamma, tau, critic_lr, actor_lr, noise_std):
        print(f'state_space: {state_space}')
        print(f'action_space: {action_space}')
        self.mem = Buffer(state_space, action_space, 50000, 64)
        self.actor = get_actor(state_space, action_high, action_space)
        self.critic = get_critic(state_space, action_space)

        self.target_actor = get_actor(state_space, action_high, action_space)
        self.target_critic = get_critic(state_space, action_space)

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        # self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr * 0.01) # test for environment 1
        # self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr * 0.01) # test for environment 1

        self.action_high = action_high
        self.action_low = action_low

        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std) * np.ones(1)) # original
        # self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.05) * np.ones(1), theta=0.05) # test for environment 1
        # self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.1) * np.ones(1), theta=0.1) # test for environment 2
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.02) * np.ones(1), theta=0.1) # Test for environment 2
    
    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.mem.buffer_counter, self.mem.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.mem.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.mem.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.mem.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.mem.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.mem.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor(state))
        noise = self.noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        # sampled_actions = np.abs(sampled_actions)
        legal_action = np.clip(sampled_actions, self.action_low, self.action_high)
        # legal_action[1] = np.clip(legal_action[1], -0.5, 0.5)  # Limiting angular velocity to avoid perpetual rotation

        return [np.squeeze(legal_action)]

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self):
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * self.tau + a * (1 - self.tau))

        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * self.tau + a * (1 - self.tau))
    
    def try_load_model_weights(self, model, file_path):
        if os.path.exists(file_path):
            model.load_weights(file_path)

    def save_models(self, directory="./stage-two-models"):
        """Saves the target actor and critic models."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.target_actor.save_weights(os.path.join(directory, "target_actor.h5"))
        self.target_critic.save_weights(os.path.join(directory, "target_critic.h5"))

    def load_models(self, directory="./stage-two-models"):
        """Loads the target actor and critic models."""
        self.try_load_model_weights(self.target_actor, os.path.join(directory, "target_actor.h5"))
        self.try_load_model_weights(self.target_critic, os.path.join(directory, "target_critic.h5"))