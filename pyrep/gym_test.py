import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent


problem = "Pendulum-v1"
env = gym.make(problem, render_mode="human")

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

agent = Agent(num_states, num_actions, upper_bound, lower_bound, gamma, tau, critic_lr, actor_lr, 0.2)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):

    prev_state = env.reset()[0]
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state)
        # Recieve state and reward from environment.
        state, reward, done, _, _ = env.step(action)

        env.render()

        agent.mem.record((prev_state, action, reward, state))
        episodic_reward += reward

        agent.learn()
        agent.update_target()

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()