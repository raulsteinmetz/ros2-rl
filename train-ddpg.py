import rclpy
from turtle_env.turtle_env import Trainer
from ddpg.agent import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer()
    num_states = 14
    num_actions = 2

    upper_bound = .25
    lower_bound = -.25
    ACTION_V_MAX = 0.22 # m/s
    ACTION_W_MAX = 1. # rad/s

    # Learning rate for actor-critic models
    critic_lr = 0.0001
    actor_lr = 0.0001

    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.001

    agent = Agent(num_states, num_actions, upper_bound, lower_bound, gamma, tau, critic_lr, actor_lr, 0.2)
    trainer.train(agent, 5000, 140, False, 1)
    trainer.kill_env()
    rclpy.shutdown()

if __name__ == '__main__':
    main()