import rclpy
from turtle_env.turtle_env import Trainer
from ddpg.ddpg_torch import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer('ddpg_')
    num_states = 14
    num_actions = 2

    upper_bound = .25
    lower_bound = -.25
    ACTION_V_MAX = 0.22 # m/s
    ACTION_W_MAX = 1. # rad/s

    alpha = 0.0001
    beta = 0.001
    input_dims = num_states
    tau = 0.001
    batch_size = 64
    fc1_dims = 400
    fc2_dims = 300
    num_actions = num_actions

    agent = Agent(alpha, beta, [input_dims], tau, batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=num_actions)
    trainer.train(agent, 5000, 500, False, stage=1)
    trainer.kill_env()
    rclpy.shutdown()

if __name__ == '__main__':
    main()