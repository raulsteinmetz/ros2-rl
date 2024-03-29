import rclpy
from util.trainer import Trainer
from util.save_csv import save_rewards_scores
from model_free.ddpg.ddpg_torch import Agent
import pandas as pd


def main(args=None):
    network = 'ddpg'
    rclpy.init(args=args)
    trainer = Trainer(algorithm_name=network, stage=4)
    num_states = 14
    num_actions = 2


    alpha = 0.0001
    beta = 0.001
    tau = 0.001
    num_actions = num_actions

    # stage - 1 & 2: batch_size=256, max_size=1000000
    agent = Agent(alpha, beta, tau, input_dims=trainer.env.num_states, batch_size=512, fc1_dims=400, fc2_dims=300, n_actions=num_actions, max_size=2000000)
    scores, steps = trainer.train(agent, 5000, 250, False)
    trainer.kill_env()
    rclpy.shutdown()

    save_rewards_scores(scores, steps, network)

if __name__ == '__main__':
    main()