import rclpy
from training.trainer import Trainer
from networks.ddpg.ddpg_torch import Agent
import pandas as pd


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer(algorithm_name='ddpg', stage=3)
    num_states = 14
    num_actions = 2


    alpha = 0.0001
    beta = 0.001
    tau = 0.001
    num_actions = num_actions

    # stage - 1 & 2: batch_size=256, max_size=1000000
    agent = Agent(alpha, beta, tau, input_dims=trainer.env.num_states, batch_size=512, fc1_dims=400, fc2_dims=300, n_actions=num_actions, max_size=2000000)
    scores = trainer.train(agent, 5000, 250, True)
    trainer.kill_env()
    rclpy.shutdown()

    # Create a DataFrame from the scores list
    df = pd.DataFrame({'scores': scores})

    # Add an 'episode' index column
    df.index.name = 'episode'

    # Save the DataFrame to a CSV file
    df.to_csv('networks/ddpg/scores.csv')

if __name__ == '__main__':
    main()