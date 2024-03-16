import rclpy
from util.trainer import Trainer
from model_free.sac.sac_torch import Agent
import pandas as pd


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer(algorithm_name='sac', stage=4)
    agent = Agent(input_dims=trainer.env.num_states, max_action=trainer.env.action_upper_bound, n_actions=trainer.env.num_actions)
    scores, steps = trainer.train(agent, 5000, 250, False)
    trainer.kill_env()
    rclpy.shutdown()

    # Create a DataFrame from the scores list
    df = pd.DataFrame({'scores': scores, 'steps':steps})

    # Add an 'episode' index column
    df.index.name = 'episode'

    # Save the DataFrame to a CSV file
    df.to_csv('networks/sac/scores.csv')

if __name__ == '__main__':
    main()