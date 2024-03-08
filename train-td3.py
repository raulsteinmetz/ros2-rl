import pandas as pd
import rclpy
from training.trainer import Trainer
from networks.td3.td3_torch import Agent

def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer(algorithm_name='td3', stage=3)
    agent = Agent(alpha=0.001, beta=0.001,
            input_dims=trainer.env.num_states, tau=0.005,
            max_action=trainer.env.action_upper_bound, min_action=trainer.env.action_lower_bound,
            batch_size=64, layer1_size=400, layer2_size=300, # 400, 300
            n_actions=trainer.env.num_actions)
    scores = trainer.train(agent, 2000, 160, load_models=False)
    trainer.kill_env()
    rclpy.shutdown()

    # Create a DataFrame from the scores list
    df = pd.DataFrame({'scores': scores})

    # Add an 'episode' index column
    df.index.name = 'episode'

    # Save the DataFrame to a CSV file
    df.to_csv('networks/td3/scores.csv')

if __name__ == '__main__':
    main()
