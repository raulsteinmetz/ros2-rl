import pandas as pd
import rclpy
from training.tester import Tester
from networks.td3.td3_torch import Agent
import numpy as np
import torch as T

def main(args=None):
    np.random.seed(42) 
    T.manual_seed(42)
    T.cuda.manual_seed_all(42)

    rclpy.init(args=args)
    tester = Tester(algorithm_name='td3', stage=2)
    agent = Agent(alpha=0.001, beta=0.001,
            input_dims=tester.env.num_states, tau=0.005,
            max_action=tester.env.action_upper_bound, min_action=tester.env.action_lower_bound,
            batch_size=64, layer1_size=150, layer2_size=256, # 400, 300
            n_actions=tester.env.num_actions, update_actor_interval=2)
    scores = tester.test(agent, 100, 250, load_models=False, warmup=20)
    tester.kill_env()
    rclpy.shutdown()

    # Create a DataFrame from the scores list
    df = pd.DataFrame({'scores': scores})

    # Add an 'episode' index column
    df.index.name = 'episode'

    # Save the DataFrame to a CSV file
    df.to_csv('networks/td3/test.csv')

if __name__ == '__main__':
    main()
