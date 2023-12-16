import rclpy
from turtle_env.turtle_env import Trainer
from td3.td3_torch import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer(algorithm_name='td3', stage=4)
    agent = Agent(alpha=0.001, beta=0.001,
            input_dims=trainer.env.num_states, tau=0.005,
            max_action=trainer.env.action_upper_bound, min_action=trainer.env.action_lower_bound,
            batch_size=64, layer1_size=150, layer2_size=256, # 400, 300
            n_actions=trainer.env.num_actions)
    trainer.train(agent, 5000, 200, False, stage=4)
    trainer.kill_env()
    rclpy.shutdown()

if __name__ == '__main__':
    main()