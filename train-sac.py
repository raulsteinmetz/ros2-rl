import rclpy
from training.trainer import Trainer
from networks.sac.sac_torch import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer(algorithm_name='sac', stage=4)
    agent = Agent(input_dims=[trainer.env.num_states], action_space_high=trainer.env.action_upper_bound, n_actions=trainer.env.num_actions)
    trainer.train(agent, 5000, 250, True)
    trainer.kill_env()
    rclpy.shutdown()

if __name__ == '__main__':
    main()