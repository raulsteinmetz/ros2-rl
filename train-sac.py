import rclpy
from turtle_env.turtle_env import Trainer
from sac.sac_torch import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer(algorithm_name='sac', stage='stage_3')
    agent = Agent(input_dims=[trainer.env.num_states], action_space_high=trainer.env.action_upper_bound, n_actions=trainer.env.num_actions)
    trainer.train(agent, 5000, 140, False, 1)
    trainer.kill_env()
    rclpy.shutdown()

if __name__ == '__main__':
    main()