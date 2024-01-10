import rclpy
from training.trainer import Trainer
from networks.dddqn.dddqn import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer('dddqn', stage=1)
    agent = Agent(gamma=0.99, epsilon=1.0, lr=1e-5,
                  input_dims=[14], n_actions=5, mem_size=50000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-4, replace=100)
    trainer.train(agent, 5000, 180, True, discrete=True)
    trainer.kill_env()
    rclpy.shutdown()

if __name__ == '__main__':
    main()