import rclpy
from training.trainer import Trainer
from networks.dqn.dqn import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer('dqn')
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.1, 
                input_dims=[14], n_actions=5, eps_end=0.01,
                batch_size=64, eps_dec=1e-5)
    trainer.train(agent, 5000, 200, False, 1, discrete=True)
    trainer.kill_env()
    rclpy.shutdown()

if __name__ == '__main__':
    main()