import rclpy
from turtle_env.turtle_env import Trainer
from d4pg.d4pg_torch import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer('d4pg')
    agent = Agent(alpha=0.0001,
                  beta=0.001,
                  input_dims=trainer.env.num_states,
                  tau=0.005,
                  n_atoms=51,
                  v_min=-10,
                  v_max=10,
                  max_action=trainer.env.action_upper_bound,
                  min_action=trainer.env.action_lower_bound,
                  batch_size=64,
                  layer1_size=256,
                  layer2_size=256,
                  n_actions=trainer.env.num_actions,
                  )
    trainer.train(agent=agent, 
                  episodes=5000, 
                  max_steps=300, 
                  load_models=False, 
                  stage=1)
    trainer.kill_env()
    rclpy.shutdown()

if __name__ == '__main__':
    main()