import rclpy
from turtle_env.turtle_env import Trainer
from d4pg.d4pg_torch import Agent


def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer(algorithm_name='d4pg', stage='stage_1')
    agent = Agent(alpha=0.0001,
                  beta=0.001,
                  input_dims=trainer.env.num_states,
                  tau=0.005,
                  n_atoms=32,
                  v_min=-1,
                  v_max=1,
                  max_action=2,
                  min_action=-2,
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