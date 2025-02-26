## SAC on Pendulum environment

First attempts of SAC on the Pendulum environment. SAC [(sac_checkpoint_Pendulum-v1_ep_400)](./checkpoints/sac_checkpoint_Pendulum-v1_ep_400) can solve the task only after 400 training iterations: 

![SAC solving the pendulum](../../assets/pendulum.gif)

remarks: 
- see [test.py](./test.py) for testing the agent
- runs privides statistics about the training run and can be viewed with `tensorboard --logdir=runs`