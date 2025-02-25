import gymnasium as gym
# import os
import numpy as np
# import hockey.hockey_env as h_env
import torch
import core
# from sac import *

env = "Pendulum-v1"
env = gym.make(env, render_mode="human")

ac = core.MLPActorCritic(env.observation_space, env.action_space)

# models_dir = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\TD3"
# models_path = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\SAC\results\SAC_pendulum_30.pth"
models_path = r".\results2\SAC_pendulum_5.pth"

ac.load_state_dict(torch.load(models_path, weights_only=False, map_location=torch.device('cpu')))

def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

obs, info = env.reset()

print(ac.pi.net(torch.tensor(obs)))
print(ac.pi.mu_layer.weight)
print(ac.pi.mu_layer(ac.pi.net(torch.tensor(obs))))
pi_action = ac.pi.mu_layer(ac.pi.net(torch.tensor(obs)))
print(torch.tanh(pi_action))
print(f"act limit: {ac.pi.act_limit}")
print(ac.pi.act_limit * torch.tanh(pi_action))
print(get_action(obs))

# for ep in range(5):
#     obs, info = env.reset()
#     d = False
#     while not d:
#         env.render()
#         action_agent = get_action(obs)
#         obs, r, d, t, info = env.step(action_agent)
#         print(action_agent)

# env.close()


