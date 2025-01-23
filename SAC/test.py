import gymnasium as gym
# import os
import numpy as np
# import hockey.hockey_env as h_env
import torch
import core

env = "Pendulum-v1"
env = gym.make(env)

ac = core.MLPActorCritic(env.observation_space, env.action_space)

# models_dir = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\TD3"
models_path = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\SAC\results\test.pth"

ac.load_state_dict(torch.load(models_path, weights_only=False))

o, i = env.reset()

print(o)

def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

o, r, d, _, _ = env.step(get_action(o))

# obs, info = env.reset()
# obs_agent2 = env.obs_agent_two()

# # print(obs)
# # print(obs_agent2)

# # env.render(mode="human")

# for ep in range(5):
#     obs, info = env.reset()
#     d = False
#     while not d:
#         env.render(mode="human")
#         action_agent = ddpg.act(obs)[:4]
#         action_opp = basic_opp.act(obs_agent2)
#         # action_agent = np.array(action_agent)
#         # action_random = np.array(env.action_space.sample()[:4])
#         action = np.hstack((action_agent, action_opp))
#         # obs, r, d, t, info = env.step(action_agent)
#         obs, r, d, t, info = env.step(action)
#         # print(r)
#         obs_agent2 = env.obs_agent_two()


# env.close()


