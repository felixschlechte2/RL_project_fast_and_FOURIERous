import gymnasium as gym
# import os
import numpy as np
import sys
# sys.path.append("C:/Users/Home/Documents/M.Sc.ML/ReinforcementLearning/ExcercisesGitHub/exercises mit venv/project_code")
import hockey.hockey_env as h_env
import torch
from DDPG_agent_shooting import DDPGAgent

env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)

ddpg = DDPGAgent(env.observation_space, env.action_space, eps = 0.1, learning_rate_actor = 0.0001,
                     update_target_every = 100)

# models_dir = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\TD3"
models_path = r".\ReinforcementLearning\ExcercisesGitHub\exercises mit venv\project_code\models\DDPG\results\DDPG_hockey_shooting_weak_opp_5000-eps0.1-t50-l0.0001-sNone.pth"

ddpg.restore_state(torch.load(models_path))

basic_opp = h_env.BasicOpponent()

obs, info = env.reset()
obs_agent2 = env.obs_agent_two()

# print(obs)
# print(obs_agent2)

# env.render(mode="human")

for ep in range(5):
    obs, info = env.reset()
    d = False
    while not d:
        env.render(mode="human")
        action_agent = ddpg.act(obs)[:4]
        action_opp = [0,0,0,0] # basic_opp.act(obs_agent2)
        # action_agent = np.array(action_agent)
        # action_random = np.array(env.action_space.sample()[:4])
        action = np.hstack((action_agent, action_opp))
        # obs, r, d, t, info = env.step(action_agent)
        obs, r, d, t, info = env.step(action)
        print(r)
        obs_agent2 = env.obs_agent_two()


env.close()


