import argparse
import gymnasium as gym
import numpy as np
# import hockey.hockey_env as h_env
import torch
import matplotlib.pyplot as plt
from sac import SAC

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

env = "Pendulum-v1"
env = gym.make(env, render_mode="human")

ac = SAC(env.observation_space.shape[0], env.action_space, args)

# models_dir = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\TD3"
# models_path = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\SAC\results3\SAC_pendulum_250.pth"
# models_path = r".\results2\SAC_pendulum_5.pth"
models_path = r"./checkpoints/sac_checkpoint_Pendulum-v1_ep_400"

ac.load_checkpoint(models_path, True)

for ep in range(5):
    obs, info = env.reset()
    d = False
    for i in range(2000): # while not d:
        if i % 1000 == 0: print(i)
        env.render()
        action_agent = ac.select_action(obs, evaluate=True)
        obs, r, d, t, info = env.step(action_agent)
        # print(obs)
        # print(r)
        # print(action_agent)

env.close()


# data = []
# for i in range(100000):
#     if i % 10000 == 0: print(i)
#     a = np.random.uniform(-1, 1)
#     b = np.random.uniform(-1, 1)
#     c = np.random.uniform(-8, 8)
#     obs = torch.tensor([a,b,c])
#     data.append(get_action(obs, True)[0])


# # Erstelle ein Histogramm
# plt.hist(data, bins=30, color='blue', edgecolor='black', alpha=0.7)

# # Achsentitel und Grafiküberschrift
# plt.xlabel('outsputs')
# plt.ylabel('Häufigkeit')
# plt.title('Verteilung der outputs für ein Model mit Tanh (nach 50 Epochen)')

# # Grafik anzeigen
# plt.show()


# obs, info = env.reset()

# print(obs)
# # print(ac.pi.net) # (torch.tensor(obs))
# # print(ac.pi.mu_layer)
# # print(ac.pi.mu_layer(ac.pi.net(torch.tensor(obs))))
# pi_action = ac.pi.mu_layer(ac.pi.net(torch.tensor(obs)))
# # print(torch.tanh(pi_action))
# # print(f"act limit: {ac.pi.act_limit}")
# print(f"falls deterministic: {ac.pi.act_limit * torch.tanh(pi_action)}")
# print(get_action(obs, True))
# print(get_action(obs, False))




