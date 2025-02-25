import argparse
import gymnasium as gym
import numpy as np
# import hockey.hockey_env as h_env
import torch
import matplotlib.pyplot as plt
from sac import SAC
from gymnasium import spaces
from replay_memory import ReplayMemory

import hockey.hockey_env as h_env

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


env = h_env.HockeyEnv()

action_space_p1 = spaces.Box(-1, +1, (4,), dtype=np.float32)
ac = SAC(env.observation_space.shape[0], action_space_p1, args)

# models_dir = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\TD3"
# models_path = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\SAC\results3\SAC_pendulum_250.pth"
# models_path = r".\results2\SAC_pendulum_5.pth"
models_path = r"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/SAC2_hockey_random/checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_8400"
buffer_path = r"checkpoints/sac_buffer_hockey_11"
# models_path = r"checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_8400"


################ matrix norm: 
# x_ticks = []
# p1 = []
# p2 = []
# c1 = []
# c2 = []
# c3 = []
# for i in range(0, 8001, 500):
#     x_ticks.append(i)
#     models_path = f"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/SAC2_hockey/checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_{i}"
#     # models_path = f"checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_{i}"
#     ac.load_checkpoint(models_path, True)
#     p1.append(torch.linalg.matrix_norm(ac.policy.linear1.weight).detach().numpy())
#     p2.append(torch.linalg.matrix_norm(ac.policy.linear2.weight).detach().numpy())
#     c1.append(torch.linalg.matrix_norm(ac.critic.linear1.weight).detach().numpy())
#     c2.append(torch.linalg.matrix_norm(ac.critic.linear2.weight).detach().numpy())
#     c3.append(torch.linalg.matrix_norm(ac.critic.linear3.weight).detach().numpy())

# x_ticks.append(8400)
# models_path = f"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/SAC2_hockey/checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_8400"
# # models_path = f"checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_8400"
# ac.load_checkpoint(models_path, True)
# p1.append(torch.linalg.matrix_norm(ac.policy.linear1.weight).detach().numpy())
# p2.append(torch.linalg.matrix_norm(ac.policy.linear2.weight).detach().numpy())
# c1.append(torch.linalg.matrix_norm(ac.critic.linear1.weight).detach().numpy())
# c2.append(torch.linalg.matrix_norm(ac.critic.linear2.weight).detach().numpy())
# c3.append(torch.linalg.matrix_norm(ac.critic.linear3.weight).detach().numpy())

# # ac.load_checkpoint(models_path, True)
# # print(ac.policy.linear2.weight)
# # print(torch.linalg.matrix_norm(ac.policy.linear2.weight))

# plt.plot(x_ticks, p1, label = 'policy layer 1', marker='o', linestyle='-')  # Optional: Punkte mit Linien verbinden
# plt.plot(x_ticks, p2, label = 'policy layer 2', marker='o', linestyle='-') #, color='b'
# plt.plot(x_ticks, c1, label = 'critic layer 1', marker='o', linestyle='-')
# plt.plot(x_ticks, c2, label = 'critic layer 2', marker='o', linestyle='-')
# plt.plot(x_ticks, c3, label = 'critic layer 3', marker='o', linestyle='-')
# plt.title("evolution layer norm")
# plt.xlabel("epochs")
# plt.ylabel(r"$\|\cdot\|_F$")
# plt.legend()
# plt.grid(True)  # Raster hinzufügen
# plt.show()

##################### play:
ac.load_checkpoint(models_path, True)

weak_opp = h_env.BasicOpponent(weak=False)

for ep in range(10):
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    d = False
    while not d: # for i in range(2000): # while not d:
        # if i % 1000 == 0: print(i)
        env.render(mode="human")
        action_agent = ac.select_action(obs_agent2, True)  # Sample action from policy
        action_opp = weak_opp.act(obs)
        # print(action_agent)
        # action_opp = np.random.uniform(-1,1,4)
        action = np.hstack((action_opp, action_agent))
        obs, r, d, t, info = env.step(action)
        obs_agent2 = env.obs_agent_two()
        # print(obs)
        # print(r)
        # print(info)
        # print(action_agent)

env.close()

############# test output :
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

############# print layers :
# obs, info = env.reset()

# action_agent = ac.select_action(obs)

# print(env.action_space.sample().shape)
# print(action_agent)

# print(ac.critic.linear2.weight)
# print(obs)
# print(ac.pi.net) # (torch.tensor(obs))
# print(ac.pi.mu_layer)
# print(ac.pi.mu_layer(ac.pi.net(torch.tensor(obs))))
# pi_action = ac.pi.mu_layer(ac.pi.net(torch.tensor(obs)))
# print(torch.tanh(pi_action))
# print(f"act limit: {ac.pi.act_limit}")
# print(f"falls deterministic: {ac.pi.act_limit * torch.tanh(pi_action)}")
# print(get_action(obs, True))
# print(get_action(obs, False))


############ print buffer :
# memory = ReplayMemory(args.replay_size, args.seed)
# memory.load_buffer(buffer_path)

# for i in range(len(memory.buffer)):
#     if i % 100 == 0: print(i)
#     if memory.buffer[i][-1] != 1: 
#         print(memory.buffer[i])
    

# state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.batch_size)

# print("state batch:")
# print(state_batch)
# print("action batch")
# print(action_batch)
# print("reward batch")
# print(reward_batch)
# print("next state batch")
# print(next_state_batch)
# print("mask batch")
# print(mask_batch)


