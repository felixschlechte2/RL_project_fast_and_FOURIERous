import argparse
import gymnasium as gym
import numpy as np
# import hockey.hockey_env as h_env
import torch
import matplotlib.pyplot as plt
from qrsac import QRSAC
from sac import SAC
from gymnasium import spaces
from replay_memory import ReplayMemory
import pandas as pd
import pickle as pkl
# from utils import statistics
# import time

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
parser.add_argument('--trajectory_length', type=int, default=7, metavar='N',
                    help='length of trajectory that is captured (default: 7)')  
parser.add_argument('--num_quantile', type=int, default=16, metavar='N',
                    help='number of quantiles (default: 16)')  
parser.add_argument('--log_test_save', type=int, nargs=3, default=[50,100,1000], metavar='N',
                    help='intervals for logs, testing, saving (default: [10,100,1000])') 
args = parser.parse_args()

env = h_env.HockeyEnv()

action_space_p1 = spaces.Box(-1, +1, (4,), dtype=np.float32)
qr_sac = QRSAC(env.observation_space.shape[0], action_space_p1, args)
qr_sac2 = QRSAC(env.observation_space.shape[0], action_space_p1, args)
sac = SAC(env.observation_space.shape[0], action_space_p1, args)
sac_strong = SAC(env.observation_space.shape[0], action_space_p1, args)

# models_dir = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\TD3"
# models_path = r".\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\SAC\results3\SAC_pendulum_250.pth"
# models_path = r".\results2\SAC_pendulum_5.pth"
# buffer_path = r"checkpoints/sac_buffer_hockey_11"
# models_path = r"checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_8400"
sac_8400_path = r"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/QR-SAC4/sac_checkpoint_hockey_vs_rand_all_rew_ep_8400"
version = 9000
models_path = "./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/"
sac_strong_path = fr"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/SAC2_hockey_strong/checkpoints/sac_checkpoint_hockey_vs_strong_and_self_all_rew_ep_35000"
qr_sac_path = fr"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/QR-SAC5/checkpoints/checkpoint_hockey_vs_4_all_rew_ep_90000_qr_sac_runx"
qr_sac_path = models_path + fr"QR-SAC5/checkpoints/checkpoint_hockey_vs_4_all_rew_ep_90000_qr_sac_runx"
qr_sac2_path = fr"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/QR-SAC5/checkpoints/checkpoint_hockey_vs_sac_and_strong_all_rew_ep_20000_qr_sac_run5"
sac_clean_path = models_path + fr"SAC_clean/checkpoints/checkpoint_hockey_vs_self_all_rew_ep_{version}_sac_run3"
sac_rnn_path = models_path + fr"SAC-RNN2/checkpoints/checkpoint_hockey_vs_4_all_rew_ep_62000_sac_rnn2_run2"
# sac_rnn_path = models_path + fr"SAC-RNN/checkpoints/checkpoint_hockey_vs_weak_all_rew_ep_20000_sac_rnn_run1"

######################## turnier: 
qr_sac.load_checkpoint(qr_sac_path, True)
# qr_sac2.load_checkpoint(qr_sac2_path, True)
sac.load_checkpoint(sac_rnn_path, True)
# sac_strong.load_checkpoint(sac_strong_path, True)
weak_opp = h_env.BasicOpponent()
strong_opp = h_env.BasicOpponent(weak=False)

# players = [sac, sac, weak_opp, strong_opp]
players_names = ["SAC_strong", 'SAC_8400', 'weak_opp', 'strong_opp']

def play(agent1, agent2, epsisodes, render, elos, name1, name2):
    results = [0,0,0]
    for ep in range(epsisodes):
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        d = False
        ep_rew = 0
        len_episodes = 0
        while not d:
            len_episodes += 1
            if render:
                env.render(mode='human')
            if agent1 == "random":
                action_1 = np.random.uniform(-1,1,4) 
            elif hasattr(agent1, 'act'):
                if hasattr(agent1, 'weak'):
                    action_1 = agent1.act(obs)
                else: 
                    action_1 = agent1.act(obs, True)
            else: 
                action_1 = agent1.select_action(obs, True)
            
            if agent2 == "random":
                action_2 = np.random.uniform(-1,1,4)
            elif hasattr(agent2, 'act'):
                if hasattr(agent2, 'weak'):
                    action_2 = agent2.act(obs_agent2)
                else: 
                    action_2 = agent2.act(obs_agent2, True)
            else: 
                action_2 = agent2.select_action(obs_agent2, True)
            action = np.hstack((action_1, action_2))
            obs, r, d, t, info = env.step(action)
            # print(info)
            ep_rew += r
            obs_agent2 = env.obs_agent_two()
        winner = info['winner']
        if winner == 1: 
            results[0] += 1
            res = [1,0,0]
        if winner == 0: 
            results[1] += 1
            res = [0,0.5,0]
        if winner == -1: 
            results[2] += 1
            res = [0,0,1]
        # print(ep_rew)
        # print(len_episodes)
        if name1 != name2:
            elos.update_elos(name1, name2, res, 1)
    if render:
        env.close()
    return np.array(results)/epsisodes

class ELO():
    def __init__(self, players_names):
        self.elos = {}
        for name in players_names:
            self.elos[name] = 1500
        self.K = {}
        for name in players_names:
            self.K[name] = True

    def update_elos(self, name1, name2, result, episodes):
        elo_1 = self.elos[name1]
        elo_2 = self.elos[name2]
        E_1 = 1/(1 + 10**((elo_2 - elo_1)/400))
        E_2 = 1/(1 + 10**((elo_1 - elo_2)/400))
        points_1 = result[0] + result[1] # int(round(result[0]*episodes)) + int(round(result[1]*episodes)) * 0.5
        points_2 = result[2] + result[1] # int(round(result[2]*episodes)) + int(round(result[1]*episodes)) * 0.5
        if self.elos[name1] >= 2400: 
            k1 = 10
        else: 
            k1 = 20
        if self.elos[name2] >= 2400: 
            k2 = 10
        else: 
            k2 = 20
        self.elos[name1] = self.elos[name1] + k1 * (points_1 - E_1) # * episodes
        self.elos[name2] = self.elos[name2] + k2 * (points_2 - E_2) # * episodes

elos = ELO(players_names)

# for i in range(3000, 45000, 3000):
#     sac_clean_path = models_path + fr"SAC_clean/checkpoints/checkpoint_hockey_vs_self_all_rew_ep_{i}_sac_run3"
#     sac.load_checkpoint(sac_clean_path, True)
#     print(f"version: {i}")
#     print("strong opponent:")
#     print(play(sac, strong_opp, 10, False, elos, "SAC_strong", 'strong_opp'))
#     print("sac strong:")
#     print(play(sac, sac_strong, 10, False, elos, "SAC_strong", 'strong_opp'))

print(play(sac, qr_sac, 10, True, elos, "SAC_strong", 'strong_opp'))
# print(play(qr_sac, sac_strong, 10, True, elos, "SAC_strong", 'strong_opp'))
# print(play(sac_strong, weak_opp, 5, True, elos, "SAC_strong", 'strong_opp'))
# print(play(sac_strong, ac, 5, True, elos, "SAC_strong", 'strong_opp'))
            
players = [sac, qr_sac, weak_opp, strong_opp]
players_names = ["SAC", 'QR-SAC', 'weak_opp', 'strong_opp']

def tournament(players, players_names, episodes): 
    results = {'left player \\ right player': []}
    elos = ELO(players_names)
    for pn in players_names:
        results[pn] = []
    for i in range(len(players)): 
        results['left player \\ right player'].append(players_names[i])
        print(players_names[i])
        for j in range(len(players)): 
            print("   ->   " + players_names[j])
            res = play(players[i], players[j], episodes, False, elos, players_names[i], players_names[j]) 
            # elos.update_elos(players_names[i], players_names[j], res, episodes)
            results[players_names[j]].append(res)
    return results, elos.elos

# models_path = "./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/"
# for i in range(2000, 8001, 2000):
#     path = models_path + fr"SAC-RNN2/checkpoints/checkpoint_hockey_vs_2_all_rew_ep_{i}_sac_rnn2_run1"
#     sac.load_checkpoint(path, True)
#     elos = ELO(players_names)
#     results, elos = tournament(players, players_names, 10)
#     print(f"version: {i}")
#     print(elos)
# print("###############################erste runde fertig#######################################")
# for i in range(10000, 62001, 10000):
#     path = models_path + fr"SAC-RNN2/checkpoints/checkpoint_hockey_vs_4_all_rew_ep_{i}_sac_rnn2_run2"
#     sac.load_checkpoint(path, True)
#     elos = ELO(players_names)
#     results, elos = tournament(players, players_names, 10)
#     print(f"version: {i}")
#     print(elos)
# elos = ELO(players_names)
# tournament_rounds = 10
# results, elos = tournament(players, players_names, tournament_rounds)
# # for key in elos:
# #     print(f"Elo for {key}: {elos[key]}")

# results = pd.DataFrame(results)
# results.set_index('left player \\ right player')

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.axis('tight')
# ax.axis('off')
# table = ax.table(cellText=results.values,
#                  rowLabels=results.index,
#                  colLabels=results.columns,
#                  cellLoc='center',
#                  loc='center')

# # Schriftgröße und Skalierung der Tabelle anpassen
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.2)

# elo_string = "ELOS: "
# for key in elos:
#     elo_string += f"{key}: {round(elos[key],1)} | "

# description_text = (
#     "detailed description how to read table:\n"
#     "left column plays left player and head of colums plays right player\n"
#     "Entry [x,y,z] means left player won x % and lost z % against right player. Game resultet in draw in y %.\n "
#     "" + elo_string
# )

# # Verwende fig.text(), um den Text unterhalb der Tabelle zu platzieren
# fig.text(0.5, 0.02, description_text, wrap=True, horizontalalignment='center', fontsize=10)

# # Passe den unteren Rand an, damit der Text nicht abgeschnitten wird
# plt.subplots_adjust(bottom=0.2)

# plt.title(f"tournament results: {tournament_rounds} episodes per match")
# plt.show()

################################################################################ matrix norm: 
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
# ac.load_checkpoint(models_path, True)

# weak_opp = h_env.BasicOpponent(weak=False)

# for ep in range(10):
#     obs, info = env.reset()
#     obs_agent2 = env.obs_agent_two()
#     d = False
#     while not d: # for i in range(2000): # while not d:
#         # if i % 1000 == 0: print(i)
#         env.render(mode="human")
#         action_agent = ac.act(obs, True)[:4]  # Sample action from policy
#         # print(action_agent)
#         action_opp = np.random.uniform(-1,1,4)
#         # action_opp = weak_opp.act(obs_agent2)
#         action = np.hstack((action_agent, action_opp))
#         obs, r, d, t, info = env.step(action)
#         obs_agent2 = env.obs_agent_two()
#         # print(obs)
#         # print(r)
#         # print(info)
#         # print(action_agent)

# env.close()

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

# action_qr_sac = qr_sac.act(obs)
# action_sac = sac.select_action(obs)

# # print(env.action_space.sample().shape)
# obs = torch.from_numpy(obs).unsqueeze(0).float()
# action_qr_sac = torch.from_numpy(np.array([action_qr_sac])).float()
# action_sac = torch.from_numpy(np.array([action_sac])).float()

# print(qr_sac.critic.q_values(obs, action_qr_sac))
# print(sac.critic(obs, action_sac))

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


####################### plot Verteilung von strong vs self play: 
# def which_opp(episode):
#     if (episode // 1000) % 3 in [0,1]: 
#         return True, 'strong' # strong opp
#     else: 
#         return False, 'self' # weak opp
    
# data = []
# for i in range(10000):
#     data.append(which_opp(i)[0])

# plt.plot(data, marker='o', linestyle='-')

# # Achsenbeschriftungen hinzufügen
# plt.xlabel('episodes')
# plt.ylabel('Wert')

# # Titel setzen
# plt.title('Verteilung strong vs self')

# # Plot anzeigen
# plt.show()