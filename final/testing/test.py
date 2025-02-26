import argparse
import gymnasium as gym
import numpy as np
# import hockey.hockey_env as h_env
import torch
import matplotlib.pyplot as plt

from QRSAC.qrsac import QRSAC
from SAC.sac import SAC
from SACHER.sac_HER import SACHER
from SIMBA.sac_SimBa import SACSIMBA
from SACRNN.rnn_sac import SACRNN
from gymnasium import spaces
# from qr_replay_memory import ReplayMemory
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
qr_sac_runx = QRSAC(env.observation_space.shape[0], action_space_p1, args)
sac_strong = SAC(env.observation_space.shape[0], action_space_p1, args)
sac_rnn = SACRNN(env.observation_space.shape[0], action_space_p1, args)
simba = SACSIMBA(env.observation_space.shape[0]+2, action_space_p1,1, "mlp", "mlp", args)
sac_her17400 = SACHER(env.observation_space.shape[0]+2, action_space_p1,args)
sac_her30000 = SACHER(env.observation_space.shape[0]+2, action_space_p1,args)


path1 = "./old_models/"
qr_sac_runx_path = path1 + fr"qr_sac_runx"
sac_strong_path = path1 + fr"sac_strong"
sac_rnn_path = path1 + fr"sac_rnn"
simba_path = path1 + "SimBa_auto_23600"
sac_her17400_path = path1 + "HER_17400"
sac_her30000_path = path1 + "HER_30000"

######################## turnier: 
qr_sac_runx.load_checkpoint(qr_sac_runx_path, True)
sac_strong.load_checkpoint(sac_strong_path, True)
sac_rnn.load_checkpoint(sac_rnn_path, True)
simba.load_checkpoint(simba_path, True)
sac_her17400.load_checkpoint(sac_her17400_path, True)
sac_her30000.load_checkpoint(sac_her30000_path, True)
weak_opp = h_env.BasicOpponent()
strong_opp = h_env.BasicOpponent(weak=False)


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

# elos = ELO(players_names)
# print(play(qr_sac, weak_opp, 100, False, elos, "SAC_strong", 'strong_opp'))
            
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

players = [qr_sac_runx, sac_strong, sac_rnn, simba, sac_her17400, sac_her30000, weak_opp, strong_opp]
players_names = ["qr_sac_runx", "sac_strong", "sac_rnn", "simba", "sac_her17400", "sac_her30000", "weak_opp", "strong_opp"]

tournament_rounds = 100
results, elos = tournament(players, players_names, tournament_rounds)


results = pd.DataFrame(results)
results.set_index('left player \\ right player')

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results.values,
                 rowLabels=results.index,
                 colLabels=results.columns,
                 cellLoc='center',
                 loc='center')

# Schriftgröße und Skalierung der Tabelle anpassen
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

elo_string = "ELOS: "
for key in elos:
    elo_string += f"{key}: {round(elos[key],1)} | "

description_text = (
    "detailed description how to read table:\n"
    "left column plays left player and head of colums plays right player\n"
    "Entry [x,y,z] means left player won x % and lost z % against right player. Game resultet in draw in y %.\n "
    "" + elo_string
)

# Verwende fig.text(), um den Text unterhalb der Tabelle zu platzieren
fig.text(0.5, 0.02, description_text, wrap=True, horizontalalignment='center', fontsize=10)

# Passe den unteren Rand an, damit der Text nicht abgeschnitten wird
plt.subplots_adjust(bottom=0.2)

plt.title(f"tournament results: {tournament_rounds} episodes per match")
plt.show()
