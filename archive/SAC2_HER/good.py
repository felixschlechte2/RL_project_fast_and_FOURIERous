import argparse
import datetime
import gymnasium as gym # hier : vorher nur gym
import numpy as np
import itertools
import torch
from sac import SAC
# from torch.utils.tensorboardx import SummaryWriter
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import time #here

import hockey.hockey_env as h_env # hier 
from gymnasium import spaces # hier 

env = h_env.HockeyEnv() # hier

def log(log):
    log_file = "./log_file.txt"
    with open(log_file, "a") as file:
        file.write(log)
        file.write("\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
log(str(device))

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

# hier :
# checkpoint_path = r"C:/Users/Home/Documents/M. Sc. ML/Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/SAC2/results/" # C:\Users\Home\Documents\M. Sc. ML\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\SAC2\results

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name)
# env.seed(args.seed)
# env.action_space.seed(args.seed)

args.env_name = "hockey_vs_rand_all_rew"

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# Agent
action_space_p1 = spaces.Box(-1, +1, (4,), dtype=np.float32)
agent = SAC(env.observation_space.shape[0]+3, action_space_p1, args)# env.action_space #added +3 for HER

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size,4,env, args.seed) #here added env, k_future

models_path = r"./checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_640_HER"
agent.load_checkpoint(models_path, True)

for ep in range(5):
    state, info = env.reset()
    d = False
    for i in range(500): # while not d:
        if i % 100 == 0: print(i)
        env.render()
        action_agent = agent.select_action(np.concatenate([state, [9.16, 2.75, 5.25]], axis=0), evaluate =True) #here# [:4] # hier # Sample action from policy
        action_opp = np.random.uniform(-1,1,4)
        action = np.hstack((action_agent, action_opp))
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        if done:
            print(done)
            print(i)
            break

env.close()