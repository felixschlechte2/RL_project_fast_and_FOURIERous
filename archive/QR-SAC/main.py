import argparse
import datetime
import gymnasium as gym # hier : vorher nur gym
import numpy as np
import itertools
import torch
from qrsac import QRSAC
# from torch.utils.tensorboardx import SummaryWriter
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import pickle as pkl # hier 

import hockey.hockey_env as h_env # hier 
from gymnasium import spaces # hier 

env = h_env.HockeyEnv() # hier

# hier bis parser :
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
                    help='batch size (default: 256)')  # hier : vorher 256
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
                    help='length of trajectory that is captured') # hier 
parser.add_argument('--num_quantile', type=int, default=16, metavar='N',
                    help='number of quantiles') # hier 
args = parser.parse_args()

args.env_name = "hockey_vs_random_all_rew"

# hier diese funktion:
def rollout_loop(env, agent1, agent2, memory, total_numsteps):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()[0] # hier das [0]
    obs_agent2 = env.obs_agent_two() # hier 
    
    trajectory = []
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            action_agent = action[:4] # hier 
        else:
            action_agent = agent1.act(state)# [:4] # hier # Sample action from policy
            if agent2 == "random":
                action_opp = np.random.uniform(-1,1,4)
            else:
                action_opp = agent2.act(obs_agent2) 
            action = np.hstack((action_agent, action_opp))

        next_state, reward, done, _, _ = env.step(action) # Step # hier das zweite _
        reward *= 100   # hier wurde diese zeile hinzugefügt. numerische Stabilität?
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.max_timesteps else float(not done)
        done = 1 if episode_steps == env.max_timesteps else float(done) # hier diese Zeile wurde hinzugefügt

        trajectory.append((state, action_agent, reward, next_state, mask))

        state = next_state
        obs_agent2 = env.obs_agent_two() # hier
    while len(trajectory) >= args.trajectory_length: # assume one rollout is longer than trajectory_length!! 
        N = args.trajectory_length
        subtraj = trajectory[-N:]
        trajectory = trajectory[:-N]
        memory.push(subtraj) # Append N-step subtrajectory to memory # hier 
    return episode_reward, episode_steps

# hier : 
def testing(env, i_episode, agent1, agent2, test_episodes, writer, saving_interval):
    """
    tests agent and saves agent
    """
    avg_reward = 0.
    for _ in range(test_episodes):
        episode_steps = 0
        state = env.reset()[0] # hier: das [0]
        obs_agent2 = env.obs_agent_two() # hier
        episode_reward = 0
        done = False
        while not done:
            action_agent = agent1.act(state, evaluate=True)# [:4] # hier # Sample action from policy
            if agent2 == "random":
                action_opp = np.random.uniform(-1,1,4)
            else:
                action_opp = agent2.act(obs_agent2)
            action = np.hstack((action_agent, action_opp))
            next_state, reward, done, _, _ = env.step(action)
            episode_steps += 1  
            reward *= 100 # hier 
            episode_reward += reward
            done = 1 if episode_steps == env.max_timesteps else float(done) # hier diese Zeile wurde hinzugefügt
            state = next_state
            obs_agent2 = env.obs_agent_two() # hier
        avg_reward += episode_reward
    avg_reward /= test_episodes

    writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    if i_episode % saving_interval == 0: # hier 
        agent.save_checkpoint(args.env_name, suffix=f"ep_{i_episode}") # hier
        # memory.save_buffer(args.env_name, str(i_episode)) # hier

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(test_episodes, round(avg_reward, 2)))
    print("----------------------------------------")
    log("----------------------------------------")
    log("Test Episodes: {}, Avg. Reward: {}".format(test_episodes, round(avg_reward, 2)))
    log("----------------------------------------")

# hier :
def save_statistics(i_episode, episode_reward):
    with open(f"./runs/QR_SAC_{args.env_name}-stats.pkl", 'wb') as f: # ./ReinforcementLearning/ExcercisesGitHub/exercises mit venv/project_code/models/DDPG
        pkl.dump({"episode" : i_episode, "reward_per_epsiode": episode_reward}, f)

# hier :
# checkpoint_path = r"C:/Users/Home/Documents/M. Sc. ML/Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/SAC2/results/" # C:\Users\Home\Documents\M. Sc. ML\Reinforcement Learning\ExcercisesGitHub\exercises mit venv\project_code\models\SAC2\results

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name)
# env.seed(args.seed)
# env.action_space.seed(args.seed)

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# Agent
action_space_p1 = spaces.Box(-1, +1, (4,), dtype=np.float32)
agent = QRSAC(env.observation_space.shape[0], action_space_p1, args)# env.action_space

# hier bis writer :
# model_path = r"./sac_checkpoint_hockey_vs_rand_all_rew_ep_8400"
# agent.load_checkpoint(model_path)
weak_opp = h_env.BasicOpponent(weak=False) # weak opp ist strong jetzt 

#Tesnorboard
writer = SummaryWriter('runs/{}_QR_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

# agent.save_checkpoint(args.env_name, suffix=f"ep_{0}") # hier

def which_opp(episode):
    if (episode // 1000) % 3 in [0,1]: 
        return True, 'strong' # strong opp
    else: 
        return False, 'self' # weak opp

for i_episode in itertools.count(1):
    episode_reward, episode_steps = rollout_loop(env, agent, "random", memory, total_numsteps) # plays one game
    total_numsteps += episode_steps
    
    if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.step_qr_sac(memory, args.batch_size, updates) # output und input ?

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

    if total_numsteps > args.num_steps:
        break

    save_statistics(i_episode, episode_reward)  # hier
    writer.add_scalar('reward/train', episode_reward, i_episode)
    if i_episode % 50 == 0: # hier diese Zeile
        print("Episode: {}, opp: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, "random", total_numsteps, episode_steps, round(episode_reward, 2)))
        log("Episode: {}, opp: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, "random", total_numsteps, episode_steps, round(episode_reward, 2)))

    test_interval = 100 # hier : Intervall wann getestet werden soll
    if i_episode % test_interval == 0 and args.eval is True: 
        testing(env, i_episode, agent, "random", 10, writer, 10000) # modify: anzahl test episodes and saving interval (see def of function)

env.close()
