import argparse
import datetime
import gymnasium as gym 
import numpy as np
import itertools
import torch
from sac import SAC
from qrsac import QRSAC
# from torch.utils.tensorboardx import SummaryWriter
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import pickle as pkl  
from utils import log, run_name, statistics

import hockey.hockey_env as h_env  
from gymnasium import spaces  

env = h_env.HockeyEnv() 

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
parser.add_argument('--start_steps', type=int, default= -1, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--log_test_save', type=int, nargs=3, default=[50,100,1000], metavar='N',
                    help='intervals for logs, testing, saving (default: [10,100,1000])') 
parser.add_argument('--trajectory_length', type=int, default=7, metavar='N',
                    help='length of trajectory that is captured (default: 7)')  
parser.add_argument('--num_quantile', type=int, default=16, metavar='N',
                    help='number of quantiles (default: 16)')  
args = parser.parse_args()

args.env_name = "hockey_vs_4_all_rew"

with open(f"./run_info/{run_name}_args.pkl", "wb") as f:
    pkl.dump(args, f)

episode_stats = statistics(["episode", "reward", "opponent"], fr"./runs/stats_{args.env_name}-{run_name}_episode.pkl")
test_stats = statistics(['episode', 'avg_reward/test'], fr"./runs/stats_{args.env_name}-{run_name}_test.pkl")

 
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
agent = SAC(env.observation_space.shape[0], action_space_p1, args)# env.action_space
old = SAC(env.observation_space.shape[0], action_space_p1, args)# env.action_space
# opp = SAC(env.observation_space.shape[0], action_space_p1, args)
qr_sac = QRSAC(env.observation_space.shape[0], action_space_p1, args)

model_path = r"./old_models/sac_rnn2_run1"
agent.load_checkpoint(model_path)
old.load_checkpoint(model_path)
model_path = r"./old_models/qr_sac_run6"
qr_sac.load_checkpoint(model_path)
# opp.load_checkpoint(model_path)
weak_opp = h_env.BasicOpponent()
strong_opp = h_env.BasicOpponent(weak=False) # weak opp ist strong jetzt  # weak=False

#Tesnorboard
writer = SummaryWriter('runs/{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "", run_name))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

# agent.save_checkpoint(args.env_name, suffix=f"ep_{0}") 

def which_opp(episode):
    if (episode // 500) % 4 == 0: 
        return weak_opp, "weak_opp" # strong opp
    if (episode // 500) % 4 == 1: 
        return strong_opp, "strong_opp" # strong opp
    if (episode // 500) % 4 == 3: 
        return old, "self_old" # strong opp
    if (episode // 500) % 4 == 2: 
        return qr_sac, "qr_sac_run6" # strong opp

for i_episode in itertools.count(1):
    opp, opp_name = which_opp(i_episode)
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()[0] 
    agent.reset()
    obs_agent2 = env.obs_agent_two()  

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            action_agent = action[:4]  
        else:
            hidden_obs = agent.policy.hidden_memory
            state_hidden_obs = np.concatenate((state, hidden_obs.cpu().detach().numpy()[0]), axis=0)
            action_agent = agent.act(state)# [:4]  # Sample action from policy
            action_opp = opp.act(obs_agent2) 
            action = np.hstack((action_agent, action_opp))

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
        hidden_obs = agent.policy.hidden_memory
        next_state, reward, done, _, _ = env.step(action) # Step 
        next_state_hidden_obs = np.concatenate((next_state, hidden_obs.cpu().detach().numpy()[0]), axis=0)
        reward *= 100   
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.max_timesteps else float(not done)
        done = 1 if episode_steps == env.max_timesteps else float(done) 

        memory.push(state_hidden_obs, action_agent, reward, next_state_hidden_obs, mask) # Append transition to memory 

        state = next_state
        obs_agent2 = env.obs_agent_two() 
    
    episode_stats.save_data(["episode", "reward", "opponent"], [i_episode, episode_reward, opp_name])

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    if i_episode % args.log_test_save[0] == 0: 
        print("Episode: {}, opp: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, opp_name, total_numsteps, episode_steps, round(episode_reward, 2)))
        log("Episode: {}, opp: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, opp_name, total_numsteps, episode_steps, round(episode_reward, 2)))

    test_interval = args.log_test_save[1] # Intervall wann getestet werden soll
    if i_episode % test_interval == 0 and args.eval is True: 
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()[0] 
            agent.reset()
            obs_agent2 = env.obs_agent_two() 
            episode_reward = 0
            done = False
            while not done:
                action_agent = agent.act(state, evaluate=True)# [:4]  # Sample action from policy
                action_opp = strong_opp.act(obs_agent2)   # np.random.uniform(-1,1,4)
                action = np.hstack((action_agent, action_opp))

                next_state, reward, done, _, _ = env.step(action)
                reward *= 100  
                episode_reward += reward
                done = 1 if episode_steps == env.max_timesteps else float(done) 


                state = next_state
                obs_agent2 = env.obs_agent_two() 
            avg_reward += episode_reward
        avg_reward /= episodes
        test_stats.save_data(['episode', 'avg_reward/test'], [i_episode, avg_reward])

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        log("----------------------------------------")
        log("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        log("----------------------------------------")
    
    if i_episode % args.log_test_save[2] == 0:  
        agent.save_checkpoint(args.env_name, args, suffix=f"ep_{i_episode}") 
        test_stats.save_statistics()
        episode_stats.save_statistics()

    if i_episode % 5000 == 0: 
        memory.save_buffer(args.env_name, args, f"ep_{i_episode}")
        try:
            path = fr"./checkpoints/checkpoint_{args.env_name}_ep_{i_episode}_{run_name}"
            old.load_checkpoint(path)
        except Exception as e:
            print(e)
            log(str(e))

env.close()

memory.save_buffer(args.env_name, args, "end")