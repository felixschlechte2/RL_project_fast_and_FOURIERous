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

# Training Loop
total_numsteps = 0
updates = 0

agent.save_checkpoint(args.env_name, suffix=f"ep_{0}_HER_2") # hier #here

#models_path = r"./checkpoints/sac_checkpoint_hockey_vs_rand_all_rew_ep_340_HER"
#agent.load_checkpoint(models_path, True)


for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()[0] # hier das [0]
    episode_dict = { #here new for Her
                    "state": [],
                    "achieved_goal": [],
                    "desired_goal": [9.16, 2.75, 5.25],
                    "next_state": [],
                    "action": [],
                    "mask": []}
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            action_agent = action[:4] # hier 
            #here: how to implemenent HER here?
        else:
            #action_agent = agent.select_action(state)# [:4] # hier # Sample action from policy #here
            action_agent = agent.select_action(np.concatenate([state, np.array(episode_dict["desired_goal"])], axis=0)) # here
            action_opp = np.random.uniform(-1,1,4)
            action = np.hstack((action_agent, action_opp))

        if len(memory) >args.batch_size: 
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

        next_state, reward, done, _, _ = env.step(action) # Step # hier das zweite _
        
        reward *= 100   # hier wurde diese zeile hinzugefügt. numerische Stabilität?
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.max_timesteps else float(not done)
        done = 1 if episode_steps == env.max_timesteps else float(done) # hier diese Zeile wurde hinzugefügt

        episode_dict['state'].append(state) #here
        episode_dict['achieved_goal'].append(next_state[8]) #here
        episode_dict['next_state'].append(next_state)#here
        episode_dict['action'].append(action_agent) #here
        episode_dict['mask'].append(mask) #here

        #memory.push(episode_dict, action_agent, reward, next_state, mask) # Append transition to memory # hier : ursprünglich action anstelle von action_agent
        #here das drüber weg gemacht
        state = next_state
        if done:
            break
    memory.push(episode_dict)
    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    if i_episode % 10 == 0: # hier diese Zeile
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
        log("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    test_interval = 40 # hier : Intervall wann getestet werden soll
    if i_episode % test_interval == 0 and args.eval is True:
        memory.save_buffer('hockey', f"{i_episode}_HER") # hier #here 
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()[0] # hier: das [0]
            env.render()
            episode_reward = 0
            done = False
            desired_dict = {"desired_goal":[9.16, 2.75, 5.25]} #here added for her
            while not done:
                action_agent = agent.select_action(np.concatenate([state, np.array(desired_dict["desired_goal"])], axis=0), evaluate =True) #here# [:4] # hier # Sample action from policy
                action_opp = np.random.uniform(-1,1,4)
                action = np.hstack((action_agent, action_opp))

                next_state, reward, done, _, _ = env.step(action)
                reward *= 100 # hier 
                episode_reward += reward
                done = 1 if episode_steps == env.max_timesteps else float(done) # hier diese Zeile wurde hinzugefügt


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        agent.save_checkpoint(args.env_name, suffix=f"ep_{i_episode}_HER") # hier

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        log("----------------------------------------")
        log("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        log("----------------------------------------")

env.close()
