import argparse
import datetime
import gymnasium as gym # type: ignore # hier : vorher nur gym
import numpy as np # type: ignore
import itertools
import torch # type: ignore
from sac import SAC
from tensorboardX import SummaryWriter # type: ignore
from replay_memory import ReplayMemory
import random
import pickle as pkl

import hockey.hockey_env as h_env # hier 
from gymnasium import spaces # type: ignore # hier 

class statistics():
    def __init__(self, attributes, saving_path):
        self.stats = {}
        for attr in attributes:
            self.stats[attr] = []
        self.saving_path = saving_path

    def save_data(self, attributes, values):
        if len(attributes) != len(values):
            log("stats saving error: both list must be of equal length")
            return
        else:
            for i in range(len(attributes)):
                self.stats[attributes[i]].append(values[i])
    def save_statistics(self):
        log("save statistics")
        with open(self.saving_path, "wb") as f:
            pkl.dump(self.stats,f)
    def load_statistics(self,path):
        print("load statistics")
        with open(path,"rb") as f:
            self.stats = pkl.load(f)
    def return_values(self,attribute):
        return self.stats[attribute]


test_stats = statistics(['episode', 'avg_reward/test'], fr"./statistics/stats_test_HER.pkl")

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
parser.add_argument('--start_steps', type=int, default=0, metavar='N', #HERE, such that no random action, because we load checkpoint in
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()


action_space_p1 = spaces.Box(-1, +1, (4,), dtype=np.float32)

# Agent #here
agent = SAC(env.observation_space.shape[0]+2, action_space_p1, args)# env.action_space #added +3 for HER #here 2
# Opponents #here
self_play = SAC(env.observation_space.shape[0]+2, action_space_p1, args)# here
player2_weak = h_env.BasicOpponent() #here
player2_strong = h_env.BasicOpponent(weak = False) #here

# # initialize if checkpoint is given
# models_path = r"./checkpoints/HER_ep_17400"
# agent.load_checkpoint(models_path, True)
# self_play.load_checkpoint(models_path, True)

#Tesnorboard #here
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),"hockey_vs_rand_all_rew",
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size,4,env, args.seed) #here added env, k_future

# Training Loop
total_numsteps = 0
updates = 0

# score: loss,draw,win #here
score = [0,0,0] 

# all opponents
opponents =[None, player2_weak, player2_strong,self_play]
current_opp = 0
opponent = opponents[current_opp]

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()[0] # hier das [0]
    if opponent != None:
        obs_opp = env.obs_agent_two() #here

    # different replay buffer for HER #here
    episode_dict = { 
                    "state": [],
                    "achieved_goal": [],
                    "desired_goal": [3.7, 0],
                    "next_state": [],
                    "action": [],
                    "reward": []}
    
    while not done:
        
        # random action for the first args.start_steps for exploration after that select_action 
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            action_agent = action[:4] # hier 
        else:
            action_agent = agent.select_action(np.concatenate([state, np.array(episode_dict["desired_goal"])], axis=0), evaluate =True) #here# [:4] # hier # Sample action from policy

            # action for each possible opponent
            if opponent == None:
                action_opp = np.random.uniform(-1,1,4) 
            elif opponent == self_play:
                action_opp = opponent.select_action(np.concatenate([obs_opp, [3.7,0]], axis=0))
            else:
                action_opp = opponent.act(obs_opp) 

            action = np.hstack((action_agent, action_opp))

        if len(memory) >=args.batch_size: 
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

        next_state, reward, done, _, info = env.step(action) # Step # hier das zweite _ #here info
        if opponent != None:
            obs_opp = env.obs_agent_two() #here
        
        reward *= 100   # hier wurde diese zeile hinzugefügt. numerische Stabilität?
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.max_timesteps else float(not done)
        done = 1 if episode_steps == env.max_timesteps else float(done) # hier diese Zeile wurde hinzugefügt

        episode_dict['state'].append(state) #here
        episode_dict['achieved_goal'].append(next_state[12:14]) #here
        episode_dict['next_state'].append(next_state)#here
        episode_dict['action'].append(action_agent) #here
        episode_dict['reward'].append(reward) #here

        state = next_state
        if done:
            break
    
    # update score #here
    score[info["winner"]+1] +=1

    # add to replay buffer #here
    memory.push(episode_dict)

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    # log/ print every 10 episodes # hier diese Zeile
    if i_episode % 10 == 0: 
        print("Episode: {}, total numsteps: {}, win_percentage: {}".format(i_episode, total_numsteps, score[2]/sum(score))) 
        log("Episode: {}, total numsteps: {}, win_percentage: {}".format(i_episode, total_numsteps, score[2]/sum(score)))


    test_interval = 100 # hier : Intervall wann getestet werden soll
    if i_episode % test_interval == 0 and args.eval is True:
        score = [0,0,0] # here
        if i_episode % 10000 == 0:
            memory.save_buffer(f"{i_episode}_HER") # here 
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()[0] # hier: das [0]
            if opponent != None:
                obs_opp = env.obs_agent_two() #here
            episode_reward = 0
            done = False
            while not done:
                action_agent = agent.select_action(np.concatenate([state, np.array([3.7, 0])], axis=0), evaluate =True) #here# [:4] # hier # Sample action from policy
                if opponent == None:
                    action_opp = np.random.uniform(-1,1,4) 
                elif opponent == self_play:
                    action_opp = opponent.select_action(np.concatenate([obs_opp, np.array([3.7,0])], axis=0)) 
                else:
                    action_opp = opponent.act(obs_opp) 

                action = np.hstack((action_agent, action_opp))

                next_state, reward, done, _, info = env.step(action)

                if opponent != None:
                    obs_opp = env.obs_agent_two() #here
                
                reward *= 100 # hier 
                episode_reward += reward
                done = 1 if episode_steps == env.max_timesteps else float(done) # hier diese Zeile wurde hinzugefügt
                if done:
                    break

                state = next_state
            avg_reward += episode_reward
            score[info["winner"]+1] +=1 #here
        avg_reward /= episodes
        test_stats.save_data(['episode', 'avg_reward/test'], [i_episode,avg_reward]) #here

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        if i_episode % 200 == 0:
            test_stats.save_statistics()
            agent.save_checkpoint(suffix=f"ep_{i_episode}") # hier
        
        # log to see the process
        if current_opp == 0:
            opp_string = "Random Opponent"
        elif current_opp == 1:
            opp_string = "Weak Opponent"
        elif current_opp == 2:
            opp_string = "Strong Opponent"
        else:
            opp_string = "Self Play"
        
        print("----------------------------------------")
        print("Test Episodes: {}, Opponent: {}, Avg. Reward: {}, win_percentage: {}, score: {}".format(episodes, opp_string, round(avg_reward, 2),score[2]/sum(score), score))
        print("----------------------------------------")
        log("----------------------------------------")
        log("Test Episodes: {}, Opponent: {}, Avg. Reward: {}, win_percentage: {}, score: {}".format(episodes, opp_string, round(avg_reward, 2),score[2]/sum(score), score))
        log("----------------------------------------")
        
        
    # choosing which agent to use
    if i_episode % test_interval == 0 and args.eval is True:

        #choose other agent if winrate is too high or too low
        if (score[2]/sum(score) >= 0.8 or score[0] == 0) and current_opp != 3:
            current_opp = (current_opp + 1) % len(opponents)
            opponent = opponents[current_opp]
        elif (score[2]/sum(score) <= 0.1 or score[0] == 0) and current_opp != 0:
            current_opp = (current_opp - 1) % len(opponents)
            opponent = opponents[current_opp]
        else:
            # avoid overfitting by introducing a little chance that another agent is choosen
            if random.random() <= 0.1:
                current_opp =random.randint(1,3)
                opponent = opponents[current_opp]
        
        #reset score
        score = [0,0,0]

    # update self_play opponent     
    if i_episode % 400 == 0:
        self_play.load_checkpoint(rf"./checkpoints/HER_ep_{i_episode-200}", True)

env.close()
