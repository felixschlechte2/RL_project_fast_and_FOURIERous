import numpy as np
#import hockey.hockey_env as h_env
import own_hockey as h_env
import gymnasium as gym
from importlib import reload
import time
import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import memory as mem   
from feedforward import Feedforward
import matplotlib.pyplot as plt
import pickle
import optparse


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, bins = 5):
        """A wrapper for converting a 1D continuous actions into discrete ones.
        Args:
            env: The environment to apply the wrapper
            bins: number of discrete actions
        """
        assert isinstance(env.action_space, gym.spaces.Box)
        super().__init__(env)
        self.env = env
        self.bins = bins
        self.orig_action_space = env.action_space
        self.action_space = gym.spaces.MultiDiscrete([self.bins, self.bins, self.bins, self.bins])
        

    def action(self, action):
        """ discrete actions from low to high in 'bins'
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        #print("ACTION")
        return self.orig_action_space.low + action/(self.bins-1.0)*(self.orig_action_space.high-self.orig_action_space.low) 
    

    def __getattr__(self, name):
        # If the attribute exists in the environment, forward the call to the environment
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            # If the attribute doesn't exist, raise an error or handle it as needed
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'") 
    

""" Q Network, input: observations, output: q-values for all actions """
class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, 
                 hidden_sizes=[100,100], learning_rate = 0.0002):
        super().__init__(input_size=observation_dim, 
                         hidden_sizes=hidden_sizes, 
                         output_size=sum(action_dim))
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        # The L1 loss is often easier for choosing learning rates etc than for L2 (MSELoss)
        #  Imagine larger q-values (in the hundreds) then an squared error can quickly be 10000!, 
        #  whereas the L1 (absolute) error is simply in the order of 100. 
        self.loss = torch.nn.SmoothL1Loss()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
    def fit(self, observations, actions, targets):
        # TODO: complete this
        self.train()
        self.optimizer.zero_grad()
        # Forward pass
        #pred = self.Q_value(observations, actions)
        # Compute Loss
        #loss = self.loss(pred, targets)
        # Backward pass
        #loss.backward()
        #self.optimizer.step()
        #return loss.item()
        
        pred = self.forward(observations.float())

        # Split predictions by action dimension
        #print(pred.shape)
        q_values_split = self._split_q_values(pred)
        #print(len(q_values_split))
        # Compute loss for each dimension
        total_loss = 0
        for i in range(len(self.action_dim)):
            acts = actions[:, i]
            targs = targets[:, i].float()
            q_values = q_values_split[i].gather(1, acts[:, None])  # Get Q-values for taken actions
            #print(targs[:,None])
            #print(q_values)
            total_loss += self.loss(q_values, targs[:, None])

        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
    
    def Q_value(self, observations, actions):
        # compute the Q value for the give actions
        # Hint: use the torch.gather function select the right outputs 
        # Complete this
        qs = self.forward(observations)   
        q_values_split = self._split_q_values(qs)
        q_values = []
        for i in range(len(self.action_dim)):
            q_values.append(q_values_split[i].gather(1, actions[:, i][:, None]))
        return torch.cat(q_values, dim=1)
    
    def maxQ(self, observations):
        # compute the maximal Q-value
        # Complete this
        pred = self.forward(torch.from_numpy(observations).float())
        q_values_split = self._split_q_values(pred)
        return torch.cat([torch.max(q_values, dim=1, keepdim=True)[0] for q_values in q_values_split], dim=1)

    def greedyAction(self, observations):
        # this computes the greedy action
        
        pred = self.forward(torch.from_numpy(observations).float())
        
        #print(f"pred={pred}")
        q_values_split = self._split_q_values(pred.unsqueeze(0))
        #print(f"split={q_values_split}")
        return np.array([torch.argmax(q_values).item() for q_values in q_values_split])
    
    
    def _split_q_values(self, q_values):
        """
        Helper function to split flat Q-values into separate tensors for each action dimension.
        """
        
        split_indices = torch.cumsum(torch.tensor(self.action_dim), dim=0)[:-1]
        
        return torch.tensor_split(q_values, split_indices, dim=1)
    

class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.    
    """
    def __init__(self, observation_space, action_space, **userconfig):
        
        if not isinstance(observation_space, gym.spaces.box.Box):
            raise Exception('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        
        #for _ in range(4):
        #    if not isinstance(action_space[_], gym.spaces.discrete.Discrete):
        #        raise Exception('Action space {} incompatible with {}.' \
        #                           ' (Reqire Discrete.)'.format(action_space, self))

        if isinstance(action_space, gym.spaces.MultiDiscrete):  # Use MultiDiscrete for multiple discrete actions
            self._action_space = action_space
            #self._action_n = sum(action_space.nvec)
            #self._action_n = len(action_space.nvec)
            self._action_n = [a.n for a in action_space]
        else:
            raise Exception('Action space {} incompatible with {}.' \
                                   ' (Require MultiDiscrete.)'.format(action_space, self))
        
        self._observation_space = observation_space
        self._config = {
            "eps": 0.05,            # Epsilon in epsilon greedy policies                        
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0002, #0.0002 normalerweise 
            # add additional parameters here  
            "target_update_count": 20
        }
        self._config.update(userconfig)        
        self._eps = self._config['eps']
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])
        
        # complete here
        self.target_net = QFunction(gym.spaces.utils.flatdim(observation_space), self._action_n)
        self.Q = QFunction(gym.spaces.utils.flatdim(observation_space), self._action_n)
        self.train_iter = 0
            
    def _update_target_net(self):        
        # complete here
        # Hint: use load_state_dict() and state_dict() functions
        self.target_net.load_state_dict(self.Q.state_dict())
    
    def act(self, obs, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(obs)
            #print(f"eps = {action}")
        else:  
            action = np.array([np.random.choice(self._action_space.nvec[i]) for i in range(len(self._action_space.nvec))])
            #print(f"random = {action}")
            #action = [a.sample() for a in self._action_space]
        #print(self._action_n)
        return action
    
    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q.state_dict())
            
    def restore_state(self, state):
        self.Q.load_state_dict(state)
        #self.policy.load_state_dict(state[1])
        self._copy_nets()

    def _copy_nets(self):
        self.target_net.load_state_dict(self.Q.state_dict())
        #self.policy_target.load_state_dict(self.policy.state_dict())


    def train(self, iter_fit=32):
        losses = []
        if self.train_iter % 100 == 0:
            self._update_target_net()
        
        for _ in range(iter_fit):
                # Sample a batch from the replay buffer
            data = self.buffer.sample(batch=self._config["batch_size"])
        
            # Extract data from the batch
            observations = torch.from_numpy(np.stack(data[:, 0]))  # Observations
            actions = torch.from_numpy(np.stack(data[:, 1]))  # Actions (shape: [batch_size, 4])
            rewards = torch.from_numpy(np.stack(data[:, 2])).float()  # Rewards (shape: [batch_size])
            next_states = np.stack(data[:, 3])  # Next states
            terminal = torch.from_numpy(np.stack(data[:, 4]))  # Terminal flags (shape: [batch_size])

            # Compute target Q-values for each action dimension
            target_max = self.target_net.maxQ(next_states)  # Shape: [batch_size, 4]
        
            # Expand rewards to match action dimensions
            rewards_expanded = rewards[:, None].repeat(1, actions.shape[1])  # Shape: [batch_size, 4]

            # Compute targets for each dimension
            targets = rewards_expanded + self._config["discount"] * (~terminal[:, None]) * target_max

            # Fit the Q-function for all dimensions
            fit_loss = self.Q.fit(observations, actions, targets)

            losses.append(fit_loss)

        return losses



def main():

    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='float',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='float',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=None,
                         help='random seed (default %default)')
    opts, args = optParser.parse_args()

    env = h_env.HockeyEnv()
    if isinstance(env.action_space, gym.spaces.Box):
        env = DiscreteActionWrapper(env,3)

    ac_space = env.action_space
    o_space = env.observation_space

    o, info = env.reset()
    #_ = env.render()

    max_episodes = 3000
    max_steps = 500
    eps = 0.4          
    lr  =  0.0002
    train_iter = 64
    log_interval = 100
    env_name = "hockey"

    q_agent = DQNAgent(o_space, ac_space, discount=0.95, eps = eps, target_update_count=20)
    q_agent2 = DQNAgent(o_space, ac_space, discount=0.95, eps = eps, target_update_count=20)
    player2 = h_env.BasicOpponent()

    q_agent.Q.predict(o)

    
    rewards = []
    lengths = []
    losses = []

    def save_statistics():
        with open(f"./results/DQN_{env_name}-eps{eps}-t{train_iter}-l{lr}--stat.pkl", 'wb') as f: # ./ReinforcementLearning/ExcercisesGitHub/exercises mit venv/project_code/models/DDPG
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    #q_agent.restore_state(torch.load("./results/DQN_random_2000.pth",weights_only=True))
    #q_agent2.restore_state(torch.load("./results/DQN_hockey_defending_81000-eps0.2-t64-l0.0002.pth",weights_only=True))
    q_agent.restore_state(torch.load("./results/DQN_reward_5.pth",weights_only=True))

    rewards = []
    losses = []

    score = [0,0,0] #[lose,draw,win]
    total_reward = 0
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    for i in range(1,max_episodes+1):
        obs, _info = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            if i % 100 == 0:
                env.render()
            
            #a2 = q_agent2.act(obs)
            a2 = player2.act(obs_agent2)
            # a2 = np.random.uniform(-1,1,4)
            a2 = [x+1 for x in a2] 
            a1 = q_agent.act(obs)

            obs_new, r, d, t, info = env.step(np.hstack([a1,a2]))
            obs_agent2 = env.obs_agent_two()   
            q_agent.store_transition((obs, a1, r, obs_new, d))  
            obs = obs_new
            total_reward+= r
            if d or t: break
        score[info["winner"]+1] += 1
        losses.extend(q_agent.train(train_iter))
        rewards.append(total_reward)
        lengths.append(_)



        # save every 500 episodes
        if i % 500 == 0: # hier
            print("########## Saving a checkpoint... ##########")
            torch.save(q_agent.state(), f'./results/DQN_reward_6.pth')
            save_statistics()

        # logging
        if i % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i, avg_length, avg_reward))
    
    print(score)
    print('----------------finished training defending-------------------------')

if __name__ == '__main__':
    main()


