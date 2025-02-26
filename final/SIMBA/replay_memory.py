import random
import numpy as np # type: ignore
import os  # hier 
import pickle  # hier 
from copy import deepcopy as dc #here
import time #here

class ReplayMemory:
    def __init__(self, capacity,k_future,env, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.env = env
        self.future_p = 1 - (1. / (1 + k_future))

    #here everything
    #def push(self, episode_dict, action, reward, next_state, done, desired_goal):
    #    if len(self.buffer) < self.capacity:
    #        self.buffer.append(None)
    #    self.buffer[self.position] = (episode_dict, action, reward, next_state, done)
    #    self.position = (self.position + 1) % self.capacity
    
    #new here
    def push(self, episode_dict):
        self.buffer.append(episode_dict)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        #assert len(self.memory) <= self.capacity

    def sample(self, batch_size):
        #batch = random.sample(self.buffer, batch_size)
        #state, action, reward, next_state, done = map(np.stack, zip(*batch))

        ep_indices = np.random.randint(0, len(self.buffer), batch_size)
        episode_lengths = [len(self.buffer[i]["next_state"]) for i in ep_indices]
        time_indices = [np.random.randint(0, episode_lengths[i]) for i in range(batch_size)]

        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []
        reward = []

        for episode, timestep in zip(ep_indices, time_indices):
            states.append(dc(self.buffer[episode]["state"][timestep]))
            actions.append(dc(self.buffer[episode]["action"][timestep]))
            desired_goals.append(dc(self.buffer[episode]["desired_goal"])) #here different
            next_achieved_goals.append(dc(self.buffer[episode]["achieved_goal"][timestep])) #here different from git
            next_states.append(dc(self.buffer[episode]["next_state"][timestep]))
            reward.append(dc(self.buffer[episode]["reward"][timestep]))
    
        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)
        reward= np.vstack(reward)

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)

        #for every epsiode, note good states for her, aka in last quarter enemy half
        #future_candidates = []
        future_t =[]
        for i in range(len(ep_indices)):
            candidate = [
                t for t in range(time_indices[i] + 1, episode_lengths[i]) 
                if self.buffer[ep_indices[i]]["achieved_goal"][t][0] > 1.85  #achieved goal is good
            ]
            #future_candidates.append(candidate)
            if len(candidate) >0:
                future_t.append(np.random.choice(candidate))
            else:
                #future_offset = np.random.randint(1, episode_lengths[i] - time_indices[i] + 1)
                #future_t.append(np.minimum(time_indices[i] + future_offset, episode_lengths[i] - 1)) #so that we are for sure < episode_len
                future_t.append(-1)
        future_t = np.array([int(x) for x in future_t])

        #future_ag = np.array([self.buffer[ep]["achieved_goal"][ft] for ep, ft in zip(ep_indices[her_indices], future_t[her_indices])])
        future_ag = np.array([ #here
            self.buffer[ep]["achieved_goal"][ft] if ft != -1 else desired_goals[idx]
            for idx, (ep, ft) in enumerate(zip(ep_indices[her_indices], future_t[her_indices]))
            ])
        desired_goals[her_indices] = future_ag #change desired_goal to the one's we achieved

        #here everything
        distance = []
        for sub1, sub2 in zip(next_achieved_goals[her_indices], desired_goals[her_indices]):  # Iterate over corresponding sub-arrays
            distance.append(max(abs(a - b) for a, b in zip(sub1, sub2)))
        
        # Define a threshold for success (e.g., 5 cm)
        success_threshold = 1 #here statt 0.1

        # Reward function (sparse or dense)
        #rewards = -distance  # Dense reward (closer is better)
        
        # If object reaches goal, give a large positive reward
        rewards = np.array([float(-x) if abs(x)> success_threshold else 10 for x in distance])
        
        # or += ??
        #reward = np.array(rewards)
        reward[her_indices] = rewards[:,np.newaxis]#[her_indices]
        #rewards = np.expand_dims(rewards, 1) #here
        #reward = np.expand_dims(reward, 1) #here

        return states, actions, reward, next_states, desired_goals


    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}".format(suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity