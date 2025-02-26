import random
import numpy as np # type: ignore
import os   
import pickle   
from copy import deepcopy as dc

class ReplayMemory:
    def __init__(self, capacity,k_future,env, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.env = env
        self.future_p = 1 - (1. / (1 + k_future)) # for HER

    # add whole episodes
    def push(self, episode_dict):
        self.buffer.append(episode_dict)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):

        # pick random episodes and random time indices for them
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
            desired_goals.append(dc(self.buffer[episode]["desired_goal"])) 
            next_achieved_goals.append(dc(self.buffer[episode]["achieved_goal"][timestep]))
            next_states.append(dc(self.buffer[episode]["next_state"][timestep]))
            reward.append(dc(self.buffer[episode]["reward"][timestep]))
    
        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)
        reward= np.vstack(reward)

        # select random episodes that get changed for HER
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # for every episode, consider all good time steps in which the puck is in the enemy quarter
        future_t =[]
        for i in range(len(ep_indices)):
            # good achieved goals:
            candidate = [
                t for t in range(time_indices[i] + 1, episode_lengths[i]) 
                if self.buffer[ep_indices[i]]["achieved_goal"][t][0] > 1.85
            ]
            # add random time steps if no good state was reached
            if len(candidate) >0:
                future_t.append(np.random.choice(candidate))
            else:
                future_t.append(-1)
        future_t = np.array([int(x) for x in future_t])

        # change desired_goal to the good one's that were achieved
        future_ag = np.array([
            self.buffer[ep]["achieved_goal"][ft] if ft != -1 else desired_goals[idx]
            for idx, (ep, ft) in enumerate(zip(ep_indices[her_indices], future_t[her_indices]))
            ])
        desired_goals[her_indices] = future_ag 

        # calculate distance of next position of the puck and the desired position of the puck
        distance = []
        for sub1, sub2 in zip(next_achieved_goals[her_indices], desired_goals[her_indices]):
            distance.append(max(abs(a - b) for a, b in zip(sub1, sub2)))
        
        # calculate new rewards based on distance
        success_threshold = 1 
        rewards = np.array([float(-x) if abs(x)> success_threshold else 10 for x in distance])
 
        reward[her_indices] = rewards[:,np.newaxis]

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
