import random
import numpy as np
import os  # hier 
import pickle  # hier 

def log(log):
    log_file = "./log_file.txt"
    with open(log_file, "a") as file:
        file.write(log)
        file.write("\n")

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, input): # hier : state, action, reward, next_state, done
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = input # hier : (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, gamma, N_STEP): # hier : ganze Funktion
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        discounting = np.power(gamma, np.arange(N_STEP))
        state = []
        action = []
        reward = []
        next_state = []
        done = []
        for traj in batch:  
            state.append(traj[0][0])
            action.append(traj[0][1]) 
            reshaped_rewards = np.array([tup[2] for tup in traj])
            reward.append(np.sum(reshaped_rewards * discounting))  # adding discounted reward for trajectory
            next_state.append(traj[-1][3])
            done.append(traj[-1][4])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, args, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/qr_sac_buffer_{}_{}_N_STEPS{}_alpha{}_quant{}_bs{}_up_ep{}".format(env_name, suffix, args.trajectory_length, args.alpha, 
                                                                                                        args.num_quantile, args.batch_size, 
                                                                                                        args.updates_per_step)
        print('Saving buffer to {}'.format(save_path))
        log('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))
        log('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity