import math
import torch
import pickle as pkl

run_name = "sac_run3"

def log(log):
    log_file = f"./run_info/log_file_{run_name}.txt"
    with open(log_file, "a") as file:
        file.write(log)
        file.write("\n")

class statistics():
    def __init__(self, attributes, saving_path):
        self.stats = {}
        for attr in attributes:
            self.stats[attr] = []
        self.saving_path = saving_path
        # self.save_statistics()

    def save_data(self, attributes, values):
        if len(attributes) != len(values):
            print("stats saving error: both list must be of equal length!")
            log("stats saving error: both list must be of equal length!")
            return
        else:
            for i in range(len(attributes)):
                self.stats[attributes[i]].append(values[i])
    def save_statistics(self):
        log("save statistics")
        with open(self.saving_path, "wb") as f:
            pkl.dump(self.stats, f)
    def load_statistics(self, path):
        print("load statistics")
        with open(path, "rb") as f:
            self.stats = pkl.load(f)
    def return_values(self, attribute):
        return self.stats[attribute]

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)