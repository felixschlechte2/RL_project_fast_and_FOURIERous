import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, Independent


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class MLPBlockActor(nn.Module):
    def __init__(self,dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        self.activation = nn.ReLU()
        self.apply(weights_init_)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.fc2(x) 
        x = self.activation(x)
        return x
    
class MLPBlockCritic(nn.Module):
    def __init__(self,dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim) 
        self.layer_norm2 = nn.LayerNorm(hidden_dim) 
        
        self.activation = nn.ReLU()
        self.apply(weights_init_)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x) 
        x = self.activation(x)
        x = self.fc2(x) 
        x = self.layer_norm2(x) 
        x = self.activation(x) 
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dtype=torch.float32):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

        self.apply(weights_init_)

    def forward(self, x):
        residual = x
        x = self.layernorm1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return residual + x


class SACEncoder(nn.Module):
    def __init__(self,dim, block_type, num_blocks, hidden_dim, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim

        if block_type == "mlp":
            self.layer = MLPBlockActor(dim,hidden_dim)

        elif block_type == "residual":
            self.input_layer = nn.Linear(dim, hidden_dim)
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dtype=dtype) 
                for _ in range(num_blocks)
            ])
            self.norm = nn.LayerNorm(hidden_dim)
        
        self.apply(weights_init_)

    def forward(self, x):
        if self.block_type == "mlp":
            x = self.layer(x)
        elif self.block_type == "residual":
            x = self.input_layer(x)
            for block in self.blocks:
                x = block(x)
            #x = self.norm(x) #here
        return x
    
class SACEncoderCritic(nn.Module):
    def __init__(self,dim, block_type, num_blocks, hidden_dim, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim

        if block_type == "mlp":
            self.layer = MLPBlockCritic(dim,hidden_dim)

        elif block_type == "residual":
            self.input_layer = nn.Linear(dim, hidden_dim)
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dtype=dtype) 
                for _ in range(num_blocks)
            ])
            self.norm = nn.LayerNorm(hidden_dim)
        
        self.apply(weights_init_)

    def forward(self, x):
        if self.block_type == "mlp":
            x = self.layer(x)
        elif self.block_type == "residual":
            x = self.input_layer(x)
            for block in self.blocks:
                x = block(x)
            #x = self.norm(x) #here
        return x
    
class NormalTanhPolicy(nn.Module):
    def __init__(self, hidden_dim, action_dim, dtype=torch.float32, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.action_dim = action_dim
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.apply(weights_init_)

    def forward(self, x):
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        #std = log_std.exp() * temperature #here
        #dist = Normal(mean, std)

        return  mean, log_std #dist #here


class SACActor(nn.Module):
    def __init__(self,dim, block_type, num_blocks, hidden_dim, action_dim, dtype=torch.float32,action_space=None):
        super().__init__()
        self.encoder = SACEncoder(
            dim = dim,
            block_type=block_type,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dtype=dtype
        )
        self.predictor = NormalTanhPolicy(hidden_dim, action_dim, dtype=dtype)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, observations):
        z = self.encoder(observations)
        mean, log_std = self.predictor(z)
        return mean, log_std
    
    def sample(self, state,temperature=1.0):
        mean, log_std = self.forward(state)
        std = log_std.exp()* temperature
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(SACActor, self).to(device)
    
class LinearCritic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, x):
        return self.fc(x)
    

class SACCritic(nn.Module):
    def __init__(self,dim, block_type, num_blocks, hidden_dim, dtype=torch.float32):
        super().__init__()
        self.encoder = SACEncoderCritic(
            dim = dim,
            block_type=block_type,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dtype=dtype
        )
        self.predictor = LinearCritic(hidden_dim)

        self.apply(weights_init_)

    def forward(self, observations, actions):
        inputs = torch.cat([observations, actions], dim=1)
        z = self.encoder(inputs)
        q = self.predictor(z)
        return q
    
class SACClippedDoubleCritic(nn.Module):
    def __init__(self,dim, block_type, num_blocks, hidden_dim, dtype=torch.float32, num_qs=2):
        super().__init__()
        self.q_networks = nn.ModuleList([
            SACCritic(dim,block_type, num_blocks, hidden_dim, dtype=dtype)
            for _ in range(num_qs)
        ])
        self.apply(weights_init_)

    def forward(self, observations, actions):
        qs = [q_net(observations, actions) for q_net in self.q_networks]
        qs = torch.stack(qs, dim=0)
        return qs



class SACTemperature(nn.Module):
    def __init__(self, initial_value=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(initial_value)))

    def forward(self):
        return self.log_temp.exp()
