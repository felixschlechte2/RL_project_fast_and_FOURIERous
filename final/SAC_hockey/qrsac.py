import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from qrmodel import GaussianPolicy, QNetwork, DeterministicPolicy
from utils import log, run_name


class QRSAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.N_STEP = args.trajectory_length # hier
        self.num_quantile = args.num_quantile

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")
        log(str(self.device))

        self.critic = QNetwork(num_inputs, action_space.shape[0], self.num_quantile, args.hidden_size).to(device=self.device) # hier : added num_quantiles
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], self.num_quantile, args.hidden_size).to(self.device) # hier : added num_quantiles
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def act(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def step_qr_sac(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size, gamma=self.gamma, N_STEP=self.N_STEP) # hier gamma und N_STEP hinzugefügt

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target.quantiles(next_state_batch, next_state_action)
            quantile_next = self.minimum_quantile(batch_size, qf1_next_target, qf2_next_target) # hier + nächste 3 Zeilen
            entropy = self.alpha * next_state_log_pi
            target = reward_batch + (self.gamma**self.N_STEP) * mask_batch * (quantile_next- entropy)

        #     min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
        #     next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # # critic update: 
        qf1, qf2 = self.critic.quantiles(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf_loss = qf1_loss + qf2_loss

        q1_loss = self.quantile_loss(qf1, target, batch_size).mean()
        q2_loss = self.quantile_loss(qf2, target, batch_size).mean()
        qf_loss = q1_loss + q2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # policy update: 
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic.q_values(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def quantile_loss(self, quantile, target, batch_size):
        y = target.view(batch_size, -1, 1)  # (N, M, 1)
        x = quantile.view(batch_size, 1, -1)        # (N, 1, M)

        # Berechnung des Huber Loss
        diff = y - x  # (N, M, M)
        huber_loss = torch.where(
            torch.abs(diff) < 1.0,
            0.5 * diff ** 2,
            torch.abs(diff) - 0.5
        )  # Huber Loss (N, M, M)

        # Berechnung der Quantils-Tau-Werte
        steps = torch.arange(self.num_quantile, dtype=torch.float32, device=quantile.device)  # (M,)
        taus = ((steps + 1) / self.num_quantile).view(1, 1, -1)  # (1, 1, M)
        taus_minus = (steps / self.num_quantile).view(1, 1, -1)  # (1, 1, M)
        taus_hat = (taus + taus_minus) / 2.0  # Mittlere Tau-Werte (1, 1, M)

        # Berechnung des quantilen Verlustes
        delta = (diff.detach() < 0.0).float()  # (N, M, M)
        element_wise_loss = torch.abs(taus_hat - delta) * huber_loss  # (N, M, M)

        # Reduktion des Losses
        loss = element_wise_loss.sum(dim=2).mean(dim=1)  # Skalar
        return loss

    
    def minimum_quantile(self, batch_size, quantile1, quantile2):
        with torch.no_grad():
            q1 = quantile1.mean(dim=1,  keepdims=True)
            q2 = quantile2.mean(dim=1,  keepdims=True)

            stacked_q = torch.cat([q1, q2], dim=1) 
            min_indices = torch.argmin(stacked_q, dim=1)
            stacked_quantile = torch.stack([quantile1, quantile2], dim=1)
            batch_indices = torch.arange(batch_size)
            minimum_quantile = stacked_quantile[batch_indices, min_indices, :]
            return minimum_quantile



    # Save model parameters
    def save_checkpoint(self, env_name, args, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/checkpoint_{}_{}_{}".format(env_name, suffix, run_name)
        print('Saving models to {}'.format(ckpt_path))
        log('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        log('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu')) # hier : added weights_only and map_location
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
