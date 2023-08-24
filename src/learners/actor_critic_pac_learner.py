# -*- coding: utf-8 -*-

import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry
from einops import rearrange
from components.standarize_stream import RunningMeanStd

class PACActorCriticLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        
        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.state_value = critic_resigtry[args.state_value_type](scheme, args)
        
        self.critic_params = list(self.critic.parameters()) + list(self.state_value.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        pi = mac_out
        advantages, critic_train_stats = self.train_critic(self.critic, self.target_critic, batch, rewards,
                                                                      critic_mask, terminated, pi)
        actions = actions[:, :-1]
        advantages = advantages.detach()
        # Calculate policy grad with mask

        pi[mask == 0] = 1.0
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)

        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

        training_ratio_now = min(1.0, t_env / (self.args.t_max * self.args.entropy_end_ratio))
        entropy_coef = training_ratio_now * self.args.final_entropy_coef + (1.0-training_ratio_now) * self.args.initial_entropy_coef
        pg_loss = -((advantages * log_pi_taken + entropy_coef * entropy) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

           
            self.logger.log_stat("entropy_coef", entropy_coef, t_env)
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_critic(self, critic, target_critic, batch, rewards, mask, terminated, pi):
        actions = batch["actions"]
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch, compute_all=True)[0][:, :-1]
            target_vals = target_vals.max(dim=3)[0]
        
        target_vals = th.gather(target_vals, -1, actions[:, :-1]).squeeze(-1)
        
        if self.args.standardise_rewards:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        

        if self.args.standardise_rewards:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }
        
        actions = batch["actions"][:, :-1]
        q = critic(batch)[0][:, :-1]
        v = self.state_value(batch)[:, :-1].squeeze(-1)
        
        
        q_curr = th.gather(q, -1, actions).squeeze(-1)
        td_error = (target_returns.detach() - q_curr)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()
        
        td_error_v = (target_returns.detach() - v)
        masked_td_error_v = td_error_v * mask
        loss += (masked_td_error_v ** 2).sum() / mask.sum()

        

        # compute the maximum Q-value and the joint action of the other agents that results in this Q-value
        q_all = critic(batch, compute_all=True)[0][:, :-1]
        q_all = q_all.max(dim=3)[0]
        
        
        
        q_all = th.gather(q_all, -1, actions).squeeze(-1)
        
        advantage = q_all.detach() - v.detach()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((q_curr * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return advantage, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):

        nstep_values = th.zeros_like(values)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
