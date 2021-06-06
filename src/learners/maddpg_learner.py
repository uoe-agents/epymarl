import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.maddpg import MADDPGCritic
import torch as th
from torch.optim import RMSprop, Adam
from controllers.maddpg_controller import gumbel_softmax
from modules.critics import REGISTRY as critic_registry

class MADDPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.agent_optimiser = Adam(params=self.agent_params, lr=self.args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=self.args.lr)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.last_target_update_episode = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions_onehot"]
        terminated = batch["terminated"][:, :-1].float()
        rewards = rewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        mask = 1 - terminated
        batch_size = batch.batch_size

        if self.args.standardise_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # Train the critic
        inputs = self._build_inputs(batch)
        actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
        q_taken = self.critic(inputs[:, :-1], actions[:, :-1].detach())
        q_taken = q_taken.view(batch_size, -1, 1)

        # Use the target actor and target critic network to compute the target q
        self.target_mac.init_hidden(batch.batch_size)
        target_actions = []
        for t in range(1, batch.max_seq_length):
            agent_target_outs = self.target_mac.target_actions(batch, t)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        target_actions = target_actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
        target_vals = self.target_critic(inputs[:, 1:], target_actions.detach())
        target_vals = target_vals.view(batch_size, -1, 1)

        targets = rewards.reshape(-1, 1) + self.args.gamma * (1 - terminated.reshape(-1, 1)) * target_vals.reshape(-1, 1)

        td_error = (q_taken.view(-1, 1) - targets.detach())
        masked_td_error = td_error * mask.reshape(-1, 1)
        loss = (masked_td_error ** 2).mean()

        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # Train the actor
        self.mac.init_hidden(batch_size)
        pis = []
        actions = []
        for t in range(batch.max_seq_length-1):
            pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1)
            actions.append(gumbel_softmax(pi, hard=True))
            pis.append(pi)
        actions = th.cat(actions, dim=1)
        actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)

        new_actions = []
        for i in range(self.n_agents):
            temp_action = th.split(actions[:, :, i, :], self.n_actions, dim=2)
            actions_i = []
            for j in range(self.n_agents):
                if i == j:
                    actions_i.append(temp_action[j])
                else:
                    actions_i.append(temp_action[j].detach())
            actions_i = th.cat(actions_i, dim=-1)
            new_actions.append(actions_i.unsqueeze(2))
        new_actions = th.cat(new_actions, dim=2)

        pis = th.cat(pis, dim=1)
        pis[pis==-1e10] = 0
        pis = pis.reshape(-1, 1)
        q = self.critic(inputs[:, :-1], new_actions)
        q = q.reshape(-1, 1)
        mask = mask.reshape(-1, 1)

        # Compute the actor loss
        pg_loss = -(q * mask).mean() + self.args.reg * (pis ** 2).mean()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (q_taken).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", targets.sum().item() / mask_elems, t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.log_stats_t = t_env

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = []
        inputs.append(batch["state"][:, ts].unsqueeze(2).expand(-1, -1, self.n_agents, -1))
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t - 1, t)])
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
                                      dim=1)
                # last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))