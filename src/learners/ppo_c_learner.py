import copy

import torch as th
from torch.optim import Adam
import torch.nn.functional as F

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry


class PPOLearner_C:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        # Add action range
        self.action_min = th.tensor(args.action_min)
        self.action_max = th.tensor(args.action_max)
        print(f"Action min/max = {self.action_min}/{self.action_max}")
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    #############################################################
    ## ----- Retrieves probabilities (old_pi) of actions ----- ##
    #############################################################
    def get_discrete_log_probs(self, batch, mask, actions, mac):
      mac = [] # The outputs of the old multi-agent controller
      mac.init_hidden(batch.batch_size) # Only for recurrent
      for t in range(batch.max_seq_length - 1):
          agent_outs = mac.forward(batch, t=t)
          mac_out.append(agent_outs)
      mac_out = th.stack(mac_out, dim=1)  # Concat over time
      p_i = mac_out # [batch, num_actions]
      p_i[mask == 0] = 1.0
      p_i_taken = th.gather(p_i, dim=3, index=actions).squeeze(3)
      log_pi_taken = th.log(p_i_taken + 1e-10)
      return log_pi_taken

    def construct_cont_dist(self, o):
      #print(o.shape) # 100,max_timesteps,1,10
      na = o.shape[-1]//2
      means, vars = o[:,:,:,:na], o[:,:,:,na:]
      means, stds = F.tanh(means), th.sqrt(F.softplus(vars))
      # Force means into the target range. Same for std. devs, but half the range (as a generally good heuristic)
      means = th.clamp(means, min=self.action_min, max=self.action_max)
      action_range = self.action_max - self.action_min
      mins = (th.zeros_like(action_range)+1e-3).unsqueeze(0).unsqueeze(0).unsqueeze(0)
      maxes = (action_range/2).unsqueeze(0).unsqueeze(0).unsqueeze(0)
      stds = th.clamp(stds, min=mins, max=maxes)
      #stds = stds * 0 + 0.5 # Try just hard-fixing it to 0.5 (OKAY, THIS LINE WORKS! Note that this is just for optimization, std is not fixed in runtime)
      # End of test
      action_dists = th.distributions.Normal(means, stds)
      return action_dists

    def get_continuous_log_probs(self, batch, mask, actions, mac):
      #print("Get Continuous Log Probs called!")
      #print(actions.shape) # 100, 100, 1, agent_output size
      #print(mask.shape) # [100,100,1], looks like all ones.
      mac_out = [] # The outputs of the old multi-agent controller
      self.mac.init_hidden(batch.batch_size) # Only for recurrent
      for t in range(batch.max_seq_length - 1): # [batch, num_actions]
          agent_outs = self.mac.forward(batch, t=t) # 100, 1, 5 in discrete, 100, 1, 10 in continuous
          mac_out.append(agent_outs)
      mac_out = th.stack(mac_out, dim=1)  # Concat over time
      dists = self.construct_cont_dist(mac_out) # [batch, num_actions] -> MAC should output probs.
      # Note that 'taken' is no longer used; continuous outputs always taken.
      log_pi = dists.log_prob(actions).sum(axis=1)
      entropy = 0
      #print(dists.entropy().shape) #100,100,1,5
      if (mac == self.mac):
        entropy = dists.entropy().squeeze()
      return log_pi, entropy

    #############################################################
    ## --------------------- Runs Training ------------------- ##
    #############################################################
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]

        if self.args.standardise_rewards:
          self.rew_ms.update(rewards)
          rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        # This needs to be replaced with continuous probabilities
        #old_log_pi = self.get_discrete_log_probs(batch, mask, actions, self.old_mac)
        old_log_pi, _ = self.get_continuous_log_probs(batch, mask, actions, self.old_mac)

        # In discrete mode, gets the indices of the old MAC's actions and uses them to grab the corresponding log probabilities.
        for k in range(self.args.epochs):
            # Training the critic shouldn't affect action probabilities, so I've moved this out of the middle of the probability calc.
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()
            ### Calculate log probabilities
            #log_pi_taken = self.get_discrete_log_probs(batch, mask, actions, self.mac)
            log_pi, entropy = self.get_continuous_log_probs(batch, mask, actions, self.mac)

            ratios = th.exp(log_pi - old_log_pi.detach())
            # surr1 can have + and - infinity as values
            surr1 = ratios * advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            # torch.Size([100, 1, 1, 5])
            #print((th.min(surr1, surr2)).shape) # torch.Size([100, 100, 5])
            #print(entropy.shape) # torch.Size([100, 100, 5])
            # Calculate entropy of normal (rather than categorical) distribution
            #entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

            # Calculate total loss
            pg_loss = (
                -(
                    (th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask
                ).sum()
                / mask.sum()
            )

            if (th.isnan(pg_loss) or th.isinf(pg_loss)):
              print(pg_loss)
              print("NEW LOG PI")
              print(log_pi.min())
              print(log_pi.max())
              print("RATIOS")
              print(ratios.min())
              print(ratios.max())
              print("SURR1")
              print(surr1.min())
              print(surr1.max())
              print("SURR2")
              print(surr2.min())
              print(surr2.max())
              print("ENTROPY")
              print(entropy.min())
              print(entropy.max())
              print("!!!!!!!!!!!!!!!!!!!!!")
              sdfafsdadsfasadfasdfasdfasdf

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            '''self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )''' # Not readily applicable to continuous spaces
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.q_nstep
        )
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args.gamma ** (step) * values[:, t] * mask[:, t]
                    )
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += (
                        self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += (
                        self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
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
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )