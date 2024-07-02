import os
import numpy as np
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_registry
from components.standarize_stream import RunningMeanStd

class HAPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.current_episode = 0
        self.n_agents = args.n_agents
        self.learners = [HAPPO(mac, scheme, logger, args) for _ in range(self.n_agents)]

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        masks = batch["filled"][:, :-1].float()
        masks[:, 1:] = masks[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)       

        masks = masks.repeat(1, 1, self.n_agents)

        factor = th.ones((batch.batch_size, masks.shape[1], 1))
        returns = th.zeros((batch.batch_size, masks.shape[1], 1))
        value_preds = th.zeros_like(returns)

        for agent_id in th.randperm(self.n_agents):
            self.learners[agent_id].update_current_agent_id(agent_id)
            self.learners[agent_id].mac.init_hidden(batch.batch_size)
            hidden_states = self.learners[agent_id].mac.get_hidden_states()
            old_log_pi_taken, _, _, _ = self.learners[agent_id].evaluate_actions(batch, actions, masks[:, :, agent_id], hidden_states)
            returns, value_preds = self.learners[agent_id].train(batch, t_env, returns, value_preds, masks[:, :, agent_id], old_log_pi_taken, factor)
            log_pi_taken, _, _, _ = self.learners[agent_id].evaluate_actions(batch, actions, masks[:, :, agent_id], hidden_states)

            factor = factor * th.prod(th.exp(log_pi_taken - old_log_pi_taken), dim=-1,keepdim=True)

    def cuda(self):
        [learner.cuda() for learner in self.learners]

    def save_models(self, path):
        [learner.save_models(path) for learner in self.learners]

    def load_models(self, path):
        [learner.load_models(path) for learner in self.learners]


class HAPPO:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
        self.huber_delta = args.huber_delta
        self.current_agent_id = None
        self.hidden_states = None
        
    def update_current_agent_id(self, agent_id):
        self.current_agent_id = agent_id.item()

    def evaluate_actions(self, batch, actions, mask, hidden_states):
        mac_out = []
        
        self.mac.update_hidden_states(hidden_states)
        self.hidden_states = hidden_states
        
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            h = self.mac.get_hidden_states().detach()
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
    
        pi = mac_out
        pi = pi[:, :, self.current_agent_id, :].unsqueeze(2)
        
        # Calculate policy grad with mask

        pi[mask == 0] = 1.0

        pi_taken = th.gather(pi, dim=3, index=actions[:, :, self.current_agent_id, :].unsqueeze(2)).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)
        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

        return log_pi_taken, entropy, pi, h

    
    def train(self, batch, t_env, returns, value_preds, masks, old_log_pi_taken, factor):
        advantages = returns - value_preds

        if self.args.standardise_advantages:
            advantages_copy = advantages.clone().detach()
            advantages_copy[masks == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        for _ in range(self.args.epochs):

            bs, episode_length = advantages.shape[0:2]
            batch_size = bs * episode_length

            mini_batch_size = batch_size // self.args.num_mini_batch
            
            rand = th.randperm(batch.batch_size).numpy()
            sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(self.args.num_mini_batch)]

            for indices in sampler:
                rewards = batch["reward"][indices][:, :-1]
                actions = batch["actions"][indices][:, :]
                terminated = batch["terminated"][indices][:, :-1].float()
                mask = batch["filled"][indices][:, :-1].float()
                adv_targ = advantages[indices].detach()
                factor = factor[indices]

                mini_batch = batch[indices]

                mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
                actions = actions[:, :-1]
                if self.args.standardise_rewards:
                    self.rew_ms.update(rewards)
                    rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

                critic_mask = mask.clone()

                log_pi_taken, entropy, pi, self.hidden_states = self.evaluate_actions(mini_batch, actions, mask, self.hidden_states)

                ratios = th.prod(th.exp(log_pi_taken - old_log_pi_taken.detach()), dim=-1,keepdim=True)
                surr1 = ratios * adv_targ
                surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * adv_targ

                pg_loss = -((factor.detach() * th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

                # Optimise agent
                self.agent_optimiser.zero_grad()
                pg_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
                self.agent_optimiser.step()

                critic_train_stats, v, ret = self.train_critic_sequential(self.critic, batch, rewards, critic_mask)

            self.critic_training_steps += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (adv_targ * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

        return ret, v
    
    def train_critic_sequential(self, critic, batch, rewards, mask):
            # Optimise critic
            with th.no_grad():
                vals = critic(batch)
                vals = vals.squeeze(3)

            if self.args.standardise_returns:
                vals = vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

            returns = self.nstep_returns(rewards, mask, vals, self.args.q_nstep)
            if self.args.standardise_returns:
                self.ret_ms.update(returns)
                returns = (returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

            running_log = {
                "critic_loss": [],
                "critic_grad_norm": [],
                "td_error_abs": [],
                "returns_mean": [],
                "q_taken_mean": [],
            }

            v = critic(batch)[:, :-1].squeeze(3)
            td_error = (returns.detach() - v)
            masked_td_error = td_error * mask

            a = (abs(masked_td_error) <= self.huber_delta).float()
            b = (abs(masked_td_error) > self.huber_delta).float()
            
            loss = (a*masked_td_error**2/2 + b*self.huber_delta*(abs(masked_td_error)-self.huber_delta/2)).sum() / mask.sum()

            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm.item())
            mask_elems = mask.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
            running_log["returns_mean"].append((returns * mask).sum().item() / mask_elems)

            return running_log, v, returns

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        if not os.path.exists(path+'/agent_'+str(self.current_agent_id)):
            os.mkdir(path+'/agent_'+str(self.current_agent_id))
        self.mac.save_models(path+'/agent_'+str(self.current_agent_id))
        th.save(self.critic.state_dict(), "{}/agent_{}/critic.th".format(path, str(self.current_agent_id)))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_{}/agent_opt.th".format(path, str(self.current_agent_id)))
        th.save(self.critic_optimiser.state_dict(), "{}/agent_{}/critic_opt.th".format(path, str(self.current_agent_id)))

    def load_models(self, path):
        self.mac.load_models(path+'/agent_'+str(self.current_agent_id))
        self.critic.load_state_dict(th.load("{}/agent_{}/critic.th".format(path, str(self.current_agent_id)), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_{}/agent_opt.th".format(path, str(self.current_agent_id)), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/agent_{}/critic_opt.th".format(path, str(self.current_agent_id)), map_location=lambda storage, loc: storage))
