# Code adapted from: https://github.com/lich14/CDS
import copy

import torch as th
import torch.optim as optim
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def weights_init_(m):
    """Initialize Policy weights"""
    if isinstance(m, nn.Linear):
        th.nn.init.xavier_uniform_(m.weight, gain=1)
        th.nn.init.constant_(m.bias, 0)


class CDSExplorer:
    def __init__(self, args, scheme):
        self.args = args
        self._device = "cuda" if args.use_cuda and th.cuda.is_available() else "cpu"

        self.obs_shape = scheme["obs"]["vshape"]
        input_shape_without_id = args.hidden_dim + (self.obs_shape if args.ifaddobs else 0) + args.n_actions
        hidden_dim = args.predict_net_dim
        num_outputs = self.obs_shape

        input_shape_with_id = args.hidden_dim + (self.obs_shape if args.ifaddobs else 0) + args.n_actions + args.n_agents

        self.eval_predict_without_id = PredictNetwork(input_shape_without_id, hidden_dim, num_outputs)
        self.target_predict_without_id = copy.deepcopy(self.eval_predict_without_id)

        self.eval_predict_with_id = PredictNetworkWithID(input_shape_with_id, hidden_dim, num_outputs, args.n_agents)
        self.target_predict_with_id = copy.deepcopy(self.eval_predict_with_id)

    # update policy for predict network
    def train_predict(self, learner, batch, t_env: int):
        # Get the relevant quantities
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)

        # Calculate estimated Q-Values
        learner.mac.init_hidden(batch.batch_size)
        initial_hidden = learner.mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self._device)
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self._device)

        _, hidden_store, _ = learner.mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach())
        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3)

        obs = batch["obs"][:, :-1]
        obs_next = batch["obs"][:, 1:]

        h_cat = hidden_store[:, :-1]
        add_id = th.eye(learner.args.n_agents).to(obs.device).expand(
            [obs.shape[0], obs.shape[1], learner.args.n_agents, learner.args.n_agents])
        mask_reshape = mask.unsqueeze(-1).expand_as(
            h_cat[..., 0].unsqueeze(-1))

        _obs = obs.reshape(-1, obs.shape[-1]).detach()
        _obs_next = obs_next.reshape(-1, obs_next.shape[-1]).detach()
        _h_cat = h_cat.reshape(-1, h_cat.shape[-1]).detach()
        _add_id = add_id.reshape(-1, add_id.shape[-1]).detach()
        _mask_reshape = mask_reshape.reshape(-1, 1).detach()
        _actions_onehot = actions_onehot.reshape(
            -1, actions_onehot.shape[-1]).detach()

        #if self.args.ifaddobs:
        h_cat_r = th.cat(
            [th.zeros_like(h_cat[:, 0]).unsqueeze(1), h_cat[:, :-1]], dim=1)
        intrinsic_input = th.cat(
            [h_cat_r, obs, actions_onehot], dim=-1)
        _inputs = intrinsic_input.detach(
        ).reshape(-1, intrinsic_input.shape[-1])
        #else:
        # _inputs = th.cat([_h_cat, _actions_onehot], dim=-1)

        loss_withid_list, loss_withoutid_list, loss_predict_id_list = [], [], []
        # update predict network
        for _ in range(learner.args.predict_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(_obs.shape[0])), 256, False):
                loss_withoutid = self.explorer.eval_predict_without_id.update(
                    _inputs[index], _obs_next[index], _mask_reshape[index])
                loss_withid = self.explorer.eval_predict_with_id.update(
                    _inputs[index], _obs_next[index], _add_id[index], _mask_reshape[index])

                if loss_withoutid:
                    loss_withoutid_list.append(loss_withoutid)
                if loss_withid:
                    loss_withid_list.append(loss_withid)

        self.logger.log_stat("predict_loss_noid", np.array(
            loss_withoutid_list).mean(), t_env)
        self.logger.log_stat("predict_loss_withid", np.array(
            loss_withid_list).mean(), t_env)


    def cuda(self):
        if self.args.use_cuda:
            self.eval_predict_with_id.to(self._device)
            self.target_predict_with_id.to(self._device)

            self.eval_predict_without_id.to(self._device)
            self.target_predict_without_id.to(self._device)
        
        
class PredictNetwork(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, lr=3e-4):
        super(PredictNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        h = F.relu(self.linear1(input))
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = th.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None


class PredictNetworkWithID(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, n_agents, lr=3e-4):
        super(PredictNetworkWithID, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + n_agents, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs, add_id):
        inputs = th.cat([inputs, add_id], dim=-1)
        h = F.relu(self.linear1(inputs))

        h = th.cat([h, add_id], dim=-1)
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = th.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, add_id, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable, add_id)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None
