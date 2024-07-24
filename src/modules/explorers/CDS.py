# Code adapted from: https://github.com/lich14/CDS
import copy

import torch as th
import torch.optim as optim
from torch import nn as nn
from torch.nn import functional as F

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

