# code adapted from https://github.com/AnujMahajanOxf/MAVEN
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..agents.cnn_agent import CNNAgent


class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = None
        self.is_image = False

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"
        if isinstance(input_shape, np.int64) or isinstance(input_shape, int):  # Vector input
            self.state_dim = input_shape
        elif isinstance(input_shape, tuple) and (len(input_shape[0]) == 4) and (input_shape[0][1] == 3):  # Image input
            self.state_dim = (args.cnn_features_dim * input_shape[0][0]) + input_shape[1]  # multiply with n_agents
        else:
            raise ValueError(f"Invalid 'input_shape': {input_shape}")

        if self.is_image is True:
            self.cnn = CNNAgent([input_shape[0][1:]], args)

        # Set up network layers
        self.fc1 = nn.Linear(self.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        if self.is_image is True:
            channels = inputs[0].shape[3]
            height = inputs[0].shape[4]
            width = inputs[0].shape[5]
            # Reshape the states
            # from [batch size, max steps, n_agents, channels, height, width]
            # to [batch size x max steps x n_agents, channels, height, width]
            inputs[0] = inputs[0].reshape(-1, channels, height, width)
            total_samples = inputs[0].shape[0]
            n_batches = math.ceil(total_samples / bs)

            # state-images are processed in batches due to memory limitations
            input_new = []
            for batch in range(n_batches):
                # from [batch size, channels, height, width]
                # to [batch size, cnn features dim]
                input_new.append(self.cnn(inputs[0][batch * bs:(batch + 1) * bs]))

            # to [batch size x max steps x n_agents, cnn features dim]
            inputs[0] = th.concat(input_new, dim=0)

            n_cnn_feats = inputs[0].shape[-1]
            # to [batch size x max steps , n_agents, cnn features dim]
            inputs[0] = inputs[0].view(bs * max_t, self.n_agents, n_cnn_feats)
            # to [batch size x max steps , n_agents x cnn features dim]
            inputs[0] = inputs[0].view(bs * max_t, self.n_agents * n_cnn_feats)
            # to [batch size x max steps , n_agents, n_agents x cnn features dim]
            inputs[0] = inputs[0].unsqueeze(1).repeat(1, self.n_agents, 1)
            # to [batch size, max steps , n_agents, n_agents x cnn features dim]
            inputs[0] = inputs[0].view(bs, max_t, self.n_agents, self.n_agents * n_cnn_feats)
            # to [batch size, max steps, n_agents, cnn features dim x n_agents + extra features]
            inputs = th.cat(inputs, dim=-1)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        if self.is_image is False:
            inputs = [batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1)]
        else:
            inputs = [batch["state"][:, ts]]

        # individual observations
        assert not (self.args.obs_individual_obs is True and self.is_image is True), \
            "In case of state image, obs_individual_obs is not supported."
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]),
                                       batch["actions_onehot"][:, :-1]],
                                      dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        # Add agents IDs in one-hot format
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        if self.is_image is False:
            inputs = th.cat(inputs, dim=-1)
        else:
            inputs[1] = th.cat(inputs[1:], dim=-1)
            del inputs[2]
            assert len(inputs) == 2, "length of inputs: {}".format(len(inputs))
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        if isinstance(input_shape, int):
            # observations
            if self.args.obs_individual_obs:
                input_shape += scheme["obs"]["vshape"] * self.n_agents
            # last actions
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            input_shape += self.n_agents
        elif isinstance(input_shape, tuple):
            assert self.args.obs_individual_obs is False, \
                "In case of state-image, 'obs_individual_obs' argument is not supported."
            self.is_image = True
            input_shape = [input_shape, 0]
            if self.args.obs_last_action:
                input_shape[1] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            input_shape[1] += self.n_agents
            input_shape = tuple(input_shape)
        return input_shape
