# code adapted from https://github.com/wendelinboehmer/dcg

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .cnn_agent import CNNAgent


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.algo_name = args.name
        self.use_rnn = args.use_rnn

        # Use CNN to encode image observations
        self.is_image = False
        if isinstance(input_shape, tuple):  # image input
            self.cnn = CNNAgent(input_shape, args)  # TODO: image support for 'rnn_feature_agent' and 'rnn_ns_agent'
            input_shape = self.cnn.features_dim + input_shape[1]
            self.is_image = True

        assert not (self.is_image is True and self.algo_name == 'emc'), \
            "EMC does not support image obs for the time being!"
        assert not (self.use_rnn is False and self.algo_name == 'emc'), \
            "EMC is implemented only to use RNN for the time being!"

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.use_rnn is True:
            if self.algo_name != 'emc':
                self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
            else:
                self.rnn = nn.GRU(
                    input_size=args.hidden_dim,
                    num_layers=1,
                    hidden_size=args.hidden_dim,
                    batch_first=True
                )
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        if self.is_image is True:
            inputs[0] = self.cnn(inputs[0])
            if len(inputs[1] > 0):
                inputs = th.concat(inputs, dim=1)
            else:
                inputs = inputs[0]

        if self.algo_name == 'emc':
            bs = inputs.shape[0]
            epi_len = inputs.shape[1]
            num_feat = inputs.shape[2]
            inputs = inputs.reshape(bs * epi_len, num_feat)

        x = F.relu(self.fc1(inputs))

        if self.algo_name != 'emc':
            h_in = hidden_state.reshape(-1, self.args.hidden_dim)
            if self.use_rnn:
                h = self.rnn(x, h_in)
            else:
                h = F.relu(self.rnn(x))
            q = self.fc2(h)
        else:
            x = x.reshape(bs, epi_len, self.args.hidden_dim)
            h_in = hidden_state.reshape(1, bs, self.args.hidden_dim).contiguous()
            x, h = self.rnn(x, h_in)
            x = x.reshape(bs * epi_len, self.args.hidden_dim)
            q = self.fc2(x)
            q = q.reshape(bs, epi_len, self.args.n_actions)

        return q, h

