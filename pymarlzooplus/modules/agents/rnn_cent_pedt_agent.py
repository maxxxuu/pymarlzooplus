import torch.nn as nn
import torch.nn.functional as F
from pymarlzooplus.modules.agents.rnn_agent import RNNAgent
from pymarlzooplus.modules.layer.PESymetry import PESymetryMeanDivided, PESymetryMeanTanh
import torch as th


class RNNCentPEDTAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNCentPEDTAgent, self).__init__()
        self.args = args
        self.algo_name = args.name
        self.use_rnn = args.use_rnn
        self.n_agents = args.n_agents
        self.input_shape = input_shape

        self.is_image = False
        if isinstance(input_shape, tuple):  # image input
            self.is_image = True

        # RNN embedding
        self.fc1 = PESymetryMeanDivided(input_shape, args.hidden_dim)
        if self.use_rnn is True:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = PESymetryMeanTanh(args.hidden_dim, args.hidden_dim)

        #
        self.fc2 = PESymetryMeanTanh(args.hidden_dim, args.n_actions)
        # Create the non-shared agents
        # self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on the same device as the model
        return self.fc1.individual.weight.new(self.n_agents, self.args.hidden_dim).zero_()
        # return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []

        # input [bs, n_agents, input_size]
        if self.is_image is False and inputs.size(0) == self.n_agents:
            inputs = inputs.unsqueeze(0)
        # else:
        #     print("batch input")
        #     pass
        x = F.elu(self.fc1(inputs))

        if self.use_rnn:
            # GRUCell only accept 2D inputs
            h_in = hidden_state.contiguous().view(-1, self.args.hidden_dim)
            x = x.view(-1, self.args.hidden_dim)
            h = self.rnn(x, h_in)
            h = h.view(-1, self.n_agents, self.args.hidden_dim)
        else:
            h = F.elu(self.rnn(x))
        q = self.fc2(h)

        return q, h