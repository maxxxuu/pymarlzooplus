import torch.nn as nn
import torch.nn.functional as F
from pymarlzooplus.modules.agents.rnn_agent import RNNAgent
from pymarlzooplus.modules.layer.PESymetry import PESymetryMean, RPESymetryMean
import torch as th


class RNNCentPEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNCentPEAgent, self).__init__()
        self.args = args
        self.algo_name = args.name
        self.use_rnn = args.use_rnn
        self.n_agents = args.n_agents
        self.input_shape = input_shape

        self.is_image = False
        if isinstance(input_shape, tuple):  # image input
            self.is_image = True

        # RNN embedding
        self.fc1 = PESymetryMean(input_shape, args.hidden_dim)
        if self.use_rnn is True:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = PESymetryMean(args.hidden_dim, args.hidden_dim)

        #
        self.fc2 = PESymetryMean(args.hidden_dim, args.n_actions)
        # Create the non-shared agents
        # self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on the same device as the model
        return self.fc1.diagonal.weight.new(self.n_agents, self.args.hidden_dim).zero_()
        # return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []

        # input [bs, n_agents, input_size]
        if self.is_image is False and inputs.size(0) == self.n_agents:
            inputs = inputs.unsqueeze(0)
        else:
            print("batch input")
            pass
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.view(-1, self.n_agents, self.args.hidden_dim)
        if self.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)

        return q, h




        # if (
        #         (self.is_image is False and inputs.size(0) == self.n_agents) or
        #         (self.is_image is True and inputs[0].size(0) == self.n_agents)
        # ):  # Single sample
        #     for i in range(self.n_agents):
        #         if self.is_image is True:  # Image observation
        #             raise NotImplemented("RNN Cent PE Agent is not adapted to image input for now")
        #             # TODO: Adapt cent PE agent for image input
        #             agent_inputs = [inputs[0][i].unsqueeze(0), []]
        #             if len(inputs[1]) > 0:
        #                 agent_inputs[1] = inputs[1][i].unsqueeze(0)
        #         # else:  # Vector observation
        #         #
        #         #     agent_inputs = inputs[i].unsqueeze(0)
        #         q, h = self.agents[i](agent_inputs, hidden_state[:, i])
        #         hiddens.append(h)
        #         qs.append(q)
        #     return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        #
        # else:  # Multiple samples
        #     if self.is_image is True:  # Image observation
        #         raise NotImplemented("RNN Cent PE Agent is not adapted to image input for now")
        #         # TODO: Adapt cent PE agent for image input
        #         inputs[0] = inputs[0].view(-1, self.n_agents, *self.input_shape[0])
        #         if len(inputs[1]) > 0:
        #             inputs[1] = inputs[1].view(-1, self.n_agents, self.input_shape[1])
        #     else:  # Vector observation
        #         inputs = inputs.view(-1, self.n_agents, self.input_shape)
        #     for i in range(self.n_agents):
        #         if self.is_image is True:  # Image observation
        #             agent_inputs = [inputs[0][:, i], []]
        #             if len(inputs[1]) > 0:
        #                 agent_inputs[1] = inputs[1][:, i]
        #         else:  # Vector observation
        #             agent_inputs = inputs[:, i]
        #         q, h = self.agents[i](agent_inputs, hidden_state[:, i])
        #         hiddens.append(h.unsqueeze(1))
        #         qs.append(q.unsqueeze(1))
        #     return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)

    # def cuda(self, device="cuda:0"):
    #     # TODO: Fix this in case of image inputs and different devices. Do the same for RNN agent
    #     for a in self.agents:
    #         a.cuda(device=device)
