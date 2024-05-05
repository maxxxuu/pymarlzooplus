import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class EOINet(nn.Module):
    def __init__(self, obs_len, n_agent):
        super(EOINet, self).__init__()
        self.fc1 = nn.Linear(obs_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_agent)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.softmax(self.fc3(y), dim=1)
        return y


class IVF(nn.Module):
    def __init__(self, obs_len, n_action):
        super(IVF, self).__init__()
        self.fc1 = nn.Linear(obs_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_action)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class EOITrainer(object):
    def __init__(self, eoi_net, ivf, ivf_tar, n_agent, n_feature, ivf_gamma, ivf_tau, ivf_lr, eoi_lr):
        super(EOITrainer, self).__init__()

        self.gamma = ivf_gamma
        self.tau = ivf_tau
        self.n_agent = n_agent
        self.n_feature = n_feature
        self.eoi_net = eoi_net
        self.ivf = ivf
        self.ivf_tar = ivf_tar
        self.optimizer_eoi = optim.Adam(self.eoi_net.parameters(), lr=eoi_lr)
        self.optimizer_ivf = optim.Adam(self.ivf.parameters(), lr=ivf_lr)

    def train(self, O, O_Next, A, D):
        device = next(self.eoi_net.parameters()).device  # TODO: get device in more proper way

        O = torch.Tensor(O).to(device)
        O_Next = torch.Tensor(O_Next).to(device)
        A = torch.Tensor(A).to(device).long()
        D = torch.Tensor(D).to(device)

        X = O_Next[:, 0:self.n_feature]
        Y = O_Next[:, self.n_feature:self.n_feature + self.n_agent]
        p = self.eoi_net(X)
        loss_1 = -(Y * (torch.log(p + 1e-8))).mean() - 0.1 * (p * (torch.log(p + 1e-8))).mean()
        self.optimizer_eoi.zero_grad()
        loss_1.backward()
        self.optimizer_eoi.step()

        I = O[:, self.n_feature: self.n_feature + self.n_agent].argmax(axis=1, keepdim=True).long()
        r = self.eoi_net(O[:, 0: self.n_feature]).gather(dim=-1, index=I)

        q_intrinsic = self.ivf(O)
        tar_q_intrinsic = q_intrinsic.clone().detach()
        next_q_intrinsic = self.ivf_tar(O_Next).max(axis=1, keepdim=True)[0]
        next_q_intrinsic = r * 10 + self.gamma * (1 - D) * next_q_intrinsic
        tar_q_intrinsic.scatter_(dim=-1, index=A, src=next_q_intrinsic)
        loss_2 = (q_intrinsic - tar_q_intrinsic).pow(2).mean()
        self.optimizer_ivf.zero_grad()
        loss_2.backward()
        self.optimizer_ivf.step()

        with torch.no_grad():
            for p, p_targ in zip(self.ivf.parameters(), self.ivf_tar.parameters()):
                p_targ.data.mul_(self.tau)
                p_targ.data.add_(self.tau * p.data)


class EOIBatchTrainer(object):
    def __init__(self, eoi_trainer, n_agent, n_feature, max_step, batch_size, eoi_batch_size):
        super(EOIBatchTrainer, self).__init__()

        self.batch_size = batch_size
        self.eoi_batch_size = eoi_batch_size
        self.n_agent = n_agent
        self.o_t = np.zeros((batch_size * n_agent * (max_step + 1), n_feature + n_agent))
        self.next_o_t = np.zeros((batch_size * n_agent * (max_step + 1), n_feature + n_agent))
        self.a_t = np.zeros((batch_size * n_agent * (max_step + 1), 1), dtype=np.int32)
        self.d_t = np.zeros((batch_size * n_agent * (max_step + 1), 1))
        self.eoi_trainer = eoi_trainer

    def train_batch(self, episode_sample):
        episode_obs = np.array(episode_sample["obs"])
        episode_actions = np.array(episode_sample["actions"])
        episode_terminated = np.array(episode_sample["terminated"])
        ind = 0

        # Add agent id
        for k in range(self.batch_size):
            for j in range(episode_obs.shape[1] - 2):
                for i in range(self.n_agent):
                    agent_id = np.zeros(self.n_agent)
                    agent_id[i] = 1
                    self.o_t[ind] = np.hstack((episode_obs[k][j][i], agent_id))
                    self.next_o_t[ind] = np.hstack((episode_obs[k][j + 1][i], agent_id))
                    self.a_t[ind] = episode_actions[k][j][i]
                    self.d_t[ind] = episode_terminated[k][j]
                    ind += 1
                if self.d_t[ind - 1] == 1:
                    break

        # Train in batches
        for k in range(int((ind - 1) / self.eoi_batch_size)):
            self.eoi_trainer.train(self.o_t[k * self.eoi_batch_size: (k + 1) * self.eoi_batch_size],
                                   self.next_o_t[k * self.eoi_batch_size: (k + 1) * self.eoi_batch_size],
                                   self.a_t[k * self.eoi_batch_size: (k + 1) * self.eoi_batch_size],
                                   self.d_t[k * self.eoi_batch_size: (k + 1) * self.eoi_batch_size])


class Explorer(object):

    def __init__(self, scheme, groups, args, episode_limit):

        self.episode_ratio = args.episode_ratio
        self.explore_ratio = args.explore_ratio
        self.n_agents = groups["agents"]
        self.device = args.device

        self.eoi_net = EOINet(
            scheme["obs"]["vshape"],
            self.n_agents
        ).to(self.device)

        self.ivf = IVF(
            scheme["obs"]["vshape"] + self.n_agents,
            scheme["avail_actions"]["vshape"][0]
        ).to(self.device)

        self.ivf_tar = IVF(
            scheme["obs"]["vshape"] + self.n_agents,
            scheme["avail_actions"]["vshape"][0]
        ).to(self.device)

        eoi_trainer = EOITrainer(
            self.eoi_net,
            self.ivf,
            self.ivf_tar,
            self.n_agents,
            scheme["obs"]["vshape"],
            args.ivf_gamma,
            args.ivf_tau,
            args.ivf_lr,
            args.eoi_lr
        )

        self.trainer = EOIBatchTrainer(
            eoi_trainer,
            self.n_agents,
            scheme["obs"]["vshape"],
            episode_limit,
            args.batch_size,
            args.eoi_batch_size
        )

        self.ivf_flag = False

    def train(self, episode_sample):
        self.trainer.train_batch(episode_sample)

    def build_obs(self, obs):
        for i in range(self.n_agents):
            index = np.zeros(self.n_agents)
            index[i] = 1
            obs[i] = np.hstack((obs[i], index))

        # List to numpy
        if isinstance(obs, list):
            obs = np.array(obs)
        assert isinstance(obs, np.ndarray), f"obs type: {type(obs)}"

        # Numpy to tensor
        obs = torch.from_numpy(obs).to(torch.float32).to(self.device)

        return obs

    def select_actions(self, actions, t, test_mode, data):

        if t == 0:
            self.ivf_flag = (np.random.rand() < self.episode_ratio)

        if (test_mode is False) & (self.ivf_flag is True):
            if np.random.rand() < self.explore_ratio:

                obs = data["obs"][0]

                # tuple to list
                if isinstance(obs, tuple):
                    obs = list(obs)

                obs = self.build_obs(obs)
                q_p = self.ivf(obs).detach().cpu().numpy()

                # Change random actions
                j = np.random.randint(self.n_agents)
                actions[0][j] = np.argmax(q_p[j] - 9e15 * (1 - np.array(data["avail_actions"][0][j])))

        return actions

