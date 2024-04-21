import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CNNAgent(nn.Module):
    """
    NatureCNN network, or in other words,
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    Code adopted from:
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
    """
    def __init__(self, input_shape, args):
        super(CNNAgent, self).__init__()
        self.args = args

        self.features_dim = args.cnn_features_dim

        n_input_channels = input_shape[0][0]  # 3 if RGB, 1 if gray scale
        assert n_input_channels in [1, 3], f"Invalid number of input channels: {n_input_channels}"
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.ones((1, *input_shape[0])).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, self.features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
