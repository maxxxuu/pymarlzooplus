import torch
from torch import nn
import torch.nn.utils.parametrize as P
import torch.nn.functional as F


class PESymetryMean(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pos: bool = False) -> None:
        """

        pos : if pos=True, all the weights will be positive
        """
        super(PESymetryMean, self).__init__()
        self.individual = nn.Linear(in_dim, out_dim)
        self.pooling = nn.Linear(in_dim, out_dim, bias=False)
        if pos:
            P.register_parametrization(self.individual, "weight", SoftplusParameterization())
            P.register_parametrization(self.pooling, "weight", SoftplusParameterization())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.pooling(x_mean)
        x = self.individual(x)
        x = x + x_mean
        return x

class PESymetryMeanTanh(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pos: bool = False) -> None:
        """

        pos : if pos=True, all the weights will be positive
        """
        super(PESymetryMeanTanh, self).__init__()
        self.individual = nn.Linear(in_dim, out_dim)
        self.pooling = nn.Linear(in_dim, out_dim, bias=False)
        if pos:
            P.register_parametrization(self.individual, "weight", SoftplusParameterization())
            P.register_parametrization(self.pooling, "weight", SoftplusParameterization())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.pooling(x_mean)
        x = self.individual(x)
        x = x + F.tanh(x_mean)
        return x


class PESymetryMeanDivided(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pos: bool = False) -> None:
        """

        pos : if pos=True, all the weights will be positive
        """
        super(PESymetryMeanDivided, self).__init__()
        if in_dim * 2 < out_dim:
            self.individual = nn.Linear(in_dim, out_dim - in_dim)
            self.pooling = nn.Linear(in_dim, in_dim, bias=False)
        elif in_dim < out_dim:
            self.individual = nn.Linear(in_dim, in_dim)
            self.pooling = nn.Linear(in_dim, out_dim - in_dim, bias=False)
        else:
            pooling_dim = out_dim // 2
            ind_dim = out_dim - pooling_dim
            self.individual = nn.Linear(in_dim, ind_dim)
            self.pooling = nn.Linear(in_dim, pooling_dim, bias=False)

        if pos:
            P.register_parametrization(self.individual, "weight", SoftplusParameterization())
            P.register_parametrization(self.pooling, "weight", SoftplusParameterization())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.pooling(x_mean)
        x_mean = x_mean.expand(-1, x.size(-2), -1) if x.dim() == 3 else x_mean.expand(x.size(-2), -1)
        x = self.individual(x)
        x = torch.cat([x, x_mean], dim=-1)
        return x


class PESymetryMax(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMax, self).__init__()
        self.individual = nn.Linear(in_dim, out_dim)
        self.pooling = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = x.max(0, keepdim=True)
        x_max = self.pooling(x_max)
        x = self.individual(x)
        x = x + x_max
        return x


class PESymetryMeanMax(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMeanMax, self).__init__()
        self.individual = nn.Linear(in_dim, out_dim)
        self.max = nn.Linear(in_dim, out_dim, bias=False)
        self.mean = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = x.max(0, keepdim=True)
        x_max = self.max(x_max)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.mean(x_mean)
        x = self.individual(x)
        x = x + x_max + x_mean
        return x


class PESymetryMeanAct(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMeanAct, self).__init__()
        self.individual = nn.Linear(in_dim, out_dim)
        self.pooling = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.pooling(x_mean)
        x_mean = nn.ELU()(x_mean)
        x = self.individual(x)
        # x = nn.ELU()(x)
        x = x + x_mean
        return x


class RPESymetryMean(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(RPESymetryMean, self).__init__()
        self.individual = nn.GRUCell(in_dim, out_dim)
        self.pooling = nn.GRUCell(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        output = []
        for i, j in zip(x.view(-1, x.shape[-2], x.shape[-1]), h.view(-1, h.shape[-2], h.shape[-1])):
            # print(f"Shape i: {i.shape}, shape j: {j.shape}")
            x_mean = i.mean(-2, keepdim=True)
            h_mean = j.mean(-2, keepdim=True)
            h_mean = self.pooling(x_mean, h_mean)
            h_out = self.individual(i, j)
            h_out = h_out + h_mean
            output.append(h_out)
        output = torch.stack(output, dim=0)
        if torch.any(torch.isnan(output)):
            pass
        return output.view(h.shape)


class SoftplusParameterization(nn.Module):
    # Make weights positive
    def forward(self, X):
        return nn.functional.softplus(X)

    # def right_inverse(self, A):
    #     return A + torch.log(-torch.expm1(-A))
