########### - Imports - ###########
#################################################################################
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Parameter

from typing import Type
########### - End - ###########
#################################################################################


########### - MLP Layers - ###########
#################################################################################
class MLPBlock(nn.Module):
    """
    Normal 2-layer MLP Block

    Args:
        embedding_dim: The number of embedding dimensions,
        mlp_dim: output embedding dimensions,
        act: activation function
        drop: dropout probability

    Returns:
        (mlp_dim, embedding_dim)

    """

    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] == nn.GELU,
            drop: float = 0.0,
            bias: bool = True
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim, bias=bias)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim, bias=bias)
        self.act = act()
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.drop(x)

        return x


class MLPBlockNorm(nn.Module):
    """
    Normalized 2-layer MLP Block

    Args:
        embedding_dim: The number of embedding dimensions,
        mlp_dim: output embedding dimensions,
        drop: dropout probability

    Returns:
        (mlp_dim, embedding_dim)

    """

    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            drop: float = 0.0,
            bias: bool = True
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim, bias=bias),
            nn.Dropout(drop),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim, bias=bias),
            nn.Dropout(drop),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.layers(x)


########### - End - ###########
#################################################################################


########### - Residual Blocks - ###########
#################################################################################
class ResidualBlockMLP(nn.Module):
    """Adding a residual connection to a output of a NN"""

    def __init__(self, nn):
        super().__init__()
        self.fn = nn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResidualBlockGNN(nn.Module):
    """Adding a residual connection to a output of a GNN layer"""

    def __init__(self, nn):
        super().__init__()
        self.layer = nn

    def forward(self, x, *args, **kwargs):
        # x: node hidden representation
        return self.layer(x, *args, **kwargs) + x

########### - End - ###########
#################################################################################



