from torch import Tensor
import torch
import torch.nn as nn

import math

from typing import Type


class MLPBlock(nn.Module):
    """
    Normal MLP Block

    Args:
        embedding_dim: The number of embedding dimensions,
        mlp_dim: output embedding dimensions,
        act: activation function

    Returns:
        (mlp_dim, embedding_dim)

    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] == nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Attention(nn.Module):
    """
    An attention layer that can be used to downscale the size of the embedding
    after projection to queries, keys, and values.

    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim,
        self.internal_dim = int(embedding_dim // downsample_rate)
        print(num_heads)
        self.num_heads = num_heads,
        print(self.num_heads)
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim"

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def seperate_to_heads(self, x: torch.Tensor, num_heads) -> torch.Tensor:
        b, n, c = x.shape
        print(num_heads)
        x = x.reshape(b, n, num_heads, int(c // num_heads))
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        # B x N_tokens x C_per_head
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Seperate into heads
        q = self.seperate_to_heads(q, self.num_heads)
        k = self.seperate_to_heads(k, self.num_heads)
        v = self.seperate_to_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        att = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        att = att / math.sqrt(c_per_head)  # Normalization
        att_score = torch.softmax(att, dim=-1)

        # Final ouput
        out = att_score @ v
        out = self.combine_heads(out)
        out = self.out_proj(out)

        return out


a = Attention(embedding_dim=64, num_heads=4, downsample_rate=0.5)
print(a)
q = torch.rand(10, 100, 64, dtype=torch.float)
a.forward(q, q, q)

class ResidualBlock(nn.Module):
    """Adding a residual connection to a output of a NN"""
    def __init__(self, nn):
        super().__init__()
        self.fn = nn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x