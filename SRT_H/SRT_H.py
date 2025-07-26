from __future__ import annotations

import torch
from torch import nn, cat, stack, Tensor, is_tensor
from torch.nn import Module, ModuleList, Parameter, Linear, Sequential

from x_transformers import Encoder, Attention

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# ACT - Action Chunking Transformer - Zhou et al.

class ACT(Module):
    def __init__(
        self,
        dim,
        *,
        dim_joint_state,
        action_chunk_len,
        dim_head = 64,
        heads = 8,
        vae_encoder_depth = 6,
        encoder_depth = 6,
        decoder_depth = 6,
        vae_encoder_kwargs: dict = dict(),
        encoder_kwargs: dict = dict(),
        decoder: dict = dict(),
        vae_kl_loss_weight = 1.,
        action_loss_fn = nn.L1Loss()
    ):
        super().__init__()

        # projections

        self.joint_to_token = nn.Linear(dim_joint_state, dim)

        # for the cvae and style vector

        self.vae_encoder = Encoder(
            dim = dim,
            depth = vae_encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
        )

        self.attn_pool_query = Parameter(torch.randn(dim)) # there is evidence attention pooling is better than CLS / global average pooling

        self.to_mean_log_variance = Sequential(
            Linear(dim, dim * 2, bias = False),
            Rearrange('... (d mean_log_var) -> mean_log_var ... d', mean_log_var = 2)
        )

        # detr like

        self.encoder = Encoder(
            dim = dim,
            depth = vae_encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
        )

        self.action_queries = Parameter(torch.randn(dim) * 1e-2)
        self.action_pos_emb = Parameter(torch.randn(action_chunk_len, dim))

        self.decoder = Encoder(
            dim = dim,
            depth = vae_encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
            cross_attend = True
        )

        # loss related

        self.action_loss_fn = action_loss_fn
        self.vae_kl_loss_weight = vae_kl_loss_weight

    def forward(
        self,
        state,
        joint_state,
        actions = None
    ):
        is_sampling = not exists(actions)

        raise NotImplementedError

# classes

class SRT(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        state
    ):
        raise NotImplementedError
