from __future__ import annotations
from collections import namedtuple

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

Losses = namedtuple('Losses', ('action_recon', 'vae_kl_div'))

class ACT(Module):
    def __init__(
        self,
        dim,
        *,
        dim_joint_state,
        action_chunk_len,
        dim_action = 20,
        dim_head = 64,
        heads = 8,
        vae_encoder_depth = 3,
        encoder_depth = 6,
        decoder_depth = 6,
        vae_encoder_kwargs: dict = dict(),
        encoder_kwargs: dict = dict(),
        decoder: dict = dict(),
        vae_kl_loss_weight = 1.,
        action_loss_fn = nn.L1Loss()
    ):
        super().__init__()

        self.dim = dim

        # projections

        self.joint_to_token = nn.Linear(dim_joint_state, dim)
        self.action_to_vae_tokens = nn.Linear(dim_action, dim)

        # for the cvae and style vector

        self.vae_encoder = Encoder(
            dim = dim,
            depth = vae_encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
        )

        self.attn_pooler = Attention(dim = dim, heads = heads, dim_head = dim_head)
        self.attn_pool_query = Parameter(torch.randn(dim)) # there is evidence attention pooling is better than CLS / global average pooling

        self.to_style_vector_mean_log_variance = Sequential(
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

        self.action_queries = Parameter(torch.randn(action_chunk_len, dim) * 1e-2)

        self.decoder = Encoder(
            dim = dim,
            depth = vae_encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
            cross_attend = True
        )

        self.decoder_embed_to_actions = nn.Linear(dim, dim_action)

        # loss related

        self.action_loss_fn = action_loss_fn
        self.vae_kl_loss_weight = vae_kl_loss_weight

    def forward(
        self,
        state_tokens,        # (b n d)
        joint_state,         # (d)
        actions = None,      # (b na da)
        style_token = None,  # (d) | (b d)
        return_loss_breakdown = False
    ):

        # variables

        batch, device = state_tokens.shape[0], state_tokens.device

        is_training = exists(actions)
        is_sampling = not is_training

        assert not (is_training and exists(style_token)), 'style token cannot be set during training'

        # joint token

        joint_tokens = self.joint_to_token(joint_state)
        joint_tokens = rearrange(joint_tokens, 'b d -> b 1 d')

        # take care of the needed style token during training

        if is_training:
            action_vae_tokens = self.action_to_vae_tokens(actions)

            vae_input = cat((action_vae_tokens, joint_tokens), dim = 1)

            vae_encoder_embed = self.vae_encoder(vae_input)

            # cross attention pool

            attn_pool_queries = repeat(self.attn_pool_query, 'd -> b 1 d', b = batch)

            pooled_vae_embed = self.attn_pooler(attn_pool_queries, vae_encoder_embed)

            style_mean, style_log_variance = self.to_style_vector_mean_log_variance(pooled_vae_embed)

            # reparam

            style_std = (0.5 * style_log_variance).exp()

            noise = torch.randn_like(style_mean)

            style_token = style_mean + style_std * noise

        elif exists(style_token) and style_token.ndim == 1:

            style_token = repeat(style_token, 'd -> b 1 d', b = batch)

        else:
            # or just zeros during inference, as in the paper

            style_token = torch.zeros((batch, 1, self.dim), device = device)

        # detr like encoder / decoder

        encoder_input = cat((style_token, state_tokens, joint_tokens), dim = 1)

        encoded = self.encoder(encoder_input)

        decoder_input = repeat(self.action_queries, 'na d -> b na d', b = batch)

        decoded = self.decoder(decoder_input, context = encoded)

        pred_actions = self.decoder_embed_to_actions(decoded)

        if not is_training:
            return pred_actions

        # take care of training loss

        action_recon_loss = self.action_loss_fn(pred_actions, actions)

        vae_kl_loss = (0.5 * (
            style_log_variance.exp()
            + style_mean.square()
            - style_log_variance
            - 1.
        )).sum(dim = -1).mean()

        loss_breakdown = Losses(action_recon_loss, vae_kl_loss)

        total_loss = (
            action_recon_loss +
            vae_kl_loss * self.vae_kl_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, loss_breakdown

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
