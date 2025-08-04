from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, cat, stack, Tensor, is_tensor
from torch.nn import Module, ModuleList, Parameter, Identity, Linear, Sequential

from x_transformers import Encoder, Attention, AttentionPool

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

from bidirectional_cross_attention import BidirectionalCrossAttentionTransformer as BiCrossAttnTransformer

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# pretrained model related
# they successfully apply

# 1. efficient net for low level vision
# 2. swin t for high level vision
# 3. distilbert for clinician language feedback

class AcceptVideoSwin(Module):
    def __init__(
        self,
        hub_url = 'SharanSMenon/swin-transformer-hub',
        model_name = 'swin_tiny_patch4_window7_224',
        dim_model = 768,
        max_time_seq_len = 8 # say 8 frames
    ):
        super().__init__()
        swin = torch.hub.load(hub_url, model_name, pretrained = True)
        swin.avgpool = Identity()
        swin.head = Rearrange('b (d n) -> b n d', d = dim_model)

        self.model = AcceptVideoWrapper(
            swin,
            add_time_pos_emb = True,
            time_seq_len = max_time_seq_len,
            dim_emb = dim_model
        )

    def forward(
        self,
        video
    ):
        embeds = self.model(video)
        return rearrange(embeds, 'b t n d -> b (t n) d')

class EfficientNetImageModel(Module):
    def __init__(
        self,
        hub_url = 'NVIDIA/DeepLearningExamples:torchhub',
        model_name = 'nvidia_efficientnet_b0',
        utils_path = 'nvidia_convnets_processing_utils',
        dim = 1280
    ):
        super().__init__()
        self.dim = dim

        net = torch.hub.load(hub_url, model_name, pretrained = True)
        utils = torch.hub.load(hub_url, utils_path)

        net.classifier = Rearrange('b d h w -> b (h w) d') # replace the classifier layer in efficient net
        self.net = net

    def forward(self, images):
        return self.net(images)

class DistilBert(Module):
    def __init__(
        self,
        hf_path = "distilbert/distilbert-base-uncased",
        dim = 768
    ):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        self.dim = dim
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
        self.model = AutoModelForMaskedLM.from_pretrained(hf_path)

    def forward(
        self,
        texts: list[str]
    ):
        inputs = self.tokenizer(texts, padding = True, truncation = True, return_tensors = 'pt')

        with torch.no_grad():
            self.model.eval()
            out = self.model(**inputs, output_hidden_states = True)

        return out.hidden_states[-1][:, 0]

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
        dim_style_vector = None,
        dim_lang_condition = None,
        lang_condition_model: Module | None = None,
        heads = 8,
        vae_encoder_depth = 3,
        encoder_depth = 6,
        decoder_depth = 6,
        vae_encoder_kwargs: dict = dict(),
        vae_encoder_attn_pool_depth = 2,
        encoder_kwargs: dict = dict(),
        decoder: dict = dict(),
        image_model: Module | None = None,
        image_model_dim_emb = None,
        dim_tactile_input = None,
        tactile_self_attn_depth = 2,
        tactile_image_fusion_cross_attn_depth = 2, # ViTacFormer
        max_num_image_frames = 32,
        vae_kl_loss_weight = 1.,
        action_loss_fn = nn.L1Loss()
    ):
        super().__init__()

        self.dim = dim

        # style vector dimension related

        dim_style_vector = default(dim_style_vector, dim)
        need_style_proj = dim_style_vector != dim

        self.dim_style_vector = dim_style_vector

        # projections

        self.joint_to_token = nn.Linear(dim_joint_state, dim)
        self.action_to_vae_tokens = nn.Linear(dim_action, dim)

        # for the cvae and style vector

        self.vae_encoder = Encoder(
            dim = dim,
            depth = vae_encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
            use_rmsnorm = True
        )

        self.attn_pooler = AttentionPool(dim = dim, depth = vae_encoder_attn_pool_depth, heads = heads, dim_head = dim_head)

        self.to_style_vector_mean_log_variance = Sequential(
            Linear(dim, dim_style_vector * 2, bias = False),
            Rearrange('... (d mean_log_var) -> mean_log_var ... d', mean_log_var = 2)
        )

        self.style_vector_to_token = nn.Linear(dim_style_vector, dim) if need_style_proj else nn.Identity()

        # detr like

        self.encoder = Encoder(
            dim = dim,
            depth = vae_encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
            use_rmsnorm = True
        )

        self.action_queries = Parameter(torch.randn(action_chunk_len, dim) * 1e-2)

        self.decoder = Encoder(
            dim = dim,
            depth = vae_encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
            cross_attend = True,
            use_rmsnorm = True,
            rotary_pos_emb = True
        )

        self.decoder_embed_to_actions = nn.Linear(dim, dim_action)

        # image model

        image_model_dim_emb = default(image_model_dim_emb, dim)
        need_image_to_state_proj = image_model_dim_emb != dim

        # they used efficient net in the paper, but allow for others

        if not exists(image_model):
            image_model = EfficientNetImageModel()
            image_model_dim_emb = image_model.dim

        # set the image model and the projection to image tokens (state tokens)

        self.image_model = image_model
        self.to_state_tokens = nn.Linear(image_model_dim_emb, dim) if exists(image_model) and need_image_to_state_proj else nn.Identity()

        if exists(image_model):
            self.accept_video_wrapper = AcceptVideoWrapper(image_model, add_time_pos_emb = True, time_seq_len = max_num_image_frames, dim_emb = image_model_dim_emb)

        # tactile

        self.to_tactile_tokens = nn.Linear(dim_tactile_input, dim) if exists(dim_tactile_input) else None

        self.tactile_self_attn = Encoder(
            dim = dim,
            depth = tactile_self_attn_depth,
            heads = heads,
            attn_dim_head = dim_head,
            pre_norm_has_final_norm = False
        )

        self.tactile_fuse = BiCrossAttnTransformer(
            dim = dim,
            context_dim = dim,
            heads = heads,
            depth = tactile_image_fusion_cross_attn_depth
        )

        # take care of clinician feedback which is conditioning the state tokens with FiLM

        self.lang_condition_model = lang_condition_model

        self.to_film_scale_offset = None

        if exists(dim_lang_condition) or exists(lang_condition_model):

            if exists(lang_condition_model):
                dim_lang_condition = default(dim_lang_condition, getattr(lang_condition_model, 'dim', None))

            assert exists(dim_lang_condition), f'`dim_lang_condition` not set'

            self.to_film_scale_offset = nn.Linear(dim_lang_condition, dim * 2, bias = False)
            nn.init.zeros_(self.to_film_scale_offset.weight)

        # loss related

        self.action_loss_fn = action_loss_fn
        self.vae_kl_loss_weight = vae_kl_loss_weight

    def forward(
        self,
        *,
        joint_state,                 # (d)
        video = None,                # (b c t h w)
        state_tokens = None,         # (b n d)
        tactile_input = None,        # (b nt dt)
        tactile_tokens = None,       # (b nt d)
        actions = None,              # (b na da)
        style_vector = None,         # (d) | (b d)
        lang_condition = None,       # (b d)
        feedback: list[str] | None = None,
        return_loss_breakdown = False
    ):

        # take care of video -> image tokens

        assert exists(state_tokens) or exists(video), '`video` or its encoded `state_tokens` must be passed in'
        assert not (exists(video) and not exists(self.image_model)), '`video` cannot be passed in if `image_model` is not set'

        if exists(video):
            assert video.ndim == 5

            images_embeds = self.accept_video_wrapper(video, eval_with_no_grad = True)
            state_tokens = self.to_state_tokens(images_embeds)

            state_tokens = rearrange(state_tokens, 'b t n d -> b (t n) d')

        # if tactile tokens are presented, fuse it with cross attention, as proposed by ViTacFormer - force feedback is becoming a thing

        if exists(tactile_input):
            assert not exists(tactile_tokens) and exists(self.to_tactile_tokens)

            tactile_tokens = self.to_tactile_tokens(tactile_input)

        if exists(tactile_tokens):
            tactile_tokens = self.tactile_self_attn(tactile_tokens)

            state_tokens, tactile_tokens = self.tactile_fuse(state_tokens, tactile_tokens)

        # maybe condition state tokens

        assert not (exists(lang_condition) and exists(feedback))

        if exists(feedback):
            assert exists(self.lang_condition_model), f'`lang_condition_model` module must be passed in for direct language conditioning on efficientnet output'

            lang_condition = self.lang_condition_model(feedback)

        if exists(lang_condition):
            assert exists(self.to_film_scale_offset), f'`dim_lang_condition` must be set if doing further conditioning (clinician feedback in this paper)'

            scale, offset = self.to_film_scale_offset(lang_condition).chunk(2, dim = -1)

            scale, offset = tuple(rearrange(t, 'b d -> b 1 d') for t in (scale, offset))

            state_tokens = state_tokens * (scale + 1.) + offset

        # variables

        batch, device = state_tokens.shape[0], state_tokens.device

        is_training = exists(actions)
        is_sampling = not is_training

        assert not (is_training and exists(style_vector)), 'style vector z cannot be set during training'

        # joint token

        joint_tokens = self.joint_to_token(joint_state)
        joint_tokens = rearrange(joint_tokens, 'b d -> b 1 d')

        # take care of the needed style token during training

        if is_training:
            action_vae_tokens = self.action_to_vae_tokens(actions)

            vae_input = cat((action_vae_tokens, joint_tokens), dim = 1)

            vae_encoder_embed = self.vae_encoder(vae_input)

            # cross attention pool

            pooled_vae_embed = self.attn_pooler(vae_encoder_embed)

            style_mean, style_log_variance = self.to_style_vector_mean_log_variance(pooled_vae_embed)

            # reparam

            style_std = (0.5 * style_log_variance).exp()

            noise = torch.randn_like(style_mean)

            style_vector = style_mean + style_std * noise

        elif exists(style_vector) and style_vector.ndim == 1:

            style_vector = repeat(style_vector, 'd -> b 1 d', b = batch)

        else:
            # or just zeros during inference, as in the paper

            style_vector = torch.zeros((batch, 1, self.dim_style_vector), device = device)

        style_token = self.style_vector_to_token(style_vector)

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
