from __future__ import annotations
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, Tensor, tensor, is_tensor
from torch.nn import Module, ModuleList, Parameter, Identity, Linear, Sequential

from x_transformers import Encoder, Attention, AttentionPool

import einx
from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange

from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

from bidirectional_cross_attention import BidirectionalCrossAttentionTransformer as BiCrossAttnTransformer

import numpy as np
from autofaiss import build_index

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def batcher(arr, batch):
    for i in range(0, len(arr), batch):
        yield arr[i:(i + batch)]

# function uses autofaiss to build the commands embedding with ann index

class CommandsIndexer(Module):
    def __init__(
        self,
        commands: list[str],
        model = None,
        embed_batch_size = 32,
        embed_on_device = None
    ):
        super().__init__()

        if not exists(model):
            model = DistilBert()

            if exists(embed_on_device):
                model = model.to(embed_on_device)

        self.commands = commands

        command_embeds = cat([model(commands_batch).cpu() for commands_batch in batcher(commands, embed_batch_size)])

        indexer, index_info = build_index(command_embeds.numpy(), save_on_disk = False)

        self.indexer = indexer
        self.index_info = index_info

        self.register_buffer('command_embeds', command_embeds)

    def forward(
        self,
        embed, # (b d)
        return_strings = False
    ):
        device = self.command_embeds.device

        query = embed.cpu().numpy()
        _, index = self.indexer.search(query, 1)

        index = torch.from_numpy(index)
        index = rearrange(index, 'b 1 -> b').to(device)

        closest_embeds = self.command_embeds[index]

        if not return_strings:
            return closest_embeds

        commands = [self.commands[i] for i in index]
        return closest_embeds, commands

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
        action_loss_fn = nn.L1Loss(),
        dropout_video_frame_prob = 0.07 # 7% chance of dropping out a frame during training, regularization mentioned in paper
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

        self.dropout_video_frame_prob = dropout_video_frame_prob

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

        state_mask = None

        if exists(video):
            device = video.device

            assert video.ndim == 5

            images_embeds = self.accept_video_wrapper(video, eval_with_no_grad = True)
            state_tokens = self.to_state_tokens(images_embeds)

            state_mask = torch.ones(state_tokens.shape[:3], dtype = torch.bool, device = device)

            if self.training:
                dropout_frame = torch.rand(state_tokens.shape[:2], device = device) < self.dropout_video_frame_prob
                state_mask = einx.logical_and('b t n, b t', state_mask, ~dropout_frame)

            state_tokens = rearrange(state_tokens, 'b t n d -> b (t n) d')

            state_mask = rearrange(state_mask, 'b t n -> b (t n)')

        # if tactile tokens are presented, fuse it with cross attention, as proposed by ViTacFormer - force feedback is becoming a thing

        if exists(tactile_input):
            assert not exists(tactile_tokens) and exists(self.to_tactile_tokens)

            tactile_tokens = self.to_tactile_tokens(tactile_input)

        if exists(tactile_tokens):
            tactile_tokens = self.tactile_self_attn(tactile_tokens)

            state_tokens, tactile_tokens = self.tactile_fuse(state_tokens, tactile_tokens, mask = state_mask)

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

        batch, device = state_tokens.shape[0], state_tokens.device

        # variables

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

        mask = None

        if exists(state_mask):
            mask = F.pad(state_mask, (1, joint_tokens.shape[1]), value = True)

        encoder_input = cat((style_token, state_tokens, joint_tokens), dim = 1)

        encoded = self.encoder(encoder_input, mask = mask)

        decoder_input = repeat(self.action_queries, 'na d -> b na d', b = batch)

        decoded = self.decoder(decoder_input, context = encoded, context_mask = mask)

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

# high level transformer
# their high-level policy is a SWiN that takes in images, passes through attention layers to yield a language embedding

class HighLevelPolicy(Module):
    def __init__(
        self,
        dim_language_embed = 768, # dimension if distilbert
        transformer: Module | dict = dict(
            dim = 768,
            attn_dim_head = 64,
            heads = 8,
            depth = 4
        ),
        attn_pool_heads = 8,
        attn_pool_dim_head = 64,
        task_loss_weight = 0.4,
        is_corrective_loss_weight = 0.3,
        corrective_motion_loss_weight = 0.3
    ):
        super().__init__()

        self.accept_video_wrapper = AcceptVideoSwin()

        if isinstance(transformer, dict):
            transformer = Encoder(**transformer)

        self.transformer = transformer

        self.dim = transformer.dim

        self.attn_pooler = AttentionPool(
            dim_language_embed,
            num_pooled_tokens = 3,
            dim_context = transformer.dim,
            heads = attn_pool_heads,
            dim_head = attn_pool_dim_head
        )

        self.to_corrective_pred = nn.Sequential(
            nn.Linear(transformer.dim, 1),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid()
        )

        # loss related

        self.task_loss_weight = task_loss_weight
        self.is_corrective_loss_weight = is_corrective_loss_weight
        self.corrective_motion_loss_weight = corrective_motion_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        video,
        task_embeds = None, # (b total_commands d)
        task_labels = None,
        is_corrective_labels = None, # (b)
        correct_motion_embeds = None, # (b total_corr_motions d) - they only had 18
        correct_motion_labels = None,
        temperature = 1.
    ):
        batch, device = video.shape[0], video.device

        tokens = self.accept_video_wrapper(video)

        attended = self.transformer(tokens)

        embeds = self.attn_pooler(attended).unbind(dim = 1)

        if not (exists(task_embeds) and exists(task_labels)):
            return embeds

        pred_task_embed, is_corrective_embed, pred_correct_motion_embeds = embeds

        if exists(pred_task_embed):
            pred_task_logits = einsum(l2norm(pred_task_embed), l2norm(task_embeds), 'b d, b n d -> b n') / temperature

        if not exists(task_labels):
            return pred_task_logits

        task_loss = F.cross_entropy(pred_task_logits, task_labels)

        # interesting technique where they scale the task loss by the l1 loss of the labels - explanation in High-level policy section (near eq 1) - 2.5% improvement

        batch_arange = torch.arange(batch, device = device)[..., None]
        target_task_embed = task_embeds[batch_arange, task_labels]

        l1_dist = F.l1_loss(pred_task_embed, target_task_embed)

        task_loss = task_loss * l1_dist

        # is corrective

        is_corrective_loss = self.zero

        if exists(is_corrective_labels):
            is_corrective_pred = self.to_corrective_pred(is_corrective_embed)

            is_corrective_loss = F.binary_cross_entropy(
                is_corrective_pred,
                is_corrective_labels.float()
            )

        # corrective motion labels

        correct_motion_loss = self.zero

        if exists(correct_motion_labels):
            correct_motion_logits = einsum(l2norm(pred_correct_motion_embeds), l2norm(correct_motion_embeds), 'b d, b n d -> b n') / temperature

            correct_motion_loss = F.cross_entropy(
                correct_motion_logits,
                correct_motion_labels
            )

        # return total loss and loss breakdown

        total_loss = (
            task_loss * self.task_loss_weight +
            is_corrective_loss * self.is_corrective_loss_weight + 
            correct_motion_loss * self.corrective_motion_loss_weight
        )

        loss_breakdown = (task_loss, is_corrective_loss, correct_motion_loss)

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
