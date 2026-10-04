from __future__ import annotations
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, Tensor, tensor, pi
from torch.nn import Module, Parameter, Identity, Linear, Sequential

from x_transformers import Encoder, AttentionPool

import einx
from einops import rearrange, repeat, einsum, reduce, pack
from einops.layers.torch import Rearrange

from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

from bidirectional_cross_attention import BidirectionalCrossAttentionTransformer as BiCrossAttnTransformer

from rectified_flow_pytorch.nano_flow import NanoFlow

from torch_einops_utils import masked_mean
from torch_einops_utils.shape import shape, size

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

# frequency-aware flow matching helpers
# https://arxiv.org/abs/2606.20135

def freq_aware_fm_forward_transform(actions, freq_coeff_cutoff): # M in the paper
    n, device = size(actions, '... [n] d'), actions.device
    j = torch.arange(freq_coeff_cutoff + 1, device = device)
    t = torch.arange(n, device = device)

    freqs = torch.outer(j, 2 * t + 1) * pi / (2 * n)
    basis = torch.cos(freqs)

    return einsum(actions, basis, '... n d, m n -> ... m d') * (2 / n)

def freq_aware_fm_inverse_transform(coeffs, n):
    m_plus_1, device = size(coeffs, '... [m] d'), coeffs.device

    j = torch.arange(m_plus_1, device = device)
    t = torch.arange(n, device = device)

    freqs = torch.outer(2 * t + 1, j) * pi / (2 * n)
    basis = torch.cos(freqs)

    coeffs = coeffs.clone()
    coeffs[..., 0, :] *= 0.5

    return einsum(coeffs, basis, '... m d, n m -> ... n d')

def freq_aware_fm_loss_fn(action_chunk_len, weight_vel = 1.):

    def loss_fn(pred, target, reduction = 'mean'):
        assert reduction in ('mean', 'none'), f'reduction must be one of `mean` or `none`'

        m_plus_1, device, dtype = size(pred, '... [m] d'), pred.device, pred.dtype
        j = torch.arange(m_plus_1, device = device, dtype = dtype)
        omega = j * pi / action_chunk_len

        # l2 velocity error in time domain simplifies to diagonal omega squared weighting on dct coefficients
        loss_weights = 1. + weight_vel * (omega ** 2)

        mse = F.mse_loss(pred, target, reduction = 'none')

        if reduction == 'none':
            return einx.multiply('... m d, m -> ... m d', mse, loss_weights)

        # weights act as a non boolean mask for masked_mean, whose sum is the denominator

        loss_weights = rearrange(loss_weights, 'm -> 1 m 1')

        return masked_mean(mse, mask = loss_weights)

    return loss_fn

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
        self.patch_size = 32

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

# decoding strategies

# 1. DETR queries to prediction with l1 loss

class DETRActionDecoder(Module):
    def __init__(
        self,
        decoder: Module,
        dim,
        dim_action,
        action_chunk_len,
        action_loss_fn = nn.L1Loss()
    ):
        super().__init__()

        self.action_queries = Parameter(torch.randn(action_chunk_len, dim) * 1e-2)
        self.decoder = decoder
        self.decoder_embed_to_actions = nn.Linear(dim, dim_action)

        self.action_loss_fn = action_loss_fn

    def sample(
        self,
        encoded,
        mask
    ):
        batch = size(encoded, '[b] ...')

        decoder_input = repeat(self.action_queries, 'na d -> b na d', b = batch)

        decoded = self.decoder(decoder_input, context = encoded, context_mask = mask)

        pred_actions = self.decoder_embed_to_actions(decoded)
        return pred_actions

    def forward(
        self,
        encoded,
        actions,
        *,
        mask,
        loss_reduction = 'mean'
    ):
        pred_actions = self.sample(encoded, mask)

        if isinstance(self.action_loss_fn, Module):
            self.action_loss_fn.reduction = loss_reduction
            return self.action_loss_fn(pred_actions, actions)

        return self.action_loss_fn(pred_actions, actions, reduction = loss_reduction)

# 2. Flow matching for decoder (flow / diffusion policy)

class WrappedDecoder(Module):
    def __init__(
        self,
        model: Module,
        dim,
        dim_action,
    ):
        super().__init__()
        self.proj_in = nn.Linear(dim_action, dim)
        self.model = model
        self.proj_out = nn.Linear(dim, dim_action)

    def forward(
        self,
        x,
        *args,
        **kwargs
    ):
        x = self.proj_in(x)

        x = self.model(x, *args, **kwargs)

        x = self.proj_out(x)

        return x

class SteeringNetwork(Module):
    def __init__(
        self,
        dim,
        dim_action,
        action_chunk_len,
        dim_hidden = 256,
        depth = 2
    ):
        super().__init__()
        self.action_chunk_len = action_chunk_len

        layers = []
        dims = (dim, *((dim_hidden,) * depth))

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.extend((Linear(dim_in, dim_out), nn.Mish()))

        layers.append(Linear(dims[-1], action_chunk_len * dim_action))

        self.net = Sequential(*layers)

    def forward(
        self,
        encoded,        # (b n d)
        mask = None
    ):
        pooled = masked_mean(encoded, mask = mask, dim = 1)
        return rearrange(self.net(pooled), 'b (n d) -> b n d', n = self.action_chunk_len)

class FlowActionDecoder(Module):
    def __init__(
        self,
        decoder: Module,
        dim,
        dim_action,
        action_chunk_len,
        loss_fn = F.mse_loss,
        steering_net: Module | None = None
    ):
        super().__init__()

        decoder = WrappedDecoder(decoder, dim = dim, dim_action = dim_action)
        self.flow_wrapper = NanoFlow(decoder, data_shape = (action_chunk_len, dim_action), loss_fn = loss_fn)

        self.steering_net = steering_net

    def sample(
        self,
        encoded,
        mask,
        noise = None,
        steps = 16
    ):
        if not exists(noise) and exists(self.steering_net):
            noise = self.steering_net(encoded, mask = mask)

        batch_size = size(encoded, '[b] ...')
        return self.flow_wrapper.sample(batch_size = batch_size, noise = noise, steps = steps, context = encoded, context_mask = mask)

    def action_to_noise_latents(
        self,
        encoded,
        actions,
        mask,
        steps = 16,
        reverse_fixed_point_steps = 5
    ):
        # reverse flow ode - action at t=1 back to noise at t=0

        assert not self.flow_wrapper.predict_clean, 'reverse flow ode requires velocity prediction (`predict_clean = False`)'

        batch_size = size(encoded, '[b] ...')
        return self.flow_wrapper.sample(batch_size = batch_size, image = actions, reverse = True, steps = steps, reverse_fixed_point_steps = reverse_fixed_point_steps, context = encoded, context_mask = mask)

    def steering_loss(
        self,
        encoded,
        actions,
        mask,
        steps = 16
    ):
        assert exists(self.steering_net), '`steering_net` must be passed in to compute `steering_loss`'

        targets = self.action_to_noise_latents(encoded, actions, mask, steps = steps)

        pred = self.steering_net(encoded, mask = mask)
        return F.mse_loss(pred, targets)

    def forward(
        self,
        encoded,
        actions,
        *,
        mask,
        loss_reduction = 'mean'
    ):
        return self.flow_wrapper(actions, context = encoded, context_mask = mask, loss_reduction = loss_reduction)

# ACT - Action Chunking Transformer - Zhou et al.

Losses = namedtuple('Losses', ('action_recon',))

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
        encoder_depth = 6,
        decoder_depth = 6,
        encoder_kwargs: dict = dict(),
        decoder: dict = dict(),
        decoder_wrapper_kwargs: dict = dict(),
        flow_policy = True,
        xm_candidates = 2,
        action_norm_stats: Tensor | None = None,
        image_model: Module | None = None,
        image_model_dim_emb = None,
        dim_tactile_input = None,
        tactile_self_attn_depth = 2,
        tactile_image_fusion_cross_attn_depth = 2, # ViTacFormer
        max_num_image_frames = 32,
        dropout_video_frame_prob = 0.07, # 7% chance of dropping out a frame during training, regularization mentioned in paper
        video_moss_kwargs: dict | None = None,
        use_freq_aware_fm = False,
        freq_aware_fm_freq_coeff_cutoff = None, # M in the paper
        freq_aware_fm_weight_vel = 1.,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.action_chunk_len = action_chunk_len

        self.use_freq_aware_fm = use_freq_aware_fm
        self.freq_aware_fm_freq_coeff_cutoff = default(freq_aware_fm_freq_coeff_cutoff, max(1, action_chunk_len // 3))

        decoder_action_chunk_len = (self.freq_aware_fm_freq_coeff_cutoff + 1) if use_freq_aware_fm else action_chunk_len

        if use_freq_aware_fm:
            freq_aware_fm_loss = freq_aware_fm_loss_fn(action_chunk_len, weight_vel = freq_aware_fm_weight_vel)

        # style vector dimension related

        dim_style_vector = default(dim_style_vector, dim)
        need_style_proj = dim_style_vector != dim

        self.dim_style_vector = dim_style_vector
        self.style_vector_to_token = nn.Linear(dim_style_vector, dim) if need_style_proj else nn.Identity()

        # explorative modeling candidates

        self.xm_candidates = xm_candidates

        # projections

        self.joint_to_token = nn.Linear(dim_joint_state, dim)

        # detr like

        self.encoder = Encoder(
            dim = dim,
            depth = encoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
            use_rmsnorm = True,
            **encoder_kwargs
        )

        self.decoder = Encoder(
            dim = dim,
            depth = decoder_depth,
            heads = heads,
            attn_dim_head = dim_head,
            cross_attend = True,
            use_rmsnorm = True,
            rotary_pos_emb = True,
            **decoder
        )

        # whether to use detr or flow matching for decoding to surgical bot actions

        if flow_policy:
            self.decoder_wrapper = FlowActionDecoder(
                decoder = self.decoder,
                dim_action = dim_action,
                dim = dim,
                action_chunk_len = decoder_action_chunk_len,
                loss_fn = freq_aware_fm_loss if use_freq_aware_fm else F.mse_loss,
                **decoder_wrapper_kwargs
            )

        else:
            self.decoder_wrapper = DETRActionDecoder(
                decoder = self.decoder,
                dim_action = dim_action,
                dim = dim,
                action_chunk_len = decoder_action_chunk_len,
                action_loss_fn = freq_aware_fm_loss if use_freq_aware_fm else nn.L1Loss(),
                **decoder_wrapper_kwargs
            )

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
            moss_kwargs = dict(dim = image_model_dim_emb, **video_moss_kwargs) if exists(video_moss_kwargs) else None
            self.accept_video_wrapper = AcceptVideoWrapper(image_model, add_time_pos_emb = True, time_seq_len = max_num_image_frames, dim_emb = image_model_dim_emb, moss = moss_kwargs)

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

        # action (inverse) norm related

        assert not exists(action_norm_stats) or action_norm_stats.shape == (2, dim_action), f'action norm stats must have shape (2, num_actions) - 2 for mean and std'

        self.register_buffer('action_norm_stats', action_norm_stats)

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
        loss_reduction = 'mean',
        return_loss_breakdown = False,
        **kwargs
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

            b, t, n = shape(state_tokens, '[b] [t] [n] d')
            state_mask = torch.ones((b, t, n), dtype = torch.bool, device = device)

            if self.training:
                dropout_frame = torch.rand((b, t), device = device) < self.dropout_video_frame_prob
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

        batch, device = size(state_tokens, '[b] ...'), state_tokens.device

        is_training = exists(actions)

        # joint token

        joint_tokens = self.joint_to_token(joint_state)
        joint_tokens = rearrange(joint_tokens, 'b d -> b 1 d')

        # handle style vector and explorative modeling (XM)
        # in Gladstone's explorative modeling: during training without an explicit style vector,
        # draw K candidate style vectors from prior N(0, I) and select the winner (min loss)

        num_candidates = self.xm_candidates if (is_training and not exists(style_vector)) else 1
        has_multiple_candidates = num_candidates > 1

        if not exists(style_vector):
            randn_or_zeros = torch.randn if is_training else torch.zeros
            style_vector = randn_or_zeros((batch * num_candidates, self.dim_style_vector), device = device)
        elif style_vector.ndim == 1:
            style_vector = repeat(style_vector, 'd -> b d', b = batch)

        style_vector, _ = pack([style_vector], 'b * d')

        if has_multiple_candidates:
            repeat_k = lambda t: repeat(t, 'b ... -> (b k) ...', k = num_candidates) if exists(t) else None
            state_tokens, joint_tokens, actions, state_mask = map(repeat_k, (state_tokens, joint_tokens, actions, state_mask))

        style_token = self.style_vector_to_token(style_vector)

        # detr like encoder / decoder

        mask = None

        if exists(state_mask):
            mask = F.pad(state_mask, (1, size(joint_tokens, 'b [n] d')), value = True)

        encoder_input = cat((style_token, state_tokens, joint_tokens), dim = 1)

        encoded = self.encoder(encoder_input, mask = mask)

        # action needs norm or inverse norm

        action_needs_norm = exists(self.action_norm_stats)

        if action_needs_norm:
            action_mean, action_std = self.action_norm_stats

        # if actions not passed in, assume inference and sample actions, whether from DETR or flow matching

        if not is_training:
            sampled_actions = self.decoder_wrapper.sample(encoded, mask)

            if self.use_freq_aware_fm:
                sampled_actions = freq_aware_fm_inverse_transform(sampled_actions, self.action_chunk_len)

            if action_needs_norm:
                sampled_actions = (sampled_actions * action_std) + action_mean

            return sampled_actions

        # take care of action norm

        if action_needs_norm:
            actions = (actions - action_mean) / action_std.clamp(min = 1e-6)

        if self.use_freq_aware_fm:
            actions = freq_aware_fm_forward_transform(actions, self.freq_aware_fm_freq_coeff_cutoff)

        # take care of training loss
        # if XM candidates > 1, evaluate candidate losses and pick winner (min loss)

        loss = self.decoder_wrapper(encoded, actions, mask = mask, loss_reduction = 'none' if has_multiple_candidates else loss_reduction)

        if has_multiple_candidates:
            candidate_losses = reduce(loss, '(b k) ... -> b k', 'mean', b = batch, k = num_candidates)
            loss = candidate_losses.amin(dim = -1)

            if loss_reduction == 'mean':
                loss = loss.mean()

        if not return_loss_breakdown:
            return loss

        return loss, Losses(loss)

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
        batch, device = size(video, '[b] ...'), video.device

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

        # interesting technique where they scale the task loss by the l1 loss of the labels - explanation in High-level policy section (near eq 1) - 2.5% improvement

        task_ce_per_sample = F.cross_entropy(pred_task_logits, task_labels, reduction = 'none')

        batch_arange = torch.arange(batch, device = device)
        target_task_embed = task_embeds[batch_arange, task_labels]

        l1_dist_matrix = F.l1_loss(pred_task_embed, target_task_embed, reduction = 'none')

        l1_dist_per_sample = reduce(l1_dist_matrix, '... d -> ...', 'mean').detach()

        task_loss = masked_mean(task_ce_per_sample, mask = l1_dist_per_sample)

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
