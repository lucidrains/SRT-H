import torch

import pytest
param = pytest.mark.parametrize

@param('pass_custom_style', (False, True))
@param('xm_candidates', (1, 2))
def test_act(
    pass_custom_style,
    xm_candidates
):
    from SRT_H.SRT_H import ACT

    act = ACT(
        dim = 512,
        dim_joint_state = 17,
        action_chunk_len = 16,
        flow_policy = True,
        xm_candidates = xm_candidates
    )

    states = torch.randn(3, 512, 512)
    joint_state = torch.randn(3, 17)

    actions = torch.randn(3, 16, 20)
    style_vector = torch.randn(3, 512) if pass_custom_style else None

    loss = act(
        state_tokens = states,
        joint_state = joint_state,
        actions = actions,
        style_vector = style_vector
    )

    loss.backward()

    # after a lot of data and training ...

    sampled_actions = act(state_tokens = states, joint_state = joint_state, style_vector = style_vector) # (3, 16, 20)

    assert sampled_actions.shape == (3, 16, 20)



@param('tactile', (False, True))
@param('efficient_net', (False, True))
@param('film', (False, True))
@param('action_norm_stats', (False, True))
@param('moss', (False, True))
def test_act_with_image_model(
    tactile,
    efficient_net,
    film,
    action_norm_stats,
    moss
):

    from SRT_H.SRT_H import ACT, DistilBert

    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    v = Extractor(v, return_embeddings_only = True)

    stats = torch.randn((2, 20)) if action_norm_stats else None

    act = ACT(
        image_model = v if not efficient_net else None,
        image_model_dim_emb = 1024,
        dim = 512,
        dim_joint_state = 17,
        action_chunk_len = 16,
        dim_tactile_input = 37,
        action_norm_stats = stats,
        lang_condition_model = DistilBert() if film else None,
        video_moss_kwargs = dict() if moss else None
    )

    states = torch.randn(3, 512, 512)
    joint_state = torch.randn(3, 17)

    tactile_input = torch.randn(3, 16, 37) if tactile else None

    actions = torch.randn(3, 16, 20)

    video = torch.randn(3, 3, 2, 224, 224)

    feedback = [
        "that looks ok, please proceed",
        "you forgot to clip the cystic artery",
        "stop, that is the common bile duct, not the cystic duct"
    ]

    loss = act(
        video = video,
        joint_state = joint_state,
        tactile_input = tactile_input,
        feedback = feedback if film else None,
        actions = actions
    )

    loss.backward()

    # after a lot of data and training ...

    sampled_actions = act(state_tokens = states, joint_state = joint_state) # (3, 16, 20)

def test_high_level():
    from SRT_H.SRT_H import HighLevelPolicy

    high_level_policy = HighLevelPolicy()

    video = torch.randn(3, 3, 2, 224, 224)

    dim = high_level_policy.dim

    task_embeds = torch.randn(3, 17, dim)
    task_labels = torch.randint(0, 17, (3,))

    is_corrective_labels = torch.randint(0, 2, (3,))

    correct_motion_embeds = torch.randn(3, 31, dim)
    correct_motion_labels = torch.randint(0, 31, (3,))

    loss, breakdown = high_level_policy(
        video,
        task_embeds = task_embeds,
        task_labels = task_labels,
        is_corrective_labels = is_corrective_labels,
        correct_motion_embeds = correct_motion_embeds,
        correct_motion_labels = correct_motion_labels
    )

    assert loss.numel() == 1

def test_act_freq_aware_fm_flow_policy():
    from SRT_H.SRT_H import ACT

    act = ACT(
        dim = 512,
        dim_joint_state = 17,
        action_chunk_len = 16,
        flow_policy = True,
        use_freq_aware_fm = True
    )

    states = torch.randn(3, 512, 512)
    joint_state = torch.randn(3, 17)

    actions = torch.randn(3, 16, 20)

    loss = act(
        state_tokens = states,
        joint_state = joint_state,
        actions = actions
    )

    loss.backward()

    sampled_actions = act(state_tokens = states, joint_state = joint_state) # (3, 16, 20)

    assert sampled_actions.shape == (3, 16, 20)

def test_freq_aware_fm_loss_reduction():
    from SRT_H.SRT_H import freq_aware_fm_loss_fn

    loss_fn = freq_aware_fm_loss_fn(action_chunk_len = 16)

    pred = torch.randn(3, 6, 20)
    target = torch.randn(3, 6, 20)

    loss = loss_fn(pred, target)
    assert loss.ndim == 0

    loss_none = loss_fn(pred, target, reduction = 'none')
    assert loss_none.shape == pred.shape

    j = torch.arange(pred.shape[-2])
    weights = 1. + (j * torch.pi / 16) ** 2
    weight_sum = weights.sum() * pred.shape[0] * pred.shape[-1]

    assert torch.allclose(loss_none.sum() / weight_sum, loss, atol = 1e-6)

    with pytest.raises(AssertionError):
        loss_fn(pred, target, reduction = 'sum')

def test_freq_aware_fm_transform_identity():
    from SRT_H.SRT_H import freq_aware_fm_forward_transform, freq_aware_fm_inverse_transform

    actions = torch.randn(2, 16, 20)

    coeffs = freq_aware_fm_forward_transform(actions, 16 - 1) # # max m is chunk_len - 1
    recon_actions = freq_aware_fm_inverse_transform(coeffs, 16)

    assert torch.allclose(actions, recon_actions, atol = 1e-5), 'FAFM forward and inverse transforms must be identity when freq_coeff_cutoff = n - 1'

def test_flow_action_to_noise_latents_roundtrip():
    from SRT_H.SRT_H import FlowActionDecoder

    class ConstantVelocity(torch.nn.Module):
        def forward(self, x, **kwargs):
            return torch.ones_like(x)

    wrapper = FlowActionDecoder(
        decoder = ConstantVelocity(),
        dim = 32,
        dim_action = 20,
        action_chunk_len = 16
    )

    encoded = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype = torch.bool)

    actions = torch.randn(2, 16, 20)

    # action at t=1 -> noise at t=0 -> action at t=1, exact for a constant velocity field

    latents = wrapper.action_to_noise_latents(encoded, actions, mask, steps = 4)
    assert latents.shape == actions.shape

    reconstructed = wrapper.sample(encoded, mask, noise = latents, steps = 4)
    assert torch.allclose(reconstructed, actions, atol = 1e-5)

def test_flow_steering_network():
    from SRT_H.SRT_H import FlowActionDecoder, SteeringNetwork
    from x_transformers import Encoder

    decoder = Encoder(dim = 32, depth = 1, heads = 4, cross_attend = True)

    steering_net = SteeringNetwork(
        dim = 32,
        dim_action = 20,
        action_chunk_len = 16,
        dim_hidden = 64
    )

    wrapper = FlowActionDecoder(
        decoder = decoder,
        dim = 32,
        dim_action = 20,
        action_chunk_len = 16,
        steering_net = steering_net
    )

    encoded = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype = torch.bool)

    noise = steering_net(encoded, mask = mask)
    assert noise.shape == (2, 16, 20)

    steered = wrapper.sample(encoded, mask, steps = 2)
    explicit = wrapper.sample(encoded, mask, noise = noise, steps = 2)

    assert torch.allclose(steered, explicit, atol = 1e-6)

    actions = torch.randn(2, 16, 20)

    loss = wrapper.steering_loss(encoded, actions, mask, steps = 2)
    loss.backward()

    assert all(param.grad is not None for param in steering_net.parameters())

def test_act_with_flow_steering():
    from SRT_H.SRT_H import ACT, SteeringNetwork

    act = ACT(
        dim = 64,
        dim_joint_state = 17,
        action_chunk_len = 16,
        flow_policy = True,
        decoder_wrapper_kwargs = dict(
            steering_net = SteeringNetwork(
                dim = 64,
                dim_action = 20,
                action_chunk_len = 16,
                dim_hidden = 64
            )
        )
    ).eval()

    states = torch.randn(3, 8, 64)
    joint_state = torch.randn(3, 17)

    sampled_actions = act(state_tokens = states, joint_state = joint_state)
    assert sampled_actions.shape == (3, 16, 20)

    # steering noise is deterministic - unlike the random noise of the base policy

    sampled_actions_again = act(state_tokens = states, joint_state = joint_state)
    assert torch.allclose(sampled_actions, sampled_actions_again, atol = 1e-6)
