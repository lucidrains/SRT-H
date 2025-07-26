import torch
import pytest

@pytest.mark.parametrize('pass_custom_style', (False, True))
def test_act(
    pass_custom_style
):
    from SRT_H.SRT_H import ACT

    act = ACT(
        dim = 512,
        dim_joint_state = 17,
        action_chunk_len = 16
    )

    states = torch.randn(3, 512, 512)
    joint_state = torch.randn(3, 17)

    actions = torch.randn(3, 16, 20)

    loss = act(states, joint_state, actions)
    loss.backward()

    # after a lot of data and training ...

    style_vector = torch.ones(512) if pass_custom_style else None

    sampled_actions = act(states, joint_state, style_vector = style_vector) # (3, 16, 20)

    assert sampled_actions.shape == (3, 16, 20)
