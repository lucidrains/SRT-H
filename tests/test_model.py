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

    loss = act(
        state_tokens = states,
        joint_state = joint_state,
        actions = actions
    )

    loss.backward()

    # after a lot of data and training ...

    style_vector = torch.ones(512) if pass_custom_style else None

    sampled_actions = act(state_tokens = states, joint_state = joint_state, style_vector = style_vector) # (3, 16, 20)

    assert sampled_actions.shape == (3, 16, 20)

def act_with_image_model():
    from SRT_H.SRT_H import ACT

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

    act = ACT(
        image_model = v,
        image_model_dim_emb = 1024,
        dim = 512,
        dim_joint_state = 17,
        action_chunk_len = 16
    )

    states = torch.randn(3, 512, 512)
    joint_state = torch.randn(3, 17)

    actions = torch.randn(3, 16, 20)

    video = torch.randn(3, 3, 2, 224, 224)

    loss = act(
        video = video,
        joint_state = joint_state,
        actions = actions
    )

    loss.backward()

    # after a lot of data and training ...

    sampled_actions = act(state_tokens = states, joint_state = joint_state) # (3, 16, 20)
