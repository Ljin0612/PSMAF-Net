import pytest
torch = pytest.importorskip("torch")

from core.psmaf import MultiScaleAdaptiveFusion, PSMAFFusion, PseudoSemanticGuidance
from detection.models.psmaf_yolo import PSMAFYOLO


def test_pseudo_semantic_guidance_preserves_spatial_size():
    attention, weights = PseudoSemanticGuidance(16)(torch.randn(2, 16, 9, 11), torch.randn(2, 16, 9, 11))
    assert attention.shape == (2, 1, 9, 11)
    assert weights.shape == (2, 2, 1, 1)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 1, 1, 1))


def test_multiscale_adaptive_fusion_shape():
    x = torch.randn(2, 16, 8, 10)
    assert MultiScaleAdaptiveFusion(16)(x, x).shape == x.shape


@pytest.mark.parametrize("mode", ["psmaf", "add", "concat"])
def test_three_level_fusion(mode):
    features = tuple(torch.randn(1, c, s, s) for c, s in zip((16, 32, 64), (16, 8, 4)))
    result = PSMAFFusion((16, 32, 64), fusion_mode=mode)(features, features)
    assert [x.shape for x in result] == [x.shape for x in features]


def test_model_forward_cpu():
    outputs = PSMAFYOLO(channels=(16, 32, 64))(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    assert [output.shape for output in outputs] == [(1, 11, 8, 8), (1, 11, 4, 4), (1, 11, 2, 2)]
