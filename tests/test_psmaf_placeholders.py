from core.psmaf.multiscale_adaptation import MultiScaleAdaptation, MultiscaleAdaptation
from core.psmaf.psmaf_fusion import PSMAFFusion
from core.psmaf.pseudo_semantic_guidance import PseudoSemanticGuidance


def test_pseudo_semantic_guidance_returns_modal_features_unchanged():
    rgb_feat = {"stage1": object()}
    ir_feat = {"stage1": object()}

    assert PseudoSemanticGuidance()(rgb_feat, ir_feat) == (rgb_feat, ir_feat)


def test_multiscale_adaptation_returns_features_unchanged():
    features = (object(), object())

    assert MultiScaleAdaptation()(features) is features
    assert MultiscaleAdaptation()(features) is features


def test_psmaf_fusion_composes_placeholders_without_changing_io():
    rgb_feat = {"stage1": object()}
    ir_feat = {"stage1": object()}

    fusion = PSMAFFusion()

    assert isinstance(fusion.pseudo_semantic_guidance, PseudoSemanticGuidance)
    assert isinstance(fusion.multiscale_adaptation, MultiScaleAdaptation)
    assert fusion(rgb_feat, ir_feat, pseudo_semantic_prior=None) == (rgb_feat, ir_feat)
