"""Placeholder config for PSMAF-Net downstream semantic segmentation.

Route: UperNet / MMSegmentation, matching the original UNIV downstream design.
"""

model = dict(
    type="EncoderDecoder",
    backbone=dict(type="PSMAF_UNIV_Backbone_PLACEHOLDER"),
    decode_head=dict(type="UPerHead"),
)

data = dict(train=dict(type="PLACEHOLDER"), val=dict(type="PLACEHOLDER"), test=dict(type="PLACEHOLDER"))
