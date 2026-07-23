# UNIV source provenance

## Official UNIV repository

https://github.com/fangyuanmao/UNIV.git

## Imported location

`third_party/UNIV/`

## Official commit hash

`dd38740a02e2c75308697f2d15361cbacb7ee7c5`

## Import policy

- `third_party/UNIV` stores the upstream UNIV source.
- `third_party/UNIV/datasets/` is an upstream Python source package and must not be ignored in source audits; distinguish it from real dataset artifacts such as repository-root `/datasets/` or `/data/`.
- Do not mix PSMAF-Net research modules into `third_party/UNIV`.
- PSMAF-Net modifications should be implemented under `core/psmaf`, `detection`, `segmentation`, or `tools`.
- If upstream UNIV code must be patched for compatibility, document the reason and patch in `docs/univ_audit_report.md`.

## Dependency note

- Official UNIV README records `python=3.6`.
- Official UNIV `requirements.txt` records `torch==2.4.1`, `torchvision==0.19.1`, `numpy==1.23.5`, `timm==1.0.12`, `mmengine==0.10.5`, `peft==0.13.0`, and related packages including `PyYAML==6.0.2` and `packaging==24.2`.
- The top-level PSMAF-Net environment may use a modern Python version, but any deviation from upstream must be documented.
