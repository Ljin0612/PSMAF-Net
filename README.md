# PSMAF-Net

PSMAF-Net is a formal research repository for pseudo-semantic guided multispectral adaptive fusion. This repository currently contains the initial project structure for organizing core PSMAF modules, detection experiments, segmentation experiments, documentation, and utility scripts.

## Repository status

This is the initial structure of the formal research repository. Implementation, dataset adapters, experiment runners, and reproducibility notes will be expanded as the project develops.

## Layout

- `core/`: Core PSMAF method components.
- `detection/`: Detection configurations, dataset notes, adapters, and scripts.
- `segmentation/`: Segmentation configurations, dataset notes, adapters, and scripts.
- `third_party/UNIV/`: Vendored official UNIV source used as the upstream baseline. See `docs/univ_source.md` for provenance.
- `docs/`: Project plans, design notes, and baseline result templates.
- `tools/`: General repository utilities.

## Artifact policy

Do not commit training results, model weights, checkpoints, exported models, logs, or generated experiment artifacts. Use external storage or ignored local directories for large or reproducible artifacts.
