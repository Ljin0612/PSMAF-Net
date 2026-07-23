# UNIV source audit report

## Audit date

2026-07-23

## Official UNIV URL

https://github.com/fangyuanmao/UNIV.git

## Official commit hash

`dd38740a02e2c75308697f2d15361cbacb7ee7c5`

## Official clone status

`git -C /tmp/UNIV-official status --short --branch` reported `## main...origin/main` with no uncommitted files.

## Official top-level files

- `.git/`
- `README.md`
- `SEG/`
- `assert/`
- `configs/`
- `datasets/`
- `loss/`
- `models/`
- `pretrain_mcmae.py`
- `requirements.txt`
- `run.sh`
- `utils/`

## Official README summary

- Environment: create a Conda environment with `python=3.6`, activate it, and install `requirements.txt`.
- Dataset: MVIP is listed as the pretraining dataset.
- Checkpoints: the README links a UNIV pretrained checkpoint.
- Pretraining: full pretraining is launched with `bash run.sh`; the README notes the MCMAE checkpoint source from Alpha-VL/ConvMAE.
- Semantic segmentation: segmentation head training is launched from `SEG/MCMAE_SEG` with `bash run_train.sh`.

## Official requirements summary

The official dependency snapshot includes PyTorch and CUDA-wheel pins (`torch==2.4.1`, `torchvision==0.19.1`, NVIDIA CUDA 12 packages), core scientific packages (`numpy==1.23.5`, `scipy==1.10.1`, `opencv-python==4.10.0.84`, `pillow==10.4.0`), model/framework packages (`timm==1.0.12`, `mmengine==0.10.5`, `peft==0.13.0`, `transformers==4.46.3`, `accelerate==1.0.1`), and utility packages including `PyYAML==6.0.2` and `packaging==24.2`.

## Comparison method

The audit cloned the official repository to `/tmp/UNIV-official` with depth 1, copied both trees to filtered temporary directories, and compared them with `git diff --no-index --name-status`. The comparison ignored `.git/`, `__pycache__/`, `*.pyc`, `data/`, `checkpoints/`, `outputs/`, `runs/`, `results/`, `logs/`, `*.pth`, `*.pt`, and `*.ckpt`. It intentionally did **not** ignore `third_party/UNIV/datasets/`, because that path is an upstream Python source package, not a dataset artifact. Root-level `/datasets/` and `/data/` remain dataset-artifact paths for this repository and must not be conflated with vendored source packages.

## Comparison conclusion

`third_party/UNIV` matches the official UNIV source at commit `dd38740a02e2c75308697f2d15361cbacb7ee7c5` for the audited upstream files. The only filtered-tree difference is an additional repository-local documentation file, `third_party/UNIV/INTEGRATION.md`, which does not modify upstream source behavior.

## Consistent key files

- `third_party/UNIV/README.md`
- `third_party/UNIV/requirements.txt`
- `third_party/UNIV/pretrain_mcmae.py`
- `third_party/UNIV/datasets/co_data_augmentation.py`
- `third_party/UNIV/datasets/datasets.py`
- `third_party/UNIV/SEG/MCMAE_SEG/configs/_base_/datasets/msrs.py`

## Inconsistent files

- `third_party/UNIV/INTEGRATION.md`: present only in the PSMAF-Net vendored tree as external integration documentation.

## Why this audit does not overwrite `third_party/UNIV`

No upstream source mismatch was found in the required key files. The only difference is an explanatory integration document, so automatic overwrite would risk removing local provenance notes without improving source consistency.

## Dataset package audit correction

`third_party/UNIV/pretrain_mcmae.py` imports `datasets.co_data_augmentation.CoDataAugmentation` and `datasets.datasets.RGB_pair_IR_dataset`, so the vendored `third_party/UNIV/datasets/` directory is required Python source. Future audits must include that directory when comparing the UNIV vendor tree. Only repository dataset-artifact roots such as `/datasets/` and `/data/` should be excluded by default.

## Follow-up recommendations

- Keep upstream UNIV files isolated under `third_party/UNIV` and document any future compatibility patches here before applying them.
- Implement PSMAF-Net research code outside `third_party/UNIV`, preferably under `core/psmaf`, `detection`, `segmentation`, or `tools`.
- Use the official UNIV requirements as the reference for full pretraining environments, but pin Detectron2/MMCV/MMSegmentation separately according to CUDA and PyTorch compatibility.
- Continue avoiding committed datasets, model weights, checkpoints, runs, results, logs, and generated CSV artifacts.
