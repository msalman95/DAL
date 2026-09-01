# DAL: Dynamic Angular Loss for Imbalanced Medical Image Classification

Official implementation of *Dynamic Angular Loss for Imbalanced Medical Image Classification* (BMVC 2026).

DAL combines a batch-adaptive angular margin (BAAM) with a dynamically weighted cross-entropy (DWCE). Both derive their per-class strength from the in-batch class counts, so rare classes receive a larger margin and a larger weight without needing the global class distribution.

## Datasets

1. ISIC 2018 — https://challenge.isic-archive.com/data/#2018
2. ISIC 2019 — https://challenge.isic-archive.com/data/#2019
3. APTOS — https://www.kaggle.com/c/aptos2019-blindness-detection/data

Set `dataset.root` in the corresponding config to your local copy.

## Usage

Baseline:

```bash
python train.py configs/isic2018.yaml
```

DAL:

```bash
python train.py configs/isic2018.yaml model.head=baam loss.name=dwce
```

Any config value can be overridden on the command line:

```bash
python train.py configs/isic2018.yaml model.head=baam loss.name=dwce model.s=8 runtime.device=cuda:3
```

