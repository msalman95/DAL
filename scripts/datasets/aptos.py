import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

NUM_CLASSES = 5  # diabetic retinopathy grades 0-4, used directly as label indices


class APTOSDataset(Dataset):
    """DataFrame-backed dataset. Expects ``image_path`` and ``label_index``."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = self.df["label_index"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(int(row["label_index"]), dtype=torch.long)


def _load_pool(cfg):
    """Read the full APTOS pool into a DataFrame of image paths and labels."""
    root = cfg["root"]
    image_dir = os.path.join(root, cfg.get("image_dir", "images"))
    image_ext = cfg.get("image_ext", ".png")

    df = pd.read_csv(os.path.join(root, cfg.get("csv_name", "aptos_all.csv")))
    df["image_path"] = df["id_code"].astype(str).apply(
        lambda x: os.path.join(image_dir, f"{x}{image_ext}")
    )

    n_listed = len(df)
    df = df[df["image_path"].apply(os.path.exists)].copy()
    if len(df) != n_listed:
        print(f"[aptos] {n_listed - len(df)} of {n_listed} images listed in the "
              f"CSV are missing on disk and were dropped.")

    df["label_index"] = df["diagnosis"].astype(int)
    return df.reset_index(drop=True)


def build_aptos(cfg, transform):
    """Build (train, val, test) datasets. ``val`` is None under the cv protocol."""
    protocol = cfg.get("protocol", "holdout").lower()
    df_all = _load_pool(cfg)

    if protocol == "holdout":
        seed = cfg.get("split_seed")  # None reproduces the paper runs
        test_fraction = cfg.get("test_fraction", 0.2)
        val_fraction = cfg.get("val_fraction", 0.2)  # of the trainval part

        trainval_df, test_df = train_test_split(
            df_all,
            test_size=test_fraction,
            stratify=df_all["label_index"],
            random_state=seed,
        )
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=val_fraction,
            stratify=trainval_df["label_index"],
            random_state=seed,
        )

        print(f"[aptos] holdout — train {len(train_df)} | "
              f"val {len(val_df)} | test {len(test_df)}")
        return (
            APTOSDataset(train_df, transform),
            APTOSDataset(val_df, transform),
            APTOSDataset(test_df, transform),
        )

    if protocol == "cv":
        n_folds = cfg.get("n_folds", 5)
        fold = cfg.get("fold", 0)
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=cfg.get("fold_seed", 42),
        )
        splits = list(skf.split(df_all["image_path"], df_all["label_index"]))
        train_index, test_index = splits[fold]

        train_df = df_all.iloc[train_index]
        test_df = df_all.iloc[test_index]

        print(f"[aptos] cv fold {fold + 1}/{n_folds} — "
              f"train {len(train_df)} | test {len(test_df)}")
        return (
            APTOSDataset(train_df, transform),
            None,
            APTOSDataset(test_df, transform),
        )

    raise ValueError(f"Unknown protocol '{protocol}'. Use 'holdout' or 'cv'.")