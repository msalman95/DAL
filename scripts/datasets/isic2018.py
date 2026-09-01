"""ISIC 2018 Task 3 (skin lesion classification, 7 classes).

Expected layout under ``dataset.root``:

    ISIC2018_Task3_Training_Input/
    ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv
    ISIC2018_Task3_Validation_Input/
    ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv
    ISIC2018_Task3_Test_Input/
    ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv

All three ground-truth CSVs share the same schema: an ``image`` column plus one
one-hot column per class.

Two protocols:

  holdout : official train and val are pooled, then split ``val_fraction``
            stratified into train/val. The official test set is held out.
  cv      : k-fold stratified CV over the official *training* set only. The
            held-out fold is the test set; there is no validation set.
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

# CSV column order. This defines the class indices (MEL=0 ... VASC=6) and must
# not be reordered — trained checkpoints depend on it.
LABEL_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

SPLIT_DIRS = {
    "train": ("ISIC2018_Task3_Training_Input",
              "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"),
    "val": ("ISIC2018_Task3_Validation_Input",
            "ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"),
    "test": ("ISIC2018_Task3_Test_Input",
             "ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"),
}


class ISIC2018Dataset(Dataset):
    """DataFrame-backed dataset. Expects ``image_path`` and ``label_index``."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = self.df["label_index"].to_numpy(dtype=np.int64)
        self.class_names = LABEL_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(int(row["label_index"]), dtype=torch.long)


def _load_split(root, split):
    """Read one official split into a DataFrame of image paths and label indices."""
    image_subdir, csv_subpath = SPLIT_DIRS[split]
    image_dir = os.path.join(root, image_subdir)
    csv_path = os.path.join(root, csv_subpath)

    df = pd.read_csv(csv_path)
    df["image_path"] = df["image"].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

    n_listed = len(df)
    df = df[df["image_path"].apply(os.path.exists)].copy()
    if len(df) != n_listed:
        print(f"[isic2018:{split}] {n_listed - len(df)} of {n_listed} images "
              f"listed in the CSV are missing on disk and were dropped.")

    df["label_index"] = df[LABEL_COLS].values.argmax(axis=1)
    return df.reset_index(drop=True)


def build_isic2018(cfg, transform):
    """Build (train, val, test) datasets. ``val`` is None under the cv protocol."""
    root = cfg["root"]
    protocol = cfg.get("protocol", "holdout").lower()

    if protocol == "holdout":
        # Official train + val pooled, then split stratified.
        df_all = pd.concat(
            [_load_split(root, "train"), _load_split(root, "val")],
            axis=0,
        ).reset_index(drop=True)

        train_df, val_df = train_test_split(
            df_all,
            test_size=cfg.get("val_fraction", 0.2),
            stratify=df_all["label_index"],
            random_state=cfg.get("split_seed"),  # None reproduces the paper runs
        )
        test_df = _load_split(root, "test")

        print(f"[isic2018] holdout — train {len(train_df)} | "
              f"val {len(val_df)} | test {len(test_df)}")
        return (
            ISIC2018Dataset(train_df, transform),
            ISIC2018Dataset(val_df, transform),
            ISIC2018Dataset(test_df, transform),
        )

    if protocol == "cv":
        # Official training set only. The held-out fold is the test set.
        df_all = _load_split(root, "train")

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

        print(f"[isic2018] cv fold {fold + 1}/{n_folds} — "
              f"train {len(train_df)} | test {len(test_df)}")
        return (
            ISIC2018Dataset(train_df, transform),
            None,
            ISIC2018Dataset(test_df, transform),
        )

    raise ValueError(f"Unknown protocol '{protocol}'. Use 'holdout' or 'cv'.")