import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

SPLIT_FILES = {
    "train": ("ISIC_2019_Training_Input", "ISIC_2019_Training_GroundTruth.csv"),
    "test": ("ISIC_2019_Test_Input", "ISIC_2019_Test_GroundTruth.csv"),
}


class ISIC2019Dataset(Dataset):

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
    """Read one split, drop UNK rows, and assign label indices."""
    image_subdir, csv_name = SPLIT_FILES[split]
    image_dir = os.path.join(root, image_subdir)

    df = pd.read_csv(os.path.join(root, csv_name))

    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[isic2019:{split}] CSV is missing class columns: {missing}")

    if "UNK" in df.columns:
        n_before = len(df)
        df = df[df["UNK"] == 0].copy()
        if len(df) != n_before:
            print(f"[isic2019:{split}] dropped {n_before - len(df)} UNK rows.")

    df["image_path"] = df["image"].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

    n_listed = len(df)
    df = df[df["image_path"].apply(os.path.exists)].copy()
    if len(df) != n_listed:
        print(f"[isic2019:{split}] {n_listed - len(df)} of {n_listed} images "
              f"listed in the CSV are missing on disk and were dropped.")

    df["label_index"] = df[LABEL_COLS].values.argmax(axis=1)
    return df.reset_index(drop=True)


def build_isic2019(cfg, transform):
    """Build (train, val, test) datasets for the holdout protocol."""
    protocol = cfg.get("protocol", "holdout").lower()
    if protocol != "holdout":
        raise ValueError(
            f"ISIC 2019 supports the 'holdout' protocol only, got '{protocol}'."
        )

    root = cfg["root"]
    df_all = _load_split(root, "train")

    train_df, val_df = train_test_split(
        df_all,
        test_size=cfg.get("val_fraction", 0.2),
        stratify=df_all["label_index"],
        random_state=cfg.get("split_seed"),  # None reproduces the paper runs
    )
    test_df = _load_split(root, "test")

    print(f"[isic2019] holdout — train {len(train_df)} | "
          f"val {len(val_df)} | test {len(test_df)}")
    return (
        ISIC2019Dataset(train_df, transform),
        ISIC2019Dataset(val_df, transform),
        ISIC2019Dataset(test_df, transform),
    )