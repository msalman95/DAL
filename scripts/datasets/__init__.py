import numpy as np
from torch.utils.data import DataLoader

from .aptos import build_aptos
from .isic2018 import build_isic2018
from .isic2019 import build_isic2019
from .transforms import build_transform

BUILDERS = {
    "isic2018": build_isic2018,
    "isic2019": build_isic2019,
    "aptos": build_aptos,
}


def build_datasets(cfg):
    """Build the train, validation and test datasets named in the config.

    Returns (train_ds, val_ds, test_ds). ``val_ds`` is None under the cv
    protocol.
    """
    cfg_ds = cfg["dataset"]
    name = cfg_ds["name"].lower()
    if name not in BUILDERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(BUILDERS)}")

    transform = build_transform(cfg)
    return BUILDERS[name](cfg_ds, transform)


def count_classes(dataset, num_classes):
    """Return the number of training samples per class, as a list of length C."""
    labels = np.asarray(dataset.labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)

    if len(counts) > num_classes:
        raise ValueError(
            f"Found label index {len(counts) - 1} but num_classes={num_classes}. "
            f"Check the dataset config."
        )
    empty = np.where(counts == 0)[0]
    if len(empty):
        print(f"[datasets] warning: classes {empty.tolist()} have no training samples.")

    return counts.tolist()


def build_dataloaders(cfg):
    """Build the DataLoaders and the per-class training counts.

    Returns:
        train_loader, val_loader, test_loader, class_counts

    ``val_loader`` is None under the cv protocol.
    """
    train_ds, val_ds, test_ds = build_datasets(cfg)

    num_classes = cfg["dataset"]["num_classes"]
    train_cfg = cfg["train"]
    runtime = cfg["runtime"]

    def loader(dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=runtime["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

    train_loader = loader(train_ds, train_cfg["batch_size"], shuffle=True)
    test_loader = loader(test_ds, train_cfg["eval_batch_size"], shuffle=False)
    val_loader = (
        None if val_ds is None
        else loader(val_ds, train_cfg["eval_batch_size"], shuffle=False)
    )

    return train_loader, val_loader, test_loader, count_classes(train_ds, num_classes)