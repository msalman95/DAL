import os
import sys

import torch.optim as optim
import yaml

from scripts.config import load_config
from scripts.datasets import build_dataloaders
from scripts.engine import save_results, set_seed, train
from scripts.losses import build_loss
from scripts.models import build_model


def run_dir(cfg):
    """Build the output directory name from the config."""
    parts = [cfg["dataset"]["name"], cfg["model"]["head"], cfg["loss"]["name"]]
    if cfg["dataset"].get("protocol") == "cv":
        parts.append(f"fold{cfg['dataset'].get('fold', 0)}")
    return os.path.join(cfg["runtime"]["out_dir"], "_".join(parts))


def main():
    cfg = load_config(sys.argv[1], sys.argv[2:])
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))

    set_seed(cfg["runtime"]["seed"])

    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg)
    criterion = build_loss(cfg)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    out_dir = run_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    results = train(
        cfg, model, criterion, optimizer,
        train_loader, val_loader, test_loader, out_dir,
    )
    save_results(results, out_dir)


if __name__ == "__main__":
    main()