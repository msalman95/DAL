import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_all_metrics, per_class_recall


def set_seed(seed):
    """Seed the Python, numpy and torch generators. Does nothing if seed is None."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_backbone(model):
    """Train the classification head only; freeze the pretrained feature extractor."""
    for name, param in model.named_parameters():
        param.requires_grad = (
            "backbone.fc" in name or "backbone.classifier" in name or "head" in name
        )


def unfreeze_all(model):
    """Train every parameter."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device, desc):
    """Run one training epoch. Returns the mean loss per sample."""
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, desc="eval"):
    """Evaluate on one split. Returns a metrics dict with per-class recall."""
    model.eval()
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        logits = model(images.to(device))
        preds_all.append(torch.argmax(logits, dim=1).cpu())
        labels_all.append(labels)

    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(labels_all).numpy()

    results = compute_all_metrics(y_true, y_pred, num_classes)
    results["tpr"] = per_class_recall(y_true, y_pred, num_classes).tolist()
    return results


def _summary(metrics):
    """One-line console summary of the scalar metrics."""
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k != "tpr")


def _tpr(metrics):
    """Per-class recall, one short field per class."""
    return " ".join(f"c{c}:{v:.3f}" for c, v in enumerate(metrics["tpr"]))


def train(cfg, model, criterion, optimizer,
          train_loader, val_loader, test_loader, out_dir):
    """Train, select the reported epoch, save its weights, and return the results."""
    device = torch.device(cfg["runtime"]["device"])
    num_classes = cfg["dataset"]["num_classes"]
    epochs = cfg["train"]["epochs"]
    freeze_epochs = cfg["train"]["freeze_epochs"]

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "model.pt")

    model = model.to(device)
    freeze_backbone(model)
    print(f"backbone frozen, training head only for {freeze_epochs} epochs")

    best_bacc = -1.0
    best = None  # (epoch, val metrics, test metrics)

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            unfreeze_all(model)
            print(f"\nepoch {epoch + 1}: backbone unfrozen, training all parameters")

        loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"epoch {epoch + 1}/{epochs} train",
        )
        print(f"epoch {epoch + 1}/{epochs} | loss {loss:.4f}")

        if val_loader is None:
            # cv protocol: no validation set, so the last epoch is reported.
            # Test is evaluated once, after the final epoch.
            if epoch == epochs - 1:
                test_metrics = evaluate(
                    model, test_loader, device, num_classes, desc="test"
                )
                best = (epoch, None, test_metrics)
                torch.save(model.state_dict(), ckpt_path)
        else:
            val_metrics = evaluate(model, val_loader, device, num_classes, desc="val")
            print(f"  val   {_summary(val_metrics)}")
            print(f"  val   tpr {_tpr(val_metrics)}")

            if val_metrics["bacc"] > best_bacc:
                best_bacc = val_metrics["bacc"]
                # Test is evaluated only for an epoch selected on validation.
                test_metrics = evaluate(
                    model, test_loader, device, num_classes, desc="test"
                )
                best = (epoch, val_metrics, test_metrics)
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> best val bacc {best_bacc:.4f}, checkpoint saved")

    epoch, val_metrics, test_metrics = best
    selection = "last" if val_loader is None else "best val bacc"

    print(f"\nselected epoch {epoch + 1} ({selection})")
    print(f"  test  {_summary(test_metrics)}")
    print(f"  test  tpr {_tpr(test_metrics)}")

    return {
        "dataset": cfg["dataset"]["name"],
        "protocol": cfg["dataset"].get("protocol", "holdout"),
        "head": cfg["model"]["head"],
        "loss": cfg["loss"]["name"],
        "backbone": cfg["model"]["backbone"],
        "selected_epoch": epoch + 1,
        "selection": selection,
        "checkpoint": ckpt_path,
        "val": val_metrics,
        "test": test_metrics,
    }


def save_results(results, out_dir):
    """Write the results dict to ``<out_dir>/results.json``."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {path}")