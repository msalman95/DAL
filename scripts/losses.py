import torch
import torch.nn as nn
import torch.nn.functional as F


class DWCE(nn.Module):
    """Dynamically Weighted Cross-Entropy.

    Class weights are recomputed for every batch from the in-batch class
    counts, so no global class distribution is needed and the weighting
    follows the batch's own composition:

        w_c = B / (n_c * C)

    where B is the batch size, n_c the count of class c in the batch, and C
    the number of classes.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] raw scores.
            targets: [B] class indices.
        Returns:
            Scalar loss.
        """
        counts = torch.bincount(targets, minlength=self.num_classes).float()

        weights = torch.zeros_like(counts)
        present = counts > 0
        weights[present] = targets.size(0) / (counts[present] * self.num_classes)

        return F.cross_entropy(logits, targets, weight=weights)


def build_loss(cfg):
    """Build the loss described by ``cfg['loss']``."""
    name = cfg["loss"]["name"].lower()
    num_classes = cfg["dataset"]["num_classes"]

    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dwce":
        return DWCE(num_classes=num_classes)

    raise ValueError(f"Unknown loss '{name}'. Available: ce, dwce")