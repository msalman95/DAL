"""Models: a pretrained backbone plus one of two classification heads.

  linear : the vanilla baseline. A standard classifier head producing logits
           directly, used for every cross-entropy-family loss.
  baam   : Batch-Adaptive Angular Margin, the paper's head. Embeddings and
           class centres are L2-normalised, and an angular margin is added to
           the ground-truth logit. The margin is set per batch from the
           in-batch class counts, so rare classes get a larger margin.

DAL = BAAM (this file) + DWCE (losses.py).

Both heads share one forward signature: training returns logits, evaluation
returns logits without any margin.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter

BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, "fc"),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, "classifier"),
    "densenet161": (models.densenet161, models.DenseNet161_Weights.IMAGENET1K_V1, "classifier"),
}


def build_backbone(name, head_module):
    """Load an ImageNet-pretrained backbone and replace its final layer."""
    name = name.lower()
    if name not in BACKBONES:
        raise ValueError(f"Unsupported backbone '{name}'. Available: {sorted(BACKBONES)}")

    constructor, weights, attr = BACKBONES[name]
    backbone = constructor(weights=weights)
    in_features = getattr(backbone, attr).in_features
    setattr(backbone, attr, head_module(in_features))
    return backbone


def l2_norm(x, axis=1):
    """L2-normalise along ``axis``."""  
    return torch.div(x, torch.norm(x, 2, axis, True))


class Head(nn.Module):
    """Vanilla classifier head: in_features -> embedding_dim -> num_classes."""

    def __init__(self, in_features, embedding_dim=128, num_classes=7, dropout=0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class ArcHead(nn.Module):
    """Embedding head for angular losses: dropout -> linear -> BN -> L2-normalise."""

    def __init__(self, in_features, embedding_dim=128, dropout=0.4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False),
        )

    def forward(self, x):
        return F.normalize(self.layers(x), p=2, dim=1)


class BAAM(nn.Module):
    """Batch-Adaptive Angular Margin head.

    Margins are derived from the class counts of the current batch, so no
    global class distribution is needed and the margin follows the batch's own
    imbalance. For the classes present in a batch:

        w_c = log(1 + 1 / n_c)                    log inverse frequency
        t_c = (w_c - min w) / (max w - min w)     min-max over present classes
        m_c = m_min + t_c * (m_max - m_min)
    """

    def __init__(self, embedding_size=128, classnum=7, s=64.0, m_min=0.0, m_max=0.20):
        super().__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = float(s)
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.eps = 1e-4

    @torch.no_grad()
    def _per_sample_margins(self, labels):
        """Return the margin of each sample in the batch, shape [B]."""
        uniq, cnt = torch.unique(labels, return_counts=True)
        cnt = cnt.to(torch.float32)

        w_raw = torch.log1p(1.0 / (cnt + 1e-12))

        w_min, w_max = w_raw.min(), w_raw.max()
        denom = w_max - w_min
        if denom < 1e-12:
            # Every class present in the batch has the same count: no imbalance
            # signal to act on, so fall back to the smallest margin.
            margins = torch.full_like(w_raw, self.m_min)
        else:
            t = (w_raw - w_min) / (denom + 1e-12)
            margins = self.m_min + t * (self.m_max - self.m_min)

        # Scatter into a full class-indexed vector, then gather by label. This
        # gives the same values as a per-sample lookup without a host sync.
        m_class = torch.full(
            (self.classnum,), self.m_min, device=labels.device, dtype=torch.float32
        )
        m_class[uniq] = margins
        return m_class[labels]

    def forward(self, embeddings, labels):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm).clamp(-1 + self.eps, 1 - self.eps)

        theta = torch.acos(cosine)
        idx = torch.arange(labels.size(0), device=labels.device)

        theta_m = theta.clone()
        theta_m[idx, labels] = torch.clamp(
            theta_m[idx, labels] + self._per_sample_margins(labels),
            min=self.eps,
            max=math.pi - self.eps,
        )
        return torch.cos(theta_m) * self.s


class Model(nn.Module):
    """Backbone plus head. ``head`` is None for the vanilla linear model."""

    def __init__(self, backbone, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, labels=None, return_embeddings=False):
        out = self.backbone(x)

        if self.head is None:
            # Vanilla: the backbone's own head already produces logits.
            return out

        # Angular: the backbone produces L2-normalised embeddings.
        if return_embeddings:
            return out
        if self.training:
            if labels is None:
                raise ValueError("labels are required during training for the BAAM head.")
            return self.head(out, labels)

        # Evaluation applies no margin: plain scaled cosine similarity.
        return torch.mm(out, l2_norm(self.head.kernel, axis=0)) * self.head.s


def build_model(cfg):
    """Build the model described by ``cfg['model']``."""
    cfg_model = cfg["model"]
    name = cfg_model["head"].lower()

    num_classes = cfg["dataset"]["num_classes"]
    embedding_dim = cfg_model["embedding_dim"]
    dropout = cfg_model["dropout"]

    if name == "linear":
        backbone = build_backbone(
            cfg_model["backbone"],
            lambda f: Head(f, embedding_dim, num_classes, dropout),
        )
        return Model(backbone)

    if name == "baam":
        backbone = build_backbone(
            cfg_model["backbone"],
            lambda f: ArcHead(f, embedding_dim, dropout),
        )
        head = BAAM(
            embedding_size=embedding_dim,
            classnum=num_classes,
            s=cfg_model["s"],
            m_min=cfg_model["m_min"],
            m_max=cfg_model["m_max"],
        )
        return Model(backbone, head)

    raise ValueError(f"Unknown head '{name}'. Available: linear, baam")