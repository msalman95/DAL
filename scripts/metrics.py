import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def per_class_recall(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(cm) / support
    return np.where(support > 0, recall, np.nan)


def compute_iba(y_true, y_pred, num_classes, alpha=1.0):
    """Index of balanced accuracy, averaged over the classes present.

    Per class: IBA = (1 + alpha * (TPR - TNR)) * TPR * TNR.

    Reference: Garcia, Mollineda and Sanchez, "Index of Balanced Accuracy:
    A Performance Measure for Skewed Class Distributions", IbPRIA 2009.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    total = cm.sum()

    scores = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - (tp + fn + fp)

        if (tp + fn) == 0:
            continue  # class absent from y_true

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        scores.append((1 + alpha * (tpr - tnr)) * tpr * tnr)

    return float(np.mean(scores)) if scores else float("nan")


def compute_cov(y_true, y_pred, num_classes):
    """Coefficient of variation of per-class recall, std / mean."""
    recall = per_class_recall(y_true, y_pred, num_classes)
    recall = recall[~np.isnan(recall)]

    if len(recall) == 0:
        return float("nan")
    mean = recall.mean()
    if mean == 0:
        return float("nan")  # no class is ever recalled; the ratio is undefined
    return float(recall.std(ddof=0) / mean)


def compute_all_metrics(y_true, y_pred, num_classes):
    """Return the five reported metrics as a dict."""
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "iba": compute_iba(y_true, y_pred, num_classes),
        "cov": compute_cov(y_true, y_pred, num_classes),
    }