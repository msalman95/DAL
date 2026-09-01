"""Configuration loading.

A run is one dataset YAML plus optional ``key=value`` overrides. Dotted keys
address nested entries:

    python train.py configs/isic2018.yaml
    python train.py configs/isic2018.yaml model.head=baam loss.name=dwce
    python train.py configs/isic2018.yaml model.head=baam model.s=8 runtime.device=cuda:3
"""

import yaml


def _apply_override(cfg, item):
    """Apply one ``a.b.c=value`` override to the config."""
    if "=" not in item:
        raise ValueError(f"Override '{item}' is not of the form key=value.")

    key, value = item.split("=", 1)
    parts = key.strip().split(".")

    node = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            raise KeyError(f"Override '{key}' addresses no section '{part}'.")
        node = node[part]

    node[parts[-1]] = yaml.safe_load(value.strip())


def load_config(config_path, overrides=()):
    """Load a config file and apply ``key=value`` overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for item in overrides:
        _apply_override(cfg, item)

    return cfg