"""settings shared by run_final and the app (models/config.json)."""
import json

CONFIG_PATH = "models/config.json"

DEFAULT = {
    "weights": {"weighting": "none", "lib_weight": 1.0, "review_weight": 1.0},
    "als": {"factors": 32, "regularization": 1.0, "alpha": 10.0, "iterations": 10},
    "hybrid": {"lam": 0.0, "beta": 0.0},
}


def load_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT


def save_config(cfg: dict, path: str = CONFIG_PATH) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
