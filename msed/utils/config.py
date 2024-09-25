import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class Config:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)


def load_config(experiment_dir: Path) -> Dict:

    # Load config from json file
    json_file = experiment_dir / "config.json"
    with open(json_file, "r") as f:
        config = json.load(f)

    # Augment config with determined threshold
    json_file = experiment_dir / "results_eval.json"
    with open(json_file, "r") as f:
        threshold = json.load(f)["threshold"]
    config["threshold"] = threshold

    return Config(**config)
