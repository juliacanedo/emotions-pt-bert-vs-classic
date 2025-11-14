import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Paths:
    data_raw: str
    data_processed: str
    external: str
    outputs_models: str
    outputs_metrics: str
    outputs_curves: str
    outputs_reports: str

    @classmethod
    def from_cfg(cls, cfg_paths: Dict[str, str]) -> "Paths":
        for p in cfg_paths.values():
            ensure_dir(p)
        return cls(**cfg_paths)
