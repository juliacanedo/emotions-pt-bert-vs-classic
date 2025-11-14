"""
Utilitários gerais do projeto emotions-pt.

Módulos principais:
- io: funções de I/O, seeds e Paths
- metrics: métricas agregadas e por classe
- timing: medidor de tempo e uso de VRAM (GPU)
"""

from .io import (
    ensure_dir,
    set_global_seed,
    read_csv,
    write_parquet,
    read_parquet,
    save_json,
    load_json,
    Paths,
)

from .metrics import compute_all_metrics

from .timing import TimerVRAM, TimeVRAM

__all__ = [
    # io
    "ensure_dir",
    "set_global_seed",
    "read_csv",
    "write_parquet",
    "read_parquet",
    "save_json",
    "load_json",
    "Paths",
    # metrics
    "compute_all_metrics",
    # timing
    "TimerVRAM",
    "TimeVRAM",
]
