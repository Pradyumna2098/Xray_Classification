"""Shared training utilities for notebook experiments.

This module exposes a convenience function for configuring a suite of
TensorFlow/Keras callbacks that we reuse across the custom CNN and the
transfer-learning notebooks.  The callbacks take care of:

* early stopping with best-weight restoration
* reducing the learning rate when validation metrics stall
* persisting the top-performing weights under ``models/``
* logging to TensorBoard (and optionally to Weights & Biases if installed)
"""

from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


def configure_training_run(
    run_name: Optional[str] = None,
    *,
    monitor: str = "val_loss",
    patience: int = 5,
    min_delta: float = 1e-3,
    log_root: Path | str = Path("logs"),
    model_root: Path | str = Path("models"),
) -> Tuple[List, Path, Path]:
    """Create a reusable set of callbacks for a notebook training run.

    Parameters
    ----------
    run_name:
        Optional identifier for the experiment.  When omitted a timestamped
        name is generated automatically so that each invocation writes to a
        distinct directory.
    monitor:
        Metric name monitored by the callbacks.  Defaults to ``"val_loss"``.
    patience:
        Number of epochs with no improvement before early stopping triggers.
    min_delta:
        Minimum change in the monitored quantity to qualify as an improvement.
    log_root / model_root:
        Base directories used for TensorBoard logs and saved weights.

    Returns
    -------
    Tuple[List, Path, Path]
        A tuple containing ``callbacks``, ``log_dir`` and ``checkpoint_path``.
    """

    log_root = Path(log_root)
    model_root = Path(model_root)
    log_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    run_name = run_name or datetime.now().strftime("run_%Y%m%d-%H%M%S")
    log_dir = log_root / run_name
    checkpoint_path = model_root / f"{run_name}_best.weights.h5"
    log_dir.mkdir(parents=True, exist_ok=True)

    mode = "min" if "loss" in monitor or monitor.startswith("val_") else "max"
    patience = max(1, patience)
    lr_patience = max(1, patience // 2)

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=lr_patience,
            min_delta=min_delta,
            mode=mode,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=str(log_dir), histogram_freq=1),
    ]

    wandb_callback = _maybe_get_wandb_callback(run_name, monitor)
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    return callbacks, log_dir, checkpoint_path


def _maybe_get_wandb_callback(run_name: str, monitor: str):
    """Return a configured Weights & Biases callback when the library exists."""

    spec = importlib.util.find_spec("wandb")
    if spec is None:
        return None

    wandb = importlib.import_module("wandb")
    if wandb.run is None:
        wandb.init(project="xray-classification", name=run_name, reinit=True)

    wandb_keras = importlib.import_module("wandb.keras")
    WandbCallback = getattr(wandb_keras, "WandbCallback")
    return WandbCallback(monitor=monitor, save_model=False)


__all__ = ["configure_training_run"]
