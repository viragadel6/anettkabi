"""Shared training engine: EMA, optimizers, checkpointing, resume, validation."""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

__all__ = ["EmaModel", "TrainingState", "fit_loop", "load_checkpoint", "save_checkpoint"]


class EmaModel:
    """Exponential moving average of a model's parameters."""

    __slots__ = ("decay", "shadow")

    def __init__(self, model: nn.Module, decay: float) -> None:
        """Initialize the shadow copy.

        Parameters:
            model: Source model.
            decay: EMA decay coefficient.
        """
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the live model into the shadow.

        Parameters:
            model: Current training model.
        """
        for shadow, live in zip(self.shadow.state_dict().values(), model.state_dict().values(), strict=True):
            if live.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(live.detach(), alpha=1 - self.decay)
            else:
                shadow.copy_(live)

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite a model with the averaged weights.

        Parameters:
            model: Destination model.
        """
        model.load_state_dict(self.shadow.state_dict(), strict=True)


@dataclass
class TrainingState:
    """Bookkeeping for one training stage.

    Attributes:
        stage: Stage name (vae/vocoder/sync/generator).
        epoch: Current epoch (0-based).
        step: Global step counter.
        best_loss: Best validation loss so far.
        history: Loss history entries.
    """

    stage: str
    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    history: list[dict[str, float]] = field(default_factory=list)


def save_checkpoint(
    directory: Path,
    stage: str,
    model: nn.Module,
    ema: EmaModel | None,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
) -> Path:
    """Persist a resumable checkpoint atomically.

    Parameters:
        directory: Checkpoint directory.
        stage: Stage name.
        model: Live model.
        ema: Optional EMA shadow.
        optimizer: Optimizer.
        state: Bookkeeping state.

    Returns:
        The checkpoint path.

    Raises:
        OSError: When the write fails.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}-last.pt"
    tmp = path.with_suffix(".pt.tmp")
    payload: dict[str, Any] = {
        "stage": stage,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": asdict(state),
    }
    if ema is not None:
        payload["ema"] = ema.shadow.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    directory: Path, stage: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> tuple[EmaModel | None, TrainingState]:
    """Restore the latest checkpoint when present.

    Parameters:
        directory: Checkpoint directory.
        stage: Stage name.
        model: Model to hydrate.
        optimizer: Optional optimizer to hydrate.

    Returns:
        (ema or None, restored TrainingState).

    Raises:
        RuntimeError: When the checkpoint is structurally incompatible.
    """
    path = directory / f"{stage}-last.pt"
    if not path.is_file():
        return None, TrainingState(stage=stage)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    unexpected = model.load_state_dict(payload["model"], strict=False)[1]
    if unexpected:
        raise RuntimeError(f"checkpoint for {stage} has unexpected keys: {unexpected[:5]}")
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    state = TrainingState(**payload["state"])
    ema = None
    if "ema" in payload:
        ema = EmaModel(model, decay=0.999)
        ema.shadow.load_state_dict(payload["ema"], strict=True)
    return ema, state


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int = 200) -> float:
    """Cosine schedule with linear warmup.

    Parameters:
        step: Current step.
        total_steps: Total planned steps.
        base_lr: Peak learning rate.
        warmup: Warmup steps.

    Returns:
        The learning rate for this step.
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def fit_loop(
    *,
    stage: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    epochs: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    checkpoint_dir: Path,
    ema_decay: float,
    accumulation_steps: int,
    loss_fn: Callable[[nn.Module, dict[str, Any]], tuple[torch.Tensor, dict[str, float]]],
    evaluate_fn: Callable[[nn.Module, dict[str, Any]], dict[str, float]],
    log_every: int = 50,
) -> TrainingState:
    """Generic training loop with EMA, checkpointing, and best-tracking.

    Parameters:
        stage: Stage name.
        model: Model to train.
        optimizer: Optimizer.
        train_dataset / val_dataset: Datasets.
        epochs: Epoch budget.
        batch_size: Batch size.
        device: Target device.
        num_workers: DataLoader workers.
        checkpoint_dir: Where checkpoints are written.
        ema_decay: EMA coefficient (0 disables).
        accumulation_steps: Gradient accumulation.
        loss_fn: Callable (model, batch) -> (loss, parts).
        evaluate_fn: Callable (model, batch) -> metrics dict.
        log_every: Log cadence in steps.

    Returns:
        The final TrainingState.

    Raises:
        RuntimeError: When the training data is empty.
    """
    if len(train_dataset) == 0:  # type: ignore[arg-type]
        raise RuntimeError(f"{stage}: empty training dataset")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=device.type == "cuda",
    )
    model.to(device).train()
    ema = EmaModel(model, ema_decay) if ema_decay > 0 else None
    state = TrainingState(stage=stage)
    if (checkpoint_dir / f"{stage}-last.pt").exists():
        restored_ema, state = load_checkpoint(checkpoint_dir, stage, model, optimizer)
        if restored_ema is not None and ema is not None:
            ema = restored_ema
    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", param_group["lr"])
    optimizer.zero_grad(set_to_none=True)
    total_steps = epochs * max(1, len(train_loader))
    started = time.time()
    while state.epoch < epochs:
        state.epoch += 1
        for raw_batch in train_loader:
            state.step += 1
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in raw_batch.items()}
            for scheduler_param in optimizer.param_groups:
                scheduler_param["lr"] = cosine_lr(state.step, total_steps, scheduler_param["initial_lr"])
            loss, parts = loss_fn(model, batch)
            (loss / accumulation_steps).backward()
            if state.step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if state.step % log_every == 0:
                elapsed = time.time() - started
                print(
                    f"[{stage}] epoch {state.epoch} step {state.step} "
                    f"loss {float(loss):.5f} parts {parts} lr {optimizer.param_groups[0]['lr']:.2e} "
                    f"elapsed {elapsed:.0f}s"
                )
                state.history.append({"step": float(state.step), "loss": float(loss), **parts})
        candidate = ema.shadow if ema is not None else model
        candidate.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in raw_batch.items()}
                metrics = evaluate_fn(candidate, batch)
                val_loss += metrics.get("loss", 0.0)
                val_count += 1
        val_loss = val_loss / max(1, val_count)
        candidate.train()
        print(f"[{stage}] epoch {state.epoch} validation loss {val_loss:.5f}")
        state.history.append({"step": float(state.step), "val_loss": val_loss})
        if val_loss < state.best_loss:
            state.best_loss = val_loss
            target = directory_best(checkpoint_dir, stage)
            torch.save(
                {"model": candidate.state_dict(), "state": asdict(state)},
                target,
            )
        save_checkpoint(checkpoint_dir, stage, model, ema, optimizer, state)
    _write_history(checkpoint_dir, stage, state)
    return state


def directory_best(checkpoint_dir: Path, stage: str) -> Path:
    """Return the best-checkpoint path for a stage.

    Parameters:
        checkpoint_dir: Checkpoint directory.
        stage: Stage name.

    Returns:
        Path of the best checkpoint.
    """
    return checkpoint_dir / f"{stage}-best.pt"


def _write_history(checkpoint_dir: Path, stage: str, state: TrainingState) -> None:
    """Persist loss history as JSON.

    Parameters:
        checkpoint_dir: Checkpoint directory.
        stage: Stage name.
        state: Final state.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / f"{stage}-history.json").write_text(
        json.dumps(state.history[-1000:], indent=2), encoding="utf-8"
    )
