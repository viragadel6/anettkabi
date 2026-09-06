"""ODE samplers (Euler / midpoint) with schedules, progress, and cancellation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["Schedule", "build_schedule", "euler_step", "integrate_flow", "midpoint_step"]

ProgressCallback = Callable[[float], None]
CancelCallback = Callable[[], bool]
VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class Schedule:
    """A discrete set of flow times from 0 (noise) to 1 (data).

    Attributes:
        times: Increasing tensor of times, starting near 0, ending exactly 1.
    """

    times: torch.Tensor

    @property
    def steps(self) -> int:
        """Return the number of solver steps (times - 1)."""
        return max(0, self.times.numel() - 1)


def build_schedule(
    num_steps: int,
    *,
    device: torch.device,
    mode: str = "uniform",
    shift: float = 1.0,
    t_start: float = 0.0,
) -> Schedule:
    """Construct the time grid for `num_steps` solver steps.

    `shifted` applies the resolution-aware remap
    ``t' = shift * t / (1 + (shift - 1) * t)`` which allocates more steps to
    the high-noise region.

    Parameters:
        num_steps: Number of integration steps (>= 1).
        device: Target device.
        mode: `uniform` or `shifted`.
        shift: Shift strength for `shifted` mode.
        t_start: Initial time (0 for full sampling; >0 for latent re-noising).

    Returns:
        Schedule with `num_steps + 1` time points ending at 1.0.

    Raises:
        ValueError: For invalid step counts or modes.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if mode not in ("uniform", "shifted"):
        raise ValueError(f"unknown schedule mode {mode!r}")
    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float64)
    if mode == "shifted" and shift != 1.0:
        base = shift * base / (1.0 + (shift - 1.0) * base)
    if t_start > 0.0:
        span = 1.0 - t_start
        base = t_start + span * base
    base[0] = max(base[0].item(), 0.0)
    base[-1] = 1.0
    return Schedule(times=base.to(torch.float32))


def euler_step(
    x: torch.Tensor, velocity: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """One explicit Euler update.

    Parameters:
        x: Current state.
        velocity: Evaluated velocity.
        dt: Time increment (scalar tensor).

    Returns:
        The next state.
    """
    return x + velocity.to(x.dtype) * dt.to(x.dtype)


def midpoint_step(
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    velocity_fn: VelocityFn,
) -> torch.Tensor:
    """One explicit midpoint (RK2) update.

    Parameters:
        x: Current state.
        t: Current time.
        dt: Time increment.
        velocity_fn: Velocity evaluator.

    Returns:
        The next state.
    """
    half_dt = dt * 0.5
    mid_velocity = velocity_fn(x, t + half_dt)
    x_mid = x + mid_velocity.to(x.dtype) * half_dt.to(x.dtype)
    end_velocity = velocity_fn(x_mid, t + dt)
    return x + end_velocity.to(x.dtype) * dt.to(x.dtype)


def integrate_flow(
    x0: torch.Tensor,
    velocity_fn: VelocityFn,
    *,
    num_steps: int,
    schedule_mode: str = "uniform",
    shift: float = 1.0,
    solver: str = "euler",
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
    t_start: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Integrate the probability-flow ODE from noise to data.

    Parameters:
        x0: Initial noise [B, T, C].
        velocity_fn: Callable (x, t_scalar_tensor) -> velocity.
        num_steps: Exact number of solver steps.
        schedule_mode: uniform|shifted.
        shift: Shift strength.
        solver: euler|midpoint.
        progress_cb: Optional callback receiving fraction in [0, 1].
        cancel_cb: Optional cooperative-cancellation check between steps.
        t_start: Initial time (for continuation sampling).
        device: Optional device override.

    Returns:
        The integrated latents at t = 1.

    Raises:
        RuntimeError: When cancellation is requested.
        ValueError: For unknown solvers.
    """
    target_device = device or x0.device
    schedule = build_schedule(
        num_steps, device=target_device, mode=schedule_mode, shift=shift, t_start=t_start
    )
    if solver not in ("euler", "midpoint"):
        raise ValueError(f"unknown solver {solver!r}")
    x = x0.to(target_device)
    total = schedule.steps
    for index in range(total):
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("sampling canceled by request")
        t_now = schedule.times[index]
        t_next = schedule.times[index + 1]
        dt = t_next - t_now
        if solver == "euler":
            velocity = velocity_fn(x, t_now.reshape(1).to(target_device))
            x = euler_step(x, velocity, dt)
        else:
            x = midpoint_step(
                x,
                t_now.reshape(1).to(target_device),
                dt,
                velocity_fn,
            )
        if progress_cb is not None:
            progress_cb((index + 1) / total)
    return x
