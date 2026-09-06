"""Rectified-flow (flow matching) objective and classifier-free guidance helpers.

Convention: ``x0`` is Gaussian noise, ``x1`` is data, and the interpolation is
``x_t = (1 - t) * x0 + t * x1`` for ``t in [0, 1]``, so integration runs from
``t = 0`` (pure noise) to ``t = 1`` (data). The target velocity field of this
path is constant along the trajectory: ``v*(x_t, t) = x1 - x0``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = [
    "FlowMatchingModel",
    "VelocityField",
    "conditional_velocity",
    "flow_matching_loss",
    "sample_t",
]

VelocityField = Callable[..., torch.Tensor]


@dataclass(slots=True)
class FlowMatchingModel:
    """Wraps an MMDiT velocity predictor for training and guided sampling.

    Attributes:
        velocity_fn: Callable mapping (x_t, t, **conditioning) -> velocity.
        guidance_scale: CFG scale; 0 disables guidance (single forward pass).
        guidance_rescale: φ rescaling factor in [0, 1] against over-saturation.
    """

    velocity_fn: VelocityField
    guidance_scale: float = 4.5
    guidance_rescale: float = 0.7

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: dict[str, torch.Tensor | None],
        uncond_conditioning: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Evaluate the CFG-combined velocity field.

        With `guidance_scale == 0` only one conditional pass is computed
        (true unconditional-off mode is achieved by passing empty conditions).

        Parameters:
            x_t: [B, T, C] current latents.
            t: [B] flow time.
            conditioning: Full (text+video) conditioning tensors.
            uncond_conditioning: Empty-text + null-video conditioning tensors.

        Returns:
            [B, T, C] guided velocity.
        """
        scale = self.guidance_scale
        if scale == 0.0:
            return self.velocity_fn(x_t, t, **conditioning)
        cond = self.velocity_fn(x_t, t, **conditioning)
        uncond = self.velocity_fn(x_t, t, **uncond_conditioning)
        guided = uncond + scale * (cond - uncond)
        if self.guidance_rescale > 0.0:
            guided = rescale_noise_scale(cond, uncond, guided, self.guidance_rescale)
        return guided


def rescale_noise_scale(
    cond: torch.Tensor, uncond: torch.Tensor, guided: torch.Tensor, factor: float
) -> torch.Tensor:
    """Apply φ-rescaling to keep guided outputs in-distribution.

    Standard deviation of the guided prediction is shrunk toward the
    conditional prediction's scale:
    ``guided' = factor * (guided * σ_cond / σ_guided) + (1 - factor) * guided``.

    Parameters:
        cond: Conditional velocity.
        uncond: Unconditional velocity.
        guided: Naively guided velocity.
        factor: Blend in [0, 1].

    Returns:
        The rescaled velocity.
    """
    del uncond
    std_cond = cond.flatten(1).std(dim=1, keepdim=True).unsqueeze(-1)
    std_guided = guided.flatten(1).std(dim=1, keepdim=True).unsqueeze(-1)
    scale = std_cond / std_guided.clamp(min=1e-8)
    rescaled = guided * scale
    return factor * rescaled + (1.0 - factor) * guided


def sample_t(
    batch: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    mode: str = "uniform",
    shift: float = 1.0,
) -> torch.Tensor:
    """Draw flow times t ~ U(0, 1) (optionally resolution-shifted for training).

    Parameters:
        batch: Batch size.
        device: Target device.
        generator: Optional seeded generator.
        mode: `uniform` (rectified flow default).
        shift: Unused for uniform; reserved for shifted training schedules.

    Returns:
        [batch] float32 times in [0, 1].
    """
    del shift
    if mode != "uniform":
        raise ValueError(f"unsupported sampling mode {mode!r}")
    return torch.rand(batch, device=device, generator=generator, dtype=torch.float32)


def flow_matching_loss(
    velocity_fn_out: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    """Compute the rectified-flow MSE loss.

    Parameters:
        velocity_fn_out: Predicted velocity [B, T, C].
        x0: Noise samples.
        x1: Data latents.

    Returns:
        Scalar MSE between prediction and (x1 - x0).
    """
    target = x1 - x0
    return torch.nn.functional.mse_loss(velocity_fn_out.float(), target.float())


def conditional_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Return the analytic target velocity between noise and data.

    Parameters:
        x0: Noise.
        x1: Data.

    Returns:
        x1 - x0.
    """
    return x1 - x0


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate the linear interpolation path x_t = (1-t) x0 + t x1.

    Parameters:
        x0: Noise.
        x1: Data.
        t: [B] times.

    Returns:
        Interpolated latents [B, T, C].
    """
    t_ = t.reshape(-1, *([1] * (x0.ndim - 1))).to(x0.dtype)
    return (1.0 - t_) * x0 + t_ * x1
