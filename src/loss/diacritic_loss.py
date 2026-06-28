"""V10 Phase 2 Step 3c — Diacritic-masked flow-matching loss for Ideogram4.

Ports the principle of V9.2 `DiacriticMaskedFlowMatchSFTLoss` to Ideogram4's
sequence-shaped latent. Differences from V9.2 (per peer review §13.4):

  - V9.2 latent: (B, C, H, W) image grid  →  mask was (B, 1, H, W)
  - Ideogram4 latent: (B, L=4096, 128) sequence  →  mask must be (B, L, 1)

Pipeline:
  1. Downsample binary mask 1024×1024 → 64×64 patch grid (max-pool)
  2. Flatten + transpose: (B, 1, 64, 64) → (B, 4096, 1)
  3. Per-token weight: w = mask * diacritic_factor + (1-mask) * bg_weight
  4. Element-wise MSE on (pred, target), shape (B, 4096, 128)
  5. weighted = w (broadcast over channel dim) * mse
  6. Optional: scheduler training_weight (timestep weighting) applied as scalar mul
  7. Mean reduce over all dims → scalar loss

Args (mirrors V9.2 defaults):
    diacritic_factor: 10.0 (boost diacritic regions)
    bg_weight: 1.0 (baseline)
    use_scheduler_weight: True (apply scheduler.training_weight(t))
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample_mask_to_patches(
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    mode: str = "max",
) -> torch.Tensor:
    """Downsample (B, 1, H, W) binary mask to (B, grid_h, grid_w) per-patch values.

    Mode 'max' is preferred: any diacritic pixel inside the patch → patch counts
    as diacritic. Mode 'mean' gives fractional coverage (use if soft weighting
    desired).
    """
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(
            f"mask must be (B, 1, H, W), got {tuple(mask.shape)}"
        )
    if mode == "max":
        out = F.adaptive_max_pool2d(mask, (grid_h, grid_w))
    elif mode == "mean":
        out = F.adaptive_avg_pool2d(mask, (grid_h, grid_w))
    else:
        raise ValueError(f"mode must be 'max' or 'mean', got {mode!r}")
    return out.squeeze(1)  # (B, grid_h, grid_w)


def diacritic_masked_flow_match_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_patches: torch.Tensor,
    diacritic_factor: float = 10.0,
    bg_weight: float = 1.0,
    training_weight: float | torch.Tensor = 1.0,
) -> dict:
    """Masked flow-match MSE loss on Ideogram4 sequence latent.

    Args:
        pred: (B, L, C) — image-token predictions after -out[:, max_text_tokens:] slice
        target: (B, L, C) — flow-match target = noise - z (from scheduler.training_target)
        mask_patches: (B, gh, gw) — downsampled diacritic mask, 0/1 binary or [0, 1] soft
        diacritic_factor: weight on diacritic-region tokens
        bg_weight: weight on non-diacritic tokens
        training_weight: scalar or (B,) — scheduler.training_weight(timestep)

    Returns dict:
        loss: scalar tensor
        loss_diacritic: scalar (mean MSE in diacritic regions, unweighted)
        loss_background: scalar (mean MSE in background, unweighted)
        diacritic_frac: float — fraction of tokens flagged as diacritic
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} ≠ target {tuple(target.shape)}")
    B, L, C = pred.shape
    gh, gw = mask_patches.shape[-2], mask_patches.shape[-1]
    if mask_patches.shape != (B, gh, gw):
        raise ValueError(f"mask_patches must be (B, gh, gw), got {tuple(mask_patches.shape)}")
    if gh * gw != L:
        raise ValueError(f"mask grid {gh}×{gw}={gh*gw} ≠ sequence length L={L}")

    # Flatten mask to (B, L, 1) for per-token weight
    mask_flat = mask_patches.reshape(B, L, 1).to(pred.dtype)
    weight = mask_flat * diacritic_factor + (1.0 - mask_flat) * bg_weight  # (B, L, 1)

    # Element-wise MSE
    mse = (pred - target).pow(2)  # (B, L, C)
    weighted_mse = mse * weight   # broadcast (B, L, 1) over channel

    # Scheduler timestep weighting (scalar or per-sample)
    if isinstance(training_weight, torch.Tensor):
        # Broadcast (B,) → (B, 1, 1)
        tw = training_weight.to(pred.dtype).view(-1, 1, 1)
    else:
        tw = float(training_weight)

    loss = (weighted_mse * tw).mean()

    # Diagnostics (no grad)
    with torch.no_grad():
        diac_mask_bool = mask_flat > 0.5  # (B, L, 1)
        bg_mask_bool = ~diac_mask_bool
        diacritic_frac = diac_mask_bool.float().mean().item()
        # Mean unweighted MSE in each region
        if diac_mask_bool.any():
            loss_diacritic = (mse * diac_mask_bool.float()).sum() / (
                diac_mask_bool.float().sum() * C
            )
        else:
            loss_diacritic = torch.tensor(0.0, device=pred.device)
        if bg_mask_bool.any():
            loss_background = (mse * bg_mask_bool.float()).sum() / (
                bg_mask_bool.float().sum() * C
            )
        else:
            loss_background = torch.tensor(0.0, device=pred.device)

    return {
        "loss": loss,
        "loss_diacritic": loss_diacritic,
        "loss_background": loss_background,
        "diacritic_frac": diacritic_frac,
    }


class DiacriticMaskedFlowMatchLoss(nn.Module):
    """Module wrapper for use in trainers."""

    def __init__(
        self,
        diacritic_factor: float = 10.0,
        bg_weight: float = 1.0,
        downsample_mode: str = "max",
    ):
        super().__init__()
        self.diacritic_factor = diacritic_factor
        self.bg_weight = bg_weight
        self.downsample_mode = downsample_mode

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_full: torch.Tensor,
        grid_h: int,
        grid_w: int,
        training_weight: float | torch.Tensor = 1.0,
    ) -> dict:
        """
        Args:
            pred, target: (B, L, C) — image-token predictions and flow target
            mask_full: (B, 1, H, W) — full-resolution binary mask
            grid_h, grid_w: patch grid dims (e.g., 64, 64 for 1024 image)
            training_weight: scalar or (B,) timestep weighting
        """
        mask_patches = downsample_mask_to_patches(
            mask_full, grid_h, grid_w, mode=self.downsample_mode
        )
        return diacritic_masked_flow_match_loss(
            pred=pred,
            target=target,
            mask_patches=mask_patches,
            diacritic_factor=self.diacritic_factor,
            bg_weight=self.bg_weight,
            training_weight=training_weight,
        )
