"""V10 Phase 2 Step 1 — LoRA wrapper for Ideogram4 FP8 modules.

DiffSynth's `Fp8Linear` stores weight as torch.float8_e4m3fn **buffer** (not
nn.Parameter), with per-row `weight_scale` buffer. Forward does dequant + matmul:

    w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
    y = F.linear(x, w, bias)

Because weight is a buffer, PyTorch autograd never tracks gradient on it — it
stays frozen automatically. We wrap an existing `Fp8Linear` instance with bf16
LoRA factors that ARE trainable parameters.

Design constraints (verified from DiffSynth source — see docs/v10_phase2_lora_design.vi.md §1):

  - 211 Fp8Linear modules total trong DiT, 13 unique patterns
  - LoRA target: `layers.*.attention.{qkv, o}` only (68 modules, ~60M params)
  - Skip text encoder (Qwen3-VL tone-collapse §11)
  - Skip FFN, AdaLN, embedding/head Linear modules
  - rsLoRA scaling alpha / sqrt(rank) (V9.2 proven sweet spot rank=64, alpha=64)
  - Standard init: Kaiming uniform A, zero B → identity at init

Smoke test (run_smoke_lora_inject.py) verifies:
  1. Injected modules = 68 (34 blocks × {qkv, o})
  2. Trainable params ≈ 60M
  3. Base output before/after injection identical at init (lora_B=0)
  4. LoRA grads non-zero after backward
  5. Base FP8 buffers no grad
  6. Checkpoint LoRA-only (~120MB), no FP8 base weights
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default LoRA targets per Phase 1 finding (Probe 4 hidden-state collapse → DiT only)
DEFAULT_LORA_TARGETS = ("attention.qkv", "attention.o")


class Fp8LoRALinear(nn.Module):
    """LoRA wrapper around an Fp8Linear base.

    Forward:
        y = base(x) + (alpha / sqrt(rank)) * (lora_B @ lora_A @ x)

    The base is kept as a submodule. Its FP8 weight + scale buffers stay
    frozen because they are not nn.Parameter (Fp8Linear uses register_buffer).
    LoRA A/B are bf16 nn.Parameter — trainable.

    Initialization:
        lora_A: Kaiming uniform (default a=sqrt(5) per nn.Linear default)
        lora_B: zeros → wrapper output identical to base at init
    """

    def __init__(
        self,
        base: nn.Module,  # Fp8Linear, but kept loose to avoid import cycle
        rank: int = 64,
        alpha: float = 64.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError(f"base must expose in_features/out_features, got {type(base)}")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        # rsLoRA scaling
        self.scaling = self.alpha / math.sqrt(self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Trainable bf16 factors
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=dtype))
        # Standard LoRA init: Kaiming A, zero B → identity at construction
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen FP8 dequant path (gradient cannot flow to buffer weight/scale)
        base_out = self.base(x)
        # LoRA delta. Cast factors to x.dtype to avoid mixed-precision linear errors.
        a_t = self.lora_A.to(x.dtype)
        b_t = self.lora_B.to(x.dtype)
        delta = F.linear(F.linear(self.dropout(x), a_t), b_t)
        return base_out + delta * self.scaling

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}")


def _is_fp8_linear(module: nn.Module) -> bool:
    """Detect Fp8Linear without import (avoid cyclic dependency on DiffSynth)."""
    return (
        type(module).__name__ == "Fp8Linear"
        and hasattr(module, "weight")
        and hasattr(module, "weight_scale")
    )


def inject_lora_into_dit(
    dit: nn.Module,
    target_patterns: Iterable[str] = DEFAULT_LORA_TARGETS,
    rank: int = 64,
    alpha: float = 64.0,
    dropout: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
) -> list[tuple[str, Fp8LoRALinear]]:
    """Walk DiT, wrap matched Fp8Linear modules in Fp8LoRALinear.

    Matching is substring-based on the dotted module path (matches V9.2 `hybrid_peft`
    pattern). Default targets: `attention.qkv`, `attention.o`.

    Returns list of (full_module_name, wrapper) so caller can log/persist.
    Modifies dit in place.
    """
    targets = tuple(target_patterns)
    to_replace: list[tuple[str, str, nn.Module]] = []
    # Collect first to avoid mutating during named_modules iteration
    for name, module in dit.named_modules():
        if not _is_fp8_linear(module):
            continue
        if not any(p in name for p in targets):
            continue
        to_replace.append((name, name.rsplit(".", 1)[-1] if "." in name else name, module))

    replaced: list[tuple[str, Fp8LoRALinear]] = []
    for full_name, _attr, base in to_replace:
        parent_path, _, attr = full_name.rpartition(".")
        parent = dit.get_submodule(parent_path) if parent_path else dit
        wrapper = Fp8LoRALinear(
            base=base,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=dtype,
        )
        # Move wrapper to base's device. Fp8Linear stores tensors as buffers
        # (not Parameters), so list(base.parameters()) is empty — must check
        # buffers explicitly.
        device = None
        for buf in base.buffers():
            device = buf.device
            break
        if device is None:
            for p in base.parameters():
                device = p.device
                break
        if device is not None:
            wrapper = wrapper.to(device)
        setattr(parent, attr, wrapper)
        replaced.append((full_name, wrapper))
    return replaced


def freeze_non_lora(dit: nn.Module) -> tuple[int, int]:
    """Set requires_grad=False everywhere except LoRA params.

    Returns (n_trainable, n_frozen) counts.
    """
    n_trainable = 0
    n_frozen = 0
    for name, p in dit.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        p.requires_grad_(is_lora)
        if is_lora:
            n_trainable += 1
        else:
            n_frozen += 1
    return n_trainable, n_frozen


def collect_lora_params(dit: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return list of (name, param) for LoRA params (lora_A / lora_B)."""
    return [
        (name, p)
        for name, p in dit.named_parameters()
        if ("lora_A" in name) or ("lora_B" in name)
    ]


def count_lora_params(dit: nn.Module) -> dict:
    """Return aggregate counts: total params, breakdown lora_A / lora_B."""
    n_total = 0
    n_a = 0
    n_b = 0
    n_modules = 0
    seen_modules = set()
    for name, p in dit.named_parameters():
        if "lora_A" in name:
            n_a += p.numel()
            n_total += p.numel()
            seen_modules.add(name.rsplit(".lora_A", 1)[0])
        elif "lora_B" in name:
            n_b += p.numel()
            n_total += p.numel()
            seen_modules.add(name.rsplit(".lora_B", 1)[0])
    n_modules = len(seen_modules)
    return {
        "n_lora_modules": n_modules,
        "n_total_params": n_total,
        "n_lora_A_params": n_a,
        "n_lora_B_params": n_b,
    }


def lora_state_dict(dit: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA-only state dict for checkpoint save.

    Includes lora_A.weight and lora_B.weight under their full module paths.
    Excludes base FP8 weights/scales/biases.
    """
    sd = {}
    for name, p in dit.named_parameters():
        if ("lora_A" in name) or ("lora_B" in name):
            sd[name] = p.detach().cpu().clone()
    return sd


def load_lora_state_dict(
    dit: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load LoRA-only state dict into dit.

    Returns (missing_keys, unexpected_keys). When strict=True, raises if either
    list is non-empty.
    """
    dit_params = dict(dit.named_parameters())
    missing = []
    unexpected = []
    loaded = 0
    for key, val in state_dict.items():
        if key not in dit_params:
            unexpected.append(key)
            continue
        param = dit_params[key]
        if param.shape != val.shape:
            raise ValueError(
                f"Shape mismatch for {key}: model {tuple(param.shape)} vs ckpt {tuple(val.shape)}"
            )
        with torch.no_grad():
            param.copy_(val.to(param.device, param.dtype))
        loaded += 1
    # Check for missing keys (LoRA params in model that weren't in state dict)
    for name in dit_params:
        if ("lora_A" in name or "lora_B" in name) and name not in state_dict:
            missing.append(name)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"strict=True but missing={len(missing)}, unexpected={len(unexpected)}. "
            f"First missing: {missing[:3]}, first unexpected: {unexpected[:3]}"
        )
    return missing, unexpected


def enable_gradient_checkpointing(dit: nn.Module) -> int:
    """Monkey-patch DiT layers with torch.utils.checkpoint — full hoặc selective.

    Ideogram4DiT doesn't have grad ckpt support upstream (commit 6d103c0+). For
    training on smaller GPUs we checkpoint transformer block forward passes to
    cut activation memory (~1 GB/layer trên 1024×1024, B=1).

    Env var IDEOGRAM4_GRAD_CKPT:
      - unset / "" / "0"            → no-op, full activations (~64 GB peak)
      - "1" / "true" / "yes" / "on" → checkpoint ALL 34 layers (~30 GB peak)
      - "<K>" (integer 2..n_layers) → SELECTIVE: checkpoint K layers evenly
        spaced (e.g. "20" trên L40S 46 GB → peak ≈ 64-20 ≈ 44 GB, ~1-2 GB
        headroom, recompute overhead chỉ ~K/34 thay vì full)

    Returns number of layers wrapped (0 if disabled).
    """
    import os
    flag = os.environ.get("IDEOGRAM4_GRAD_CKPT", "").strip().lower()
    n_layers = len(dit.layers)
    if flag in {"", "0", "false", "no", "off"}:
        return 0
    if flag in {"1", "true", "yes", "on"}:
        ckpt_indices = set(range(n_layers))  # full
    else:
        try:
            k = int(flag)
        except ValueError:
            raise ValueError(
                f"IDEOGRAM4_GRAD_CKPT={flag!r} invalid: expect 0/1/true/false "
                f"or integer 2..{n_layers} (số layers checkpoint selective)")
        if not (2 <= k <= n_layers):
            raise ValueError(
                f"IDEOGRAM4_GRAD_CKPT={k} out of range: expect 2..{n_layers}")
        # Evenly-spaced selection across depth (linspace rounding)
        ckpt_indices = {round(i * (n_layers - 1) / (k - 1)) for i in range(k)}
        # Rounding collisions có thể make len < k → fill từ các index chưa chọn
        if len(ckpt_indices) < k:
            for idx in range(n_layers):
                if len(ckpt_indices) >= k:
                    break
                ckpt_indices.add(idx)

    if getattr(dit, "_grad_ckpt_enabled", False):
        # Already wrapped in a previous call. Report what was ACTUALLY wrapped —
        # never the env-derived set computed above, which may differ if the env
        # var changed between calls (re-wrapping mid-process is not supported).
        prev = dit._grad_ckpt_indices  # AttributeError here = invariant broken
        if set(prev) != ckpt_indices:
            import warnings
            warnings.warn(
                f"IDEOGRAM4_GRAD_CKPT changed after layers were wrapped "
                f"(wrapped {len(prev)} layers, env now requests {len(ckpt_indices)}); "
                f"keeping original wrapping. Restart process to change selection.")
        return len(prev)

    from torch.utils.checkpoint import checkpoint

    n_wrapped = 0
    for i, layer in enumerate(dit.layers):
        if i not in ckpt_indices or getattr(layer, "_ckpt_wrapped", False):
            continue
        original_layer_forward = layer.forward

        def make_ckpt_forward(orig_fwd, layer_ref):
            # Positional signature mirrors Ideogram4DiT block forward at
            # DiffSynth commit 6d103c0 (x, segment_ids, cos, sin, adaln_input).
            # If upstream changes the block signature, update here in sync —
            # a mismatch fails loudly at first training forward.
            def ckpt_forward(x, segment_ids, cos, sin, adaln_input):
                if torch.is_grad_enabled() and layer_ref.training:
                    return checkpoint(
                        orig_fwd, x, segment_ids, cos, sin, adaln_input,
                        use_reentrant=False,
                    )
                return orig_fwd(x, segment_ids, cos, sin, adaln_input)
            return ckpt_forward

        layer.forward = make_ckpt_forward(original_layer_forward, layer)
        layer._ckpt_wrapped = True
        layer._ckpt_orig_forward = original_layer_forward  # for disable/restore
        n_wrapped += 1

    dit._grad_ckpt_enabled = True
    dit._grad_ckpt_indices = sorted(ckpt_indices)
    return n_wrapped


def disable_gradient_checkpointing(dit: nn.Module) -> int:
    """Unwrap layers wrapped by enable_gradient_checkpointing().

    Restores each layer's original forward. Returns number of layers unwrapped
    (0 if nothing was wrapped). Safe to call multiple times.
    """
    if not getattr(dit, "_grad_ckpt_enabled", False):
        return 0
    n_unwrapped = 0
    for layer in dit.layers:
        if getattr(layer, "_ckpt_wrapped", False):
            layer.forward = layer._ckpt_orig_forward
            del layer._ckpt_orig_forward
            layer._ckpt_wrapped = False
            n_unwrapped += 1
    dit._grad_ckpt_enabled = False
    dit._grad_ckpt_indices = []
    return n_unwrapped


def save_lora_checkpoint(dit: nn.Module, path: str | Path) -> int:
    """Save LoRA-only state dict (safetensors format).

    Returns bytes written.
    """
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = lora_state_dict(dit)
    save_file(sd, str(path))
    return path.stat().st_size


def load_lora_checkpoint(dit: nn.Module, path: str | Path, strict: bool = True) -> tuple[list[str], list[str]]:
    """Load LoRA-only state dict from safetensors file."""
    from safetensors.torch import load_file
    sd = load_file(str(path))
    return load_lora_state_dict(dit, sd, strict=strict)
