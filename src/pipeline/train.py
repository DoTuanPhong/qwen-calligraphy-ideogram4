"""V10 Phase 2 Step 4b — Full LoRA trainer for Ideogram4.

Production trainer scaling smoke_5step_trainer.py to full N-step runs with:
  - argparse CLI
  - LR schedule (cosine + warmup, V9.2 conventions)
  - Checkpoint save (LoRA-only) + optimizer state save (resume support)
  - Auto-detect latest checkpoint for resume
  - Optional bitsandbytes 8-bit AdamW (--use_8bit_adam)
  - Grad norm logging + tqdm + per-step JSON event log
  - Outbid recovery hooks (V9.2 pattern: ckpt + state every N steps)

Critical lessons applied (peer review §13 + memory [[v10_phase2_step3_findings]]):
  - VAE encode via centralized `vae_encode_mu_only()` helper (NEVER call pipe.vae_encoder.encode)
  - Loss sign + slice via `dit_predict_image_velocity()` (-out[:, max_text_tokens:])
  - Scheduler API: add_noise + training_target + training_weight
  - batch_size=1 hardcoded (Ideogram4 pipeline text_z_padding limit)
  - LoRA target: attention.{qkv, o} only (DiT, not text encoder)
  - Resume support from day 1 (V9.2 outbid lesson)
  - Grad ckpt opt-in via env IDEOGRAM4_GRAD_CKPT (default OFF, ON for A100 40GB)

Usage:
    # Smoke 50 steps
    python3 train_ideogram4_v10.py --max_steps 50 --save_steps 50 \\
        --output_dir experiments/checkpoints/v10_smoke50

    # Full baseline 14000 steps
    python3 train_ideogram4_v10.py --max_steps 14000 --save_steps 500 \\
        --output_dir experiments/checkpoints/v10_baseline \\
        --use_8bit_adam

    # Resume from latest
    python3 train_ideogram4_v10.py --max_steps 14000 \\
        --output_dir experiments/checkpoints/v10_baseline \\
        --resume auto
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSynth-Studio"))

from diffsynth.core import ModelConfig
from diffsynth.pipelines.ideogram4 import Ideogram4Pipeline
from diffsynth.diffusion import FlowMatchScheduler

from src.peft.hybrid_peft import (
    inject_lora_into_dit,
    freeze_non_lora,
    collect_lora_params,
    count_lora_params,
    enable_gradient_checkpointing,
    save_lora_checkpoint,
    load_lora_checkpoint,
)
from src.dataset.loader import V10IdeogramDataset
from src.loss.diacritic_loss import DiacriticMaskedFlowMatchLoss
from src.peft.utils import (
    vae_encode_mu_only,
    prepare_real_inputs,
    build_dit_input_sequence,
    dit_predict_image_velocity,
)

DEFAULT_MODEL_DIR = "models/ideogram-ai/ideogram-4-fp8"
DEFAULT_DATASET_METADATA = "data/coverage_v10/metadata.jsonl"
DEFAULT_IMAGE_BASE = "data/coverage_v10"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────


def find_latest_checkpoint(output_dir: Path) -> tuple[Path, int] | None:
    """Find latest step-N.safetensors in output_dir. Returns (path, step) or None."""
    if not output_dir.exists():
        return None
    candidates = []
    pattern = re.compile(r"step-(\d+)\.safetensors$")
    for f in output_dir.glob("step-*.safetensors"):
        m = pattern.match(f.name)
        if m:
            candidates.append((int(m.group(1)), f))
    if not candidates:
        return None
    candidates.sort()
    step, path = candidates[-1]
    return path, step


def save_full_checkpoint(
    output_dir: Path,
    dit,
    optimizer,
    scheduler_state: dict,
    step: int,
    trainer_state: dict,
) -> dict:
    """Save LoRA-only weights + optimizer state + trainer state.

    Layout (mirrors V9.2):
        output_dir/
            step-N.safetensors           # LoRA only (~115 MB)
            step-N_state/
                optimizer.pt
                rng_state.pt
            trainer_state.json           # latest step + config
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"step-{step}.safetensors"
    state_dir = output_dir / f"step-{step}_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    lora_bytes = save_lora_checkpoint(dit, ckpt_path)
    torch.save(optimizer.state_dict(), state_dir / "optimizer.pt")
    torch.save(
        {
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        state_dir / "rng_state.pt",
    )

    trainer_state_path = output_dir / "trainer_state.json"
    trainer_state["last_step"] = step
    with trainer_state_path.open("w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2, ensure_ascii=False)

    return {
        "ckpt": str(ckpt_path),
        "state_dir": str(state_dir),
        "lora_bytes": lora_bytes,
    }


def load_full_checkpoint(
    output_dir: Path,
    ckpt_path: Path,
    dit,
    optimizer,
) -> int:
    """Load LoRA weights + optimizer state from a checkpoint pair.

    Returns the resumed step (parsed from ckpt filename).
    """
    m = re.match(r"step-(\d+)\.safetensors$", ckpt_path.name)
    if not m:
        raise ValueError(f"checkpoint filename must match step-N.safetensors, got {ckpt_path.name}")
    step = int(m.group(1))

    load_lora_checkpoint(dit, ckpt_path, strict=True)

    state_dir = output_dir / f"step-{step}_state"
    if state_dir.exists():
        optimizer.load_state_dict(torch.load(state_dir / "optimizer.pt", weights_only=False))
        rng = torch.load(state_dir / "rng_state.pt", weights_only=False)
        torch.set_rng_state(rng["torch_rng"])
        torch.cuda.set_rng_state_all(rng["cuda_rng"])
        print(f"[resume] loaded optimizer state from {state_dir}")
    else:
        print(f"[resume] WARNING: no state_dir at {state_dir}, optimizer fresh")
    print(f"[resume] LoRA weights from {ckpt_path} → resumed at step {step}")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler (cosine + warmup)
# ─────────────────────────────────────────────────────────────────────────────


def get_lr_multiplier(
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.10,
) -> float:
    """Cosine decay with linear warmup. Returns multiplier in [min_lr_ratio, 1.0]."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser(description="V10 Ideogram4 LoRA trainer")
    # Paths
    ap.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--metadata_path", type=Path, default=DEFAULT_DATASET_METADATA)
    ap.add_argument("--image_base_dir", type=Path, default=DEFAULT_IMAGE_BASE)
    ap.add_argument("--mask_base_dir", type=Path, default=None,
                    help="default: <image_base_dir>/masks")
    ap.add_argument("--output_dir", type=Path, required=True)
    # LoRA
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=float, default=64.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_targets", nargs="+", default=["attention.qkv", "attention.o"])
    # Training
    ap.add_argument("--max_steps", type=int, required=True)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--min_lr_ratio", type=float, default=0.10)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--use_8bit_adam", action="store_true",
                    help="Use bitsandbytes 8-bit AdamW (smoke-test first via --max_steps 20)")
    # Loss
    ap.add_argument("--diacritic_factor", type=float, default=10.0)
    ap.add_argument("--bg_weight", type=float, default=1.0)
    ap.add_argument("--use_scheduler_weight", action="store_true",
                    help="Apply scheduler.training_weight(t) — default OFF for simplicity")
    # Resume
    ap.add_argument("--resume", type=str, default=None,
                    help="'auto' to find latest in output_dir, or path to step-N.safetensors")
    ap.add_argument("--init_lora_from", type=Path, default=None,
                    help="FORK mode: load LoRA weights từ ckpt nhưng start step 0 với "
                         "fresh optimizer (dùng cho fork experiment từ baseline, e.g. "
                         "T1b tone-zone mask fork từ step-14177). Mutually exclusive với --resume.")
    ap.add_argument("--diacritic_factor_anneal_steps", type=int, default=0,
                    help="Anneal diacritic_factor từ 1.0 → target qua N steps đầu "
                         "(0 = off). Per deep research Q5: anneal UP tránh shock "
                         "gradient variance trên vùng nhỏ ở batch nhỏ.")
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--dataset_shuffle", action="store_true",
                    help="Shuffle dataset order (default: in-order for reproducibility)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}\nV10 Phase 2 LoRA Trainer\n{'='*72}")
    print(f"max_steps={args.max_steps}, save_steps={args.save_steps}")
    print(f"lr={args.lr}, warmup={args.warmup_steps}, min_lr_ratio={args.min_lr_ratio}")
    print(f"rank={args.lora_rank}, alpha={args.lora_alpha}, targets={args.lora_targets}")
    print(f"grad_accum={args.gradient_accumulation_steps}, "
          f"effective batch={args.gradient_accumulation_steps}")
    print(f"output_dir={args.output_dir}")
    print(f"use_8bit_adam={args.use_8bit_adam}")
    print(f"IDEOGRAM4_GRAD_CKPT={os.environ.get('IDEOGRAM4_GRAD_CKPT', '<unset>')}")

    # ── Load pipeline ───────────────────────────────────────────────────────
    print(f"\n[1] Loading pipeline...")
    t0 = time.time()
    pipe = Ideogram4Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(args.model_dir / "transformer" / "diffusion_pytorch_model.safetensors"), skip_download=True),
            ModelConfig(path=str(args.model_dir / "text_encoder" / "model.safetensors"), skip_download=True),
            ModelConfig(path=str(args.model_dir / "vae" / "diffusion_pytorch_model.safetensors"), skip_download=True),
        ],
        tokenizer_config=ModelConfig(path=str(args.model_dir / "tokenizer"), skip_download=True),
    )
    print(f"    loaded in {time.time()-t0:.1f}s, free VRAM {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # ── Inject LoRA + freeze ────────────────────────────────────────────────
    print(f"\n[2] Injecting LoRA...")
    replaced = inject_lora_into_dit(
        pipe.dit,
        target_patterns=tuple(args.lora_targets),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    n_train, n_frozen = freeze_non_lora(pipe.dit)
    counts = count_lora_params(pipe.dit)
    print(f"    {len(replaced)} modules wrapped, {counts['n_total_params']/1e6:.1f}M LoRA params")

    # FORK mode: load LoRA weights, step 0, fresh optimizer
    if args.init_lora_from is not None:
        if args.resume is not None:
            raise SystemExit("--init_lora_from và --resume mutually exclusive")
        if not args.init_lora_from.exists():
            raise FileNotFoundError(f"--init_lora_from {args.init_lora_from} not found")
        missing, unexpected = load_lora_checkpoint(pipe.dit, args.init_lora_from, strict=True)
        print(f"    [fork] LoRA weights init từ {args.init_lora_from.name} "
              f"(missing={len(missing)}, unexpected={len(unexpected)}), step 0 + fresh optimizer")

    n_ckpt = enable_gradient_checkpointing(pipe.dit)
    if n_ckpt > 0:
        print(f"    gradient checkpointing: {n_ckpt} layers wrapped")

    # ── Scheduler ───────────────────────────────────────────────────────────
    print(f"\n[3] Setting up FlowMatchScheduler in training mode...")
    scheduler = FlowMatchScheduler("Ideogram4")
    scheduler.set_timesteps(
        num_inference_steps=1000,
        training=True,
        image_resolution=(args.height, args.width),
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    print(f"\n[4] Loading dataset...")
    dataset = V10IdeogramDataset(
        metadata_path=args.metadata_path,
        image_base_dir=args.image_base_dir,
        mask_base_dir=args.mask_base_dir,
        image_size=args.height,
        return_pil=True,
    )
    print(f"    {len(dataset)} records")

    # ── Loss ────────────────────────────────────────────────────────────────
    loss_fn = DiacriticMaskedFlowMatchLoss(
        diacritic_factor=args.diacritic_factor,
        bg_weight=args.bg_weight,
        downsample_mode="max",
    )

    # ── Optimizer ───────────────────────────────────────────────────────────
    print(f"\n[5] Setting up optimizer...")
    lora_param_list = [p for _, p in collect_lora_params(pipe.dit)]
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                lora_param_list, lr=args.lr, weight_decay=args.weight_decay,
            )
            print(f"    bitsandbytes AdamW8bit, {len(lora_param_list)} params")
        except ImportError:
            print(f"    WARNING: bitsandbytes not installed, falling back to bf16 AdamW")
            optimizer = torch.optim.AdamW(
                lora_param_list, lr=args.lr, weight_decay=args.weight_decay,
            )
    else:
        optimizer = torch.optim.AdamW(
            lora_param_list, lr=args.lr, weight_decay=args.weight_decay,
        )
        print(f"    bf16 AdamW, {len(lora_param_list)} params")

    # ── Resume ──────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume is not None:
        if args.resume == "auto":
            latest = find_latest_checkpoint(args.output_dir)
            if latest is None:
                print(f"[resume] no checkpoint found in {args.output_dir}, starting fresh")
            else:
                ckpt_path, _ = latest
                start_step = load_full_checkpoint(args.output_dir, ckpt_path, pipe.dit, optimizer)
        else:
            ckpt_path = Path(args.resume)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"--resume {ckpt_path} does not exist")
            start_step = load_full_checkpoint(args.output_dir, ckpt_path, pipe.dit, optimizer)

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n[6] Starting training from step {start_step}/{args.max_steps}")
    pipe.dit.train()

    trainer_state = {
        "config": vars(args),
        "last_step": start_step,
    }
    trainer_state["config"] = {k: str(v) if isinstance(v, Path) else v
                              for k, v in trainer_state["config"].items()}

    log_path = args.output_dir / "train_log.jsonl"
    log_f = log_path.open("a", encoding="utf-8")

    t_train_start = time.time()
    last_log_t = t_train_start
    optimizer.zero_grad()

    try:
        for step in range(start_step, args.max_steps):
            # Sample idx (in-order or shuffled)
            if args.dataset_shuffle:
                idx = torch.randint(0, len(dataset), (1,)).item()
            else:
                idx = step % len(dataset)
            sample = dataset[idx]

            # Build inputs
            inputs = prepare_real_inputs(
                pipe, sample["prompt"], height=args.height, width=args.width,
                no_grad_text_encoder=True,
            )
            z = vae_encode_mu_only(pipe, sample["image"], inputs["grid_h"], inputs["grid_w"]).to(torch.float32)

            # Sample timestep + noise
            timestep_id = torch.randint(0, len(scheduler.timesteps), (1,)).item()
            timestep = scheduler.timesteps[timestep_id]
            timestep_t = timestep.to(z.device, torch.float32).unsqueeze(0)
            noise = torch.randn_like(z)
            x_t_img = scheduler.add_noise(z, noise, timestep)
            target = scheduler.training_target(z, noise, timestep)
            if args.use_scheduler_weight:
                training_weight = scheduler.training_weight(timestep).to(z.device, z.dtype)
            else:
                training_weight = 1.0

            # Build full sequence + forward
            x_full = build_dit_input_sequence(x_t_img, inputs["max_text_tokens"])
            pred = dit_predict_image_velocity(
                pipe.dit,
                llm_features=inputs["llm_features"],
                x_full=x_full,
                timestep=timestep_t,
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
                max_text_tokens=inputs["max_text_tokens"],
            )

            # Anneal diacritic_factor 1.0 → target (deep research Q5: ramp UP
            # tránh gradient-variance shock trên vùng nhỏ ở batch nhỏ)
            if args.diacritic_factor_anneal_steps > 0:
                frac = min(1.0, step / args.diacritic_factor_anneal_steps)
                loss_fn.diacritic_factor = 1.0 + (args.diacritic_factor - 1.0) * frac

            # Build mask tensor + loss
            mask_arr = (TF.to_tensor(sample["mask"]).unsqueeze(0) > 0.5).float().to(z.device)
            loss_dict = loss_fn(
                pred=pred, target=target,
                mask_full=mask_arr,
                grid_h=inputs["grid_h"], grid_w=inputs["grid_w"],
                training_weight=training_weight,
            )
            loss = loss_dict["loss"] / args.gradient_accumulation_steps
            loss.backward()

            # Step optimizer every grad_accum
            do_step = ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == args.max_steps)
            grad_norm = None
            if do_step:
                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        lora_param_list, max_norm=args.grad_clip,
                    ).item()
                # LR schedule
                lr_mult = get_lr_multiplier(step, args.warmup_steps, args.max_steps, args.min_lr_ratio)
                current_lr = args.lr * lr_mult
                for g in optimizer.param_groups:
                    g["lr"] = current_lr
                optimizer.step()
                optimizer.zero_grad()
            else:
                current_lr = args.lr * get_lr_multiplier(
                    step, args.warmup_steps, args.max_steps, args.min_lr_ratio,
                )

            # NaN check
            loss_val = loss_dict["loss"].item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f"[ABORT] NaN/Inf loss at step {step}: {loss_val}")
                save_full_checkpoint(
                    args.output_dir, pipe.dit, optimizer,
                    scheduler_state={}, step=step,
                    trainer_state=trainer_state,
                )
                sys.exit(1)

            # Log
            if step % args.log_every == 0 or step + 1 == args.max_steps:
                now = time.time()
                step_time = (now - last_log_t) / max(1, args.log_every)
                last_log_t = now
                vram_peak = torch.cuda.max_memory_allocated() / 1e9
                event = {
                    "step": step,
                    "word": sample["content"],
                    "timestep": float(timestep.item()),
                    "loss": loss_val,
                    "loss_diac": loss_dict["loss_diacritic"].item(),
                    "loss_bg": loss_dict["loss_background"].item(),
                    "diac_frac": loss_dict["diacritic_frac"],
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "step_time_s": step_time,
                    "vram_gb": vram_peak,
                }
                log_f.write(json.dumps(event, ensure_ascii=False) + "\n")
                log_f.flush()
                print(
                    f"step {step:>6d}/{args.max_steps}  word={sample['content']:10s}  "
                    f"t={timestep.item():.3f}  loss={loss_val:.4e}  "
                    f"lr={current_lr:.2e}  "
                    f"{'gn='+format(grad_norm, '.2e')+' ' if grad_norm else '          '}"
                    f"{step_time:.1f}s/step  vram={vram_peak:.1f}GB"
                )

            # Save checkpoint
            if (step + 1) % args.save_steps == 0 or (step + 1 == args.max_steps):
                saved = save_full_checkpoint(
                    args.output_dir, pipe.dit, optimizer,
                    scheduler_state={}, step=step + 1,
                    trainer_state=trainer_state,
                )
                print(f"    [ckpt] saved {saved['ckpt']} ({saved['lora_bytes']/1e6:.1f} MB)")

        # Training done
        total_h = (time.time() - t_train_start) / 3600
        print(f"\n[DONE] Trained {args.max_steps - start_step} steps in {total_h:.2f}h")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
