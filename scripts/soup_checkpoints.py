#!/usr/bin/env python3
"""Checkpoint Souping and Inference Format Converter Utility.

This script supports:
  1. Souping: Averaging two official PEFT LoRA checkpoints with a weighted ratio.
  2. Conversion: Converting an official DiffSynth-style LoRA checkpoint (standard scale)
     to our optimized rsLoRA inference format (pre-scaled weights for Fp8LoRALinear).

Usage:
    # 1. Soup two checkpoints and convert the output to inference format:
    python3 scripts/soup_checkpoints.py \
        --base path/to/gold4.safetensors \
        --new path/to/lr3e5.safetensors \
        --out path/to/soup_9to1.safetensors \
        --weight 9.0

    # 2. Convert a single checkpoint to inference format:
    python3 scripts/soup_checkpoints.py \
        --src path/to/official_lora.safetensors \
        --dst path/to/inference_lora.safetensors
"""
import argparse
import math
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file


def convert(src: str, dst: str, rank: int = 64, alpha: float = 64.0):
    official_scale = alpha / rank            # 1.0
    infer_scale = alpha / math.sqrt(rank)    # 8.0 (rsLoRA)
    factor = official_scale / infer_scale    # 0.125 -> applied to lora_B
    out = {}
    nA = nB = 0
    with safe_open(src, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            nk = k.replace(".default.weight", "")  # strip PEFT suffix
            if nk.endswith(".lora_B"):
                t = t * factor
                nB += 1
            elif nk.endswith(".lora_A"):
                nA += 1
            out[nk] = t.contiguous()
    save_file(out, dst)
    print(f"[convert] {Path(src).name} -> {Path(dst).name}")
    print(f"  scale: official {official_scale:.3f} -> infer rsLoRA {infer_scale:.3f}; lora_B x{factor:.4f}")
    print(f"  keys: {len(out)} ({nA} lora_A, {nB} lora_B scaled)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="Path to base official LoRA checkpoint")
    ap.add_argument("--new", help="Path to new official LoRA checkpoint")
    ap.add_argument("--out", help="Path to save the souped official checkpoint")
    ap.add_argument("--weight", type=float, default=9.0, help="Weight multiplier for --base (e.g. 9.0 for 90/10)")
    ap.add_argument("--src", help="Direct path to single checkpoint for conversion only")
    ap.add_argument("--dst", help="Direct output path for conversion only")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64.0)
    a = ap.parse_args()

    # Direct conversion mode
    if a.src and a.dst:
        convert(a.src, a.dst, a.rank, a.alpha)
        return

    # Checkpoint averaging mode
    if not (a.base and a.new and a.out):
        ap.error("Must specify either (--src and --dst) for conversion or (--base, --new, and --out) for souping.")

    base_path = Path(a.base)
    new_path = Path(a.new)
    out_path = Path(a.out)

    print(f"Souping: base={base_path} (weight {a.weight:g}) + new={new_path} (weight 1.0) ...")
    denom = a.weight + 1.0

    with safe_open(str(base_path), framework="pt") as bf, safe_open(str(new_path), framework="pt") as nf:
        bkeys = list(bf.keys())
        nkeys = list(nf.keys())
        if bkeys != nkeys:
            raise SystemExit(f"Key mismatch: base={len(bkeys)} keys, new={len(nkeys)} keys.")
        
        sd = {}
        for k in bkeys:
            sd[k] = ((bf.get_tensor(k).float() * a.weight + nf.get_tensor(k).float()) / denom).contiguous()

    save_file(sd, str(out_path))
    print(f"[soup] Wrote averaged official checkpoint to: {out_path}")

    # Auto-convert to inference wrapper format
    infer_out = out_path.with_name(f"{out_path.stem}_infer{out_path.suffix}")
    print(f"Auto-converting to rsLoRA inference format -> {infer_out}")
    convert(str(out_path), str(infer_out), a.rank, a.alpha)


if __name__ == "__main__":
    main()
