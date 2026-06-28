"""Render model outputs cho 1 bộ compound eval groups (no-bbox, seed cố định).
Tên file chứa các từ theo thứ tự để chấm tay.

Usage:
  python3 render_compound_eval.py --groups <groups.json> --lora_ckpt <infer.safetensors>
      --out_dir <dir> [--seed 7000]
"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "DiffSynth-Studio"))

from diffsynth.core import ModelConfig
from diffsynth.pipelines.ideogram4 import Ideogram4Pipeline
from src.peft.hybrid_peft import inject_lora_into_dit, load_lora_checkpoint
from src.pipeline.infer import build_no_bbox_prompt

TARGETS = ("attention.qkv","attention.o","feed_forward.w1","feed_forward.w2","feed_forward.w3","adaln_modulation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--lora_ckpt", required=True)
    ap.add_argument("--model_dir", default="models/ideogram-ai/ideogram-4-fp8")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--rank", type=int, default=64)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    groups = json.load(open(a.groups))

    MD = Path(a.model_dir)
    print("[load] pipeline…"); t0 = time.time()
    pipe = Ideogram4Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(path=str(MD/"transformer/diffusion_pytorch_model.safetensors"), skip_download=True),
            ModelConfig(path=str(MD/"unconditional_transformer/diffusion_pytorch_model.safetensors"), skip_download=True),
            ModelConfig(path=str(MD/"text_encoder/model.safetensors"), skip_download=True),
            ModelConfig(path=str(MD/"vae/diffusion_pytorch_model.safetensors"), skip_download=True),
        ],
        tokenizer_config=ModelConfig(path=str(MD/"tokenizer"), skip_download=True))
    print(f"  {time.time()-t0:.0f}s")
    inj = inject_lora_into_dit(pipe.dit, target_patterns=TARGETS, rank=a.rank, alpha=float(a.rank))
    miss, unexp = load_lora_checkpoint(pipe.dit, Path(a.lora_ckpt), strict=False)
    if pipe.dit_uncond is not None:
        inject_lora_into_dit(pipe.dit_uncond, target_patterns=TARGETS, rank=a.rank, alpha=float(a.rank))
        load_lora_checkpoint(pipe.dit_uncond, Path(a.lora_ckpt), strict=False)
    print(f"  injected {len(inj)} | missing={len(miss)} unexpected={len(unexp)}")

    for idx, g in enumerate(groups, 1):
        n = len(g); cols = (n + 1) // 2
        prompt = build_no_bbox_prompt(" ".join(g), f"cols{cols}")
        img = pipe(prompt=prompt, negative_prompt="", cfg_scale=7.0,
                   height=1024, width=1024, num_inference_steps=48, seed=a.seed)
        safe = "-".join(g)
        fn = out / f"{idx:02d}_n{n}_{safe}_s{a.seed}.jpg"
        img.save(fn, quality=92)
        print(f"  [{idx:02d}/{len(groups)}] n={n} {' '.join(g)}")
    print(f"[DONE] {len(groups)} imgs -> {out}")


if __name__ == "__main__":
    main()
