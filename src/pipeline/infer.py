"""V10 Phase 2 Step 4e — Inference with LoRA checkpoint.

Loads raw Ideogram4 → injects LoRA wrappers (identity at init) → loads LoRA-only
checkpoint → runs inference on either hard test cases or arbitrary prompts.

Critical: uses centralized μ-only VAE encoding ONLY if --input_image is supplied
(image-to-image). Default text-to-image path doesn't touch VAE encoder; the
pipeline's normal inference path works untouched.

Use cases:
  1. Post-smoke50 gate (peer review §4f): compare raw Ideogram4 vs step-50 LoRA on
     6 hard cases — must NOT regress quality (no gray_block, Hán-tự, broken layout)
  2. Post-baseline evaluation: compare step-14000 LoRA vs raw on full test set
  3. Cherry-pick best checkpoint by visual A/B

Prompt modes (2026-06-11):
  - DEFAULT = no-bbox Option A (single text element, `\\n` line structure) —
    verified trên step-14177 yxyx là phương án duy nhất không lỗi ký tự;
    model tự layout tự nhiên hơn bbox cứng. Multi-word phrase OK.
  - --use_bbox = legacy v3.1 per-sample bbox (position control tường minh).

Usage:
    # Hard cases (default no-bbox, multi-word phrases supported)
    python3 infer_v10_lora.py \\
        --lora_ckpt experiments/checkpoints/v10_baseline_yxyx/step-14177.safetensors \\
        --hard_cases Vượng Nhẫn "An Khang Thịnh Vượng" \\
        --output_dir experiments/results/v10_infer \\
        --also_raw

    # Legacy bbox mode
    python3 infer_v10_lora.py --use_bbox --lora_ckpt ... --hard_cases Vượng ...

    # Custom prompt
    python3 infer_v10_lora.py --lora_ckpt ckpt.safetensors --custom_prompt "..."
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSynth-Studio"))

from diffsynth.core import ModelConfig
from diffsynth.pipelines.ideogram4 import Ideogram4Pipeline

from src.peft.hybrid_peft import (
    inject_lora_into_dit,
    load_lora_checkpoint,
)

STYLE = {
    "aesthetics": "Vietnamese traditional calligraphy, balanced composition, hand-drawn brush strokes, minimalist",
    "lighting":   "soft daylight, no shadows",
    "photo":      "flat lay, sharp focus, front view",
    "medium":     "ink on paper",
}

ANTI_HAN_PHRASE = "Chữ Latin tiếng Việt, KHÔNG dùng Hán tự."
FONT_NAME = "Thu Phap Thanh Cong"
DEFAULT_FONT_PATH = str(PROJECT_ROOT / "assets/fonts/Thu_Phap_Thanh_Cong_Unicode.ttf")
DEFAULT_BBOX = [350, 200, 674, 824]

def split_syllables(content: str) -> list[str]:
    import re
    s = unicodedata.normalize("NFC", content).strip()
    parts = re.split(r"\s+", s)
    return [p for p in parts if p]

def _v87_optimal_font(text: str, font_path: str) -> tuple[ImageFont.FreeTypeFont, int]:
    from PIL import ImageFont
    W, H = 1024, 1024
    mw = W * (1 - 0.15 * 2)
    mh = H * (1 - 0.15 * 2)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    fs = 1200
    while fs > 50:
        try:
            font = ImageFont.truetype(font_path, fs)
        except Exception:
            return ImageFont.load_default(), 50
        b = dummy.textbbox((0, 0), text, font=font)
        if (b[2] - b[0]) < mw and (b[3] - b[1]) < mh:
            return font, fs
        fs -= 10
    return ImageFont.load_default(), 50

def replay_v87_bbox(text: str, font_path: str = DEFAULT_FONT_PATH) -> list[int] | None:
    if not Path(font_path).exists():
        return None
    try:
        from PIL import Image, ImageDraw
        font, _ = _v87_optimal_font(text, font_path)
    except Exception:
        return None
    W, H = 1024, 1024
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    b = dummy.textbbox((0, 0), text, font=font)
    w_px, h_px = b[2] - b[0], b[3] - b[1]
    x0 = (W - w_px) / 2
    y0 = (H - h_px) / 2
    x1 = x0 + w_px
    y1 = y0 + h_px
    scale = 1000.0 / W
    return [round(y0 * scale), round(x0 * scale), round(y1 * scale), round(x1 * scale)]

def bbox_for_syllables(n: int, idx: int, syllables: list[str] | None = None, per_sample: bool = True) -> list[int]:
    if n == 1:
        if per_sample and syllables:
            replayed = replay_v87_bbox(syllables[0])
            if replayed is not None:
                return replayed
        return list(DEFAULT_BBOX)
    margin = 50
    gap = 25
    col_w = (1000 - 2 * margin - (n - 1) * gap) // n
    x1 = margin + idx * (col_w + gap)
    x2 = x1 + col_w
    return [350, x1, 650, x2]

def build_record(content: str, *, per_sample_bbox: bool = True) -> dict:
    sylls = split_syllables(content)
    elements = []
    for i, syl in enumerate(sylls):
        elements.append({
            "type": "text",
            "bbox": bbox_for_syllables(len(sylls), i, syllables=sylls, per_sample=per_sample_bbox),
            "text": syl,
            "desc": f'Vietnamese calligraphy character "{syl}" — Latin Vietnamese letters with diacritics, NOT Chinese characters. Font: {FONT_NAME}.',
        })
    return {
        "high_level_description": f'Vietnamese calligraphy artwork of the word "{content}" in traditional brush style. {ANTI_HAN_PHRASE}',
        "style_description": dict(STYLE),
        "compositional_deconstruction": {
            "background": "Plain white rice-paper background, no texture, no border.",
            "elements": elements,
        },
    }

import unicodedata
from PIL import Image, ImageDraw, ImageFont

DEFAULT_MODEL_DIR = "models/ideogram-ai/ideogram-4-fp8"

# Layout-aware desc templates cho no-bbox mode (mirror test_no_bbox.py Option A —
# verified 2026-06-11 trên step-14177 yxyx: model tự layout theo `\n`,
# phương án duy nhất không lỗi ký tự trong loạt compound eval)
_NO_BBOX_DESC = {
    "single":     "Traditional Vietnamese calligraphy character, centered, written in bold black ink brush strokes. Font: {font}.",
    "horizontal": "Traditional Vietnamese calligraphy characters on a single line, centered, written in bold black ink brush strokes. Font: {font}.",
    "vertical":   "Traditional Vietnamese calligraphy characters, one word per line stacked vertically, centered, written in bold black ink brush strokes. Font: {font}.",
    "grid":       "Traditional Vietnamese calligraphy characters. The words are stacked on two lines in a tight 2x2 grid, centered, written in bold black ink brush strokes. Font: {font}.",
    "rows":       "Traditional Vietnamese calligraphy characters arranged in a tidy grid of several stacked rows, multiple words per row, evenly spaced and centered, written in bold black ink brush strokes. Font: {font}.",
}


def _chunk_rows(words: list[str], cols: int) -> str:
    """Chunk words into rows of `cols`, join rows by `\\n` (each row space-joined)."""
    rows = [" ".join(words[i:i + cols]) for i in range(0, len(words), cols)]
    return "\n".join(rows)


def _layout_text(words: list[str], layout: str) -> tuple[str, str]:
    """Map (words, layout) → (element text với `\\n` line structure, desc key)."""
    n = len(words)
    if n == 1:
        return words[0], "single"
    if layout == "auto":
        # scale với độ dài: 2-3 từ → vertical (\n mỗi từ 1 dòng); 4 → grid 2×2;
        # 5-8 → cols2 (hàng 2 từ, nhiều dòng \n). Giữ \n làm primitive layout.
        if n <= 3:
            layout = "vertical"
        elif n == 4:
            layout = "grid"
        else:
            layout = "cols2"
    if layout == "horizontal":
        return " ".join(words), "horizontal"
    if layout == "vertical":
        return "\n".join(words), "vertical"
    if layout == "grid":
        if n != 4:
            raise ValueError(f"grid layout needs 4 words, got {n}")
        return f"{words[0]} {words[1]}\n{words[2]} {words[3]}", "grid"
    if layout.startswith("cols"):
        cols = int(layout[4:] or "2")
        if cols < 1:
            raise ValueError(f"cols layout needs cols>=1, got {cols}")
        # 1 hàng → single-line (dùng desc horizontal); >1 hàng → rows
        desc_key = "horizontal" if cols >= n else ("vertical" if cols == 1 else "rows")
        return _chunk_rows(words, cols), desc_key
    raise ValueError(f"unknown layout: {layout}")


def build_no_bbox_prompt(phrase: str, layout: str = "auto") -> str:
    """DEFAULT inference prompt — single text element, KHÔNG bbox (Option A).

    Verified 2026-06-11 (docs/v10_inference_evaluation_report.vi.md §4): LoRA
    step-14177 yxyx không còn safety-abstain khi thiếu bbox (khác raw model),
    tự layout theo `\\n` tự nhiên hơn bbox cứng, và là phương án duy nhất
    không lỗi ký tự trong loạt thử nghiệm compound.
    Layout: auto = single-line cho 1-3 từ, grid 2×2 cho 4 từ.
    """
    words = phrase.split()
    text, desc_key = _layout_text(words, layout)
    record = {
        "high_level_description": (
            f'Vietnamese calligraphy artwork of the '
            f'{"word" if len(words) == 1 else "phrase"} "{phrase}" '
            f'in traditional brush style. {ANTI_HAN_PHRASE}'
        ),
        "style_description": dict(STYLE),
        "compositional_deconstruction": {
            "background": "Plain white rice-paper background, no texture, no border.",
            "elements": [{
                "type": "text",
                "text": text,
                "desc": _NO_BBOX_DESC[desc_key].format(font=FONT_NAME),
            }],
        },
    }
    return json.dumps(record, ensure_ascii=False)


def build_v3_1_prompt(word: str) -> str:
    """LEGACY bbox mode (--use_bbox): v3.1 schema với per-sample bbox replay.

    Dùng khi cần position control tường minh (bbox [y0,x0,y1,x1] image-axis).
    Default inference giờ là build_no_bbox_prompt() — xem
    docs/v10_inference_evaluation_report.vi.md §4.
    """
    record = build_record(word, per_sample_bbox=True)
    return json.dumps(record, ensure_ascii=False, indent=2)


def run_inference(
    pipe,
    prompt: str,
    *,
    height: int = 1024,
    width: int = 1024,
    cfg_scale: float = 7.0,
    num_inference_steps: int = 32,
    seed: int = 42,
):
    """Generate one image via the standard Ideogram4 pipeline."""
    t0 = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        cfg_scale=cfg_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    return image, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--lora_ckpt", type=Path, required=True,
                    help="step-N.safetensors LoRA checkpoint")
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=float, default=64.0)
    ap.add_argument("--lora_targets", nargs="+", default=["attention.qkv", "attention.o"])
    ap.add_argument("--hard_cases", nargs="+", default=None,
                    help="Vietnamese words/phrases to render (multi-word phrases OK, "
                         "e.g., Vượng Nhẫn 'An Khang Thịnh Vượng')")
    ap.add_argument("--custom_prompt", type=str, default=None,
                    help="Use a custom JSON prompt instead of word list")
    ap.add_argument("--use_bbox", action="store_true",
                    help="LEGACY: dùng v3.1 per-sample bbox prompt. Default là "
                         "no-bbox Option A (verified 2026-06-11, không lỗi ký tự)")
    ap.add_argument("--layout", type=str, default="auto",
                    choices=["auto", "horizontal", "vertical", "grid"],
                    help="Layout cho no-bbox multi-word: auto = 1-line cho 1-3 từ, "
                         "grid 2×2 cho 4 từ")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--also_raw", action="store_true",
                    help="Also generate raw (no LoRA) for A/B comparison")
    ap.add_argument("--cfg_scale", type=float, default=7.0)
    ap.add_argument("--num_inference_steps", type=int, default=48,
                    help="48 = verified config (v10_inference_evaluation_report §1); "
                         "pipeline default 50; 32 chỉ cho quick smoke")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--skip_unconditional", action="store_true",
                    help="Don't load unconditional_transformer (saves ~9 GB VRAM)")
    args = ap.parse_args()

    if not args.hard_cases and not args.custom_prompt:
        ap.error("must provide --hard_cases or --custom_prompt")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": [],
    }
    log_path = args.output_dir / "inference_log.json"

    # ── Load pipeline ────────────────────────────────────────────────────────
    print(f"[1] Loading pipeline (skip_unconditional={args.skip_unconditional})...")
    t0 = time.time()
    model_configs = [
        ModelConfig(path=str(args.model_dir / "transformer" / "diffusion_pytorch_model.safetensors"), skip_download=True),
        ModelConfig(path=str(args.model_dir / "text_encoder" / "model.safetensors"), skip_download=True),
        ModelConfig(path=str(args.model_dir / "vae" / "diffusion_pytorch_model.safetensors"), skip_download=True),
    ]
    if not args.skip_unconditional:
        model_configs.insert(1, ModelConfig(
            path=str(args.model_dir / "unconditional_transformer" / "diffusion_pytorch_model.safetensors"),
            skip_download=True,
        ))
    pipe = Ideogram4Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(args.model_dir / "tokenizer"), skip_download=True),
    )
    print(f"    loaded in {time.time()-t0:.1f}s, free VRAM {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # Build prompt list (default no-bbox Option A; --use_bbox cho legacy v3.1)
    if args.custom_prompt:
        cases = [("custom", args.custom_prompt)]
    elif args.use_bbox:
        cases = [(phrase.replace(" ", "_"), build_v3_1_prompt(phrase))
                 for phrase in args.hard_cases]
    else:
        cases = [(phrase.replace(" ", "_"), build_no_bbox_prompt(phrase, args.layout))
                 for phrase in args.hard_cases]
    mode = "custom" if args.custom_prompt else ("bbox-v3.1" if args.use_bbox else "no-bbox")
    print(f"[2] {len(cases)} prompts queued (mode={mode})")

    # ── Raw inference (optional) ────────────────────────────────────────────
    if args.also_raw:
        print(f"\n[3a] Running RAW inference (no LoRA)...")
        for name, prompt in cases:
            img, dt = run_inference(
                pipe, prompt,
                height=args.height, width=args.width,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
            )
            img_path = args.output_dir / f"raw_{name}.jpg"
            img.save(img_path, quality=92)
            print(f"    raw {name!r:15s} → {img_path.name}  ({dt:.1f}s)")
            log["results"].append({
                "kind": "raw",
                "name": name,
                "image": str(img_path),
                "seconds": round(dt, 1),
            })
            # Persist log incrementally
            log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Inject LoRA + load checkpoint ───────────────────────────────────────
    print(f"\n[3b] Injecting LoRA + loading {args.lora_ckpt.name}...")
    replaced = inject_lora_into_dit(
        pipe.dit,
        target_patterns=tuple(args.lora_targets),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    print(f"    injected {len(replaced)} modules")
    missing, unexpected = load_lora_checkpoint(pipe.dit, args.lora_ckpt, strict=True)
    print(f"    loaded LoRA from {args.lora_ckpt}: missing={len(missing)}, unexpected={len(unexpected)}")

    # If we have unconditional_transformer, inject + load LoRA there too
    if pipe.dit_uncond is not None:
        print(f"    (also injecting LoRA into unconditional_transformer)")
        unc_replaced = inject_lora_into_dit(
            pipe.dit_uncond,
            target_patterns=tuple(args.lora_targets),
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        load_lora_checkpoint(pipe.dit_uncond, args.lora_ckpt, strict=True)
        print(f"    {len(unc_replaced)} modules in dit_uncond")

    # ── LoRA inference ──────────────────────────────────────────────────────
    print(f"\n[4] Running LoRA inference...")
    for name, prompt in cases:
        img, dt = run_inference(
            pipe, prompt,
            height=args.height, width=args.width,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )
        img_path = args.output_dir / f"lora_{name}.jpg"
        img.save(img_path, quality=92)
        print(f"    lora {name!r:15s} → {img_path.name}  ({dt:.1f}s)")
        log["results"].append({
            "kind": "lora",
            "name": name,
            "image": str(img_path),
            "seconds": round(dt, 1),
        })
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] {len(log['results'])} images saved to {args.output_dir}")
    print(f"       log: {log_path}")


if __name__ == "__main__":
    main()
