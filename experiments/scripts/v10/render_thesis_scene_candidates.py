"""Render scene-rich thesis candidates across raw/soup/final checkpoints.

This is intentionally a candidate renderer: it tries alternate seeds/prompts for
Figure 3.8 without overwriting the promoted figure unless the user chooses one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "DiffSynth-Studio"))
sys.path.insert(0, str(ROOT / "experiments/scripts/v10"))

from diffsynth.core import ModelConfig  # noqa: E402
from diffsynth.pipelines.ideogram4 import Ideogram4Pipeline  # noqa: E402
from hybrid_peft_ideogram4 import inject_lora_into_dit, load_lora_checkpoint  # noqa: E402

MODEL_DIR = ROOT / "models/ideogram-ai/ideogram-4-fp8"
SOUP567 = ROOT / "experiments/checkpoints/coverage_v10_widetarget_soup567/step-soup_infer.safetensors"
FINAL = ROOT / "experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors"
OUT_ROOT = ROOT / "experiments/results/coverage_v10_eval/thesis_scene_candidates"
FIG = ROOT / "docs/thesis/figures"

TARGETS = (
    "attention.qkv",
    "attention.o",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "adaln_modulation",
)


def build_scene_prompt(phrase: str, text: str, *, dense: bool = False) -> str:
    text_bbox = [205, 205, 785, 795] if dense else [245, 245, 760, 780]
    lines = [ln for ln in text.split("\n") if ln.strip()]
    n = len(lines)
    line_spec = "; ".join(f'line {i + 1}: "{ln}"' for i, ln in enumerate(lines))
    size = "Large, compact but clearly readable" if dense else "Large, centered"
    record = {
        "high_level_description": (
            "A square editorial photograph of a Vietnamese Tet calligraphy still life, "
            "combining polished product photography with traditional brush calligraphy. "
            "A FLAT cream rice-paper panel, photographed straight-on (frontal view), carries the black "
            f'Vietnamese calligraphy text "{phrase}", written horizontally and upright in exactly {n} '
            f"centered level lines ({line_spec}), parallel to the bottom edge of the image — the text is "
            "NOT slanted, NOT rotated, NOT italic, and NOT on a curved or hanging scroll. The panel is "
            "surrounded by apricot blossoms, kumquat fruit, bamboo brushes, an inkstone, and a small red "
            "seal. Preserve rich image-generation details while keeping the Vietnamese diacritics readable. "
            "No Chinese Han characters."
        ),
        "style_description": {
            "aesthetics": "elegant Vietnamese New Year still life, refined and tactile",
            "lighting": "soft late-afternoon window light from the left with gentle shadows",
            "medium": "editorial product photograph",
            "art_style": "high-resolution realistic photography with shallow depth of field",
            "color_palette": ["#B31B1B", "#F2E6C9", "#111111", "#D6A13A", "#2E6B3F"],
        },
        "compositional_deconstruction": {
            "background": (
                "A warm domestic studio corner softly blurred behind the tabletop. "
                "The red lacquer table fills the lower frame; a cream linen curtain and "
                "muted wooden wall sit in the background. Fine paper fibers, ink texture, "
                "lacquer reflections, and soft bokeh are visible."
            ),
            "elements": [
                {
                    "type": "obj",
                    "bbox": [115, 135, 890, 870],
                    "desc": (
                        "A FLAT cream rice-paper panel standing upright on a red lacquer table and facing the "
                        "camera directly (frontal, straight-on view). Its surface is flat and smooth — NOT "
                        "curved, NOT draped, NOT a hanging scroll tilted at an angle — with deckled edges and "
                        "subtle paper fibers."
                    ),
                },
                {
                    "type": "text",
                    "bbox": text_bbox,
                    "text": text,
                    "desc": (
                        f"{size} traditional Vietnamese brush-calligraphy text in bold black ink, centered on "
                        f"the flat cream paper, arranged in exactly {n} horizontal centered lines. The text is "
                        "upright and level, parallel to the bottom edge of the image — not slanted, not rotated, "
                        "not italic, not following any curve. Preserve all Vietnamese diacritics exactly."
                    ),
                },
                {
                    "type": "obj",
                    "desc": (
                        "A black ceramic inkstone and two bamboo calligraphy brushes resting "
                        "near the lower-left corner of the scroll."
                    ),
                },
                {
                    "type": "obj",
                    "desc": "A small square red seal stamp and a round cinnabar paste dish near the lower-right edge.",
                },
                {
                    "type": "obj",
                    "desc": "Yellow apricot blossoms crossing the upper-left corner, with a few petals on the table.",
                },
                {
                    "type": "obj",
                    "desc": "Three glossy kumquat fruits with dark green leaves at the upper-right.",
                },
            ],
        },
    }
    return json.dumps(record, ensure_ascii=False, indent=2)


PROMPT_PRESETS = {
    "aktv_8words": {
        "aktv": {
            "phrase": "An Khang Thịnh Vượng",
            "text": "An Khang\nThịnh Vượng",
            "dense": False,
            "row_label": "Prompt A\n4 words",
            "subtitle": 'Prompt A = "An Khang Thinh Vuong", Prompt B = eight-word auspicious phrase.',
        },
        "eight_words": {
            "phrase": "Tâm An Phúc Lộc Thọ Tài Đức Vượng",
            "text": "Tâm An Phúc Lộc\nThọ Tài Đức Vượng",
            "dense": True,
            "row_label": "Prompt B\n8 words",
        },
    },
    "phuc_duc": {
        "phuc_tho_an_khang": {
            "phrase": "Phúc Thọ An Khang",
            "text": "Phúc Thọ\nAn Khang",
            "dense": False,
            "row_label": "Prompt A\n4 words",
            "subtitle": 'Prompt A = "Phuc Tho An Khang", Prompt B = "Tam Duc Tri Nhan / Phuc Tho Khang Vuong".',
        },
        "tam_duc_tri_nhan": {
            "phrase": "Tâm Đức Trí Nhân Phúc Thọ Khang Vượng",
            "text": "Tâm Đức Trí Nhân\nPhúc Thọ Khang Vượng",
            "dense": True,
            "row_label": "Prompt B\n8 words",
        },
    },
}


def load_pipe() -> Ideogram4Pipeline:
    model_configs = [
        ModelConfig(path=str(MODEL_DIR / "transformer/diffusion_pytorch_model.safetensors"), skip_download=True),
        ModelConfig(path=str(MODEL_DIR / "unconditional_transformer/diffusion_pytorch_model.safetensors"), skip_download=True),
        ModelConfig(path=str(MODEL_DIR / "text_encoder/model.safetensors"), skip_download=True),
        ModelConfig(path=str(MODEL_DIR / "vae/diffusion_pytorch_model.safetensors"), skip_download=True),
    ]
    return Ideogram4Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(MODEL_DIR / "tokenizer"), skip_download=True),
    )


def inject_once(pipe: Ideogram4Pipeline) -> None:
    inject_lora_into_dit(pipe.dit, target_patterns=TARGETS, rank=64, alpha=64.0)
    if pipe.dit_uncond is not None:
        inject_lora_into_dit(pipe.dit_uncond, target_patterns=TARGETS, rank=64, alpha=64.0)


def load_ckpt(pipe: Ideogram4Pipeline, ckpt: Path) -> tuple[int, int]:
    missing, unexpected = load_lora_checkpoint(pipe.dit, ckpt, strict=False)
    if pipe.dit_uncond is not None:
        load_lora_checkpoint(pipe.dit_uncond, ckpt, strict=False)
    return len(missing), len(unexpected)


def run(pipe: Ideogram4Pipeline, prompt: str, out: Path, *, seed: int, steps: int, cfg: float) -> float:
    t0 = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        cfg_scale=cfg,
        height=1024,
        width=1024,
        num_inference_steps=steps,
        seed=seed,
    )
    image.save(out, quality=92)
    return time.time() - t0


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(p, size=size) if Path(p).exists() else ImageFont.load_default()


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    bg = Image.new("RGB", size, "white")
    bg.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return bg


def center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt, fill=(25, 25, 25)) -> None:
    bb = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=5, align="center")
    x = box[0] + (box[2] - box[0] - (bb[2] - bb[0])) // 2
    y = box[1] + (box[3] - box[1] - (bb[3] - bb[1])) // 2
    draw.multiline_text((x, y), text, font=fnt, fill=fill, spacing=5, align="center")


def make_sheet(out_dir: Path, seed: int, promoted: bool, prompts: dict[str, dict[str, object]], preset: str) -> Path:
    variants = [
        ("Ideogram4 base", "raw"),
        ("soup567", "soup567"),
        ("Final gold4 9:1", "final"),
    ]
    rows = [(str(spec["row_label"]), key) for key, spec in prompts.items()]
    cell_w, cell_h = 360, 360
    left_w, pad, top_h, cap_h = 180, 22, 94, 60
    w = left_w + pad + len(variants) * cell_w + (len(variants) - 1) * pad + pad
    h = top_h + len(rows) * (cell_h + cap_h) + (len(rows) - 1) * pad + pad
    canvas = Image.new("RGB", (w, h), (248, 247, 244))
    d = ImageDraw.Draw(canvas)
    center(d, (pad, 8, w - pad, 44), f"Scene-rich calligraphy candidates, seed {seed}", font(28, True))
    note = "promoted figure source" if promoted else "candidate sheet, not promoted yet"
    center(d, (pad, 48, w - pad, top_h - 8), f"{note}; preset={preset}", font(17), fill=(70, 70, 70))
    for i, (label, _) in enumerate(variants):
        x = left_w + pad + i * (cell_w + pad)
        center(d, (x, top_h - 42, x + cell_w, top_h - 8), label, font(18, True))
    for r, (row_label, key) in enumerate(rows):
        y = top_h + r * (cell_h + cap_h + pad)
        center(d, (pad, y, left_w, y + cell_h), row_label, font(18, True))
        for c, (_, variant) in enumerate(variants):
            x = left_w + pad + c * (cell_w + pad)
            path = out_dir / f"{variant}_{key}_s{seed}.jpg"
            canvas.paste(fit(path, (cell_w, cell_h)), (x, y))
            d.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(202, 199, 191), width=2)
            center(d, (x, y + cell_h + 5, x + cell_w, y + cell_h + cap_h), path.name, font(11), fill=(70, 70, 70))
    sheet = out_dir / f"scene_candidates_seed{seed}.png"
    canvas.save(sheet)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--preset", choices=sorted(PROMPT_PRESETS), default="aktv_8words")
    ap.add_argument("--promote", action="store_true", help="copy the contact sheet to docs/thesis/figures/fig_3_8...")
    args = ap.parse_args()

    prompts = PROMPT_PRESETS[args.preset]
    out_dir = OUT_ROOT / f"{args.preset}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    log = {
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "preset": args.preset,
        "prompts": prompts,
        "results": [],
    }
    for key, spec in prompts.items():
        prompt = build_scene_prompt(spec["phrase"], spec["text"], dense=spec["dense"])
        (out_dir / f"prompt_{key}.json").write_text(prompt, encoding="utf-8")

    print("[load] pipeline")
    pipe = load_pipe()
    print(f"  free VRAM {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    print("[render] raw Ideogram4")
    for key in prompts:
        prompt = (out_dir / f"prompt_{key}.json").read_text(encoding="utf-8")
        path = out_dir / f"raw_{key}_s{args.seed}.jpg"
        dt = run(pipe, prompt, path, seed=args.seed, steps=args.steps, cfg=args.cfg)
        print(f"  raw     {key}: {dt:.1f}s")
        log["results"].append({"variant": "raw", "prompt": key, "image": str(path.relative_to(ROOT)), "seconds": round(dt, 1)})
        (out_dir / "render_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[lora] inject")
    inject_once(pipe)

    print("[render] soup567")
    miss, unexp = load_ckpt(pipe, SOUP567)
    print(f"  loaded soup567 missing={miss} unexpected={unexp}")
    for key in prompts:
        prompt = (out_dir / f"prompt_{key}.json").read_text(encoding="utf-8")
        path = out_dir / f"soup567_{key}_s{args.seed}.jpg"
        dt = run(pipe, prompt, path, seed=args.seed, steps=args.steps, cfg=args.cfg)
        print(f"  soup567 {key}: {dt:.1f}s")
        log["results"].append({"variant": "soup567", "prompt": key, "image": str(path.relative_to(ROOT)), "seconds": round(dt, 1)})
        (out_dir / "render_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[render] final")
    miss, unexp = load_ckpt(pipe, FINAL)
    print(f"  loaded final missing={miss} unexpected={unexp}")
    for key in prompts:
        prompt = (out_dir / f"prompt_{key}.json").read_text(encoding="utf-8")
        path = out_dir / f"final_{key}_s{args.seed}.jpg"
        dt = run(pipe, prompt, path, seed=args.seed, steps=args.steps, cfg=args.cfg)
        print(f"  final   {key}: {dt:.1f}s")
        log["results"].append({"variant": "final", "prompt": key, "image": str(path.relative_to(ROOT)), "seconds": round(dt, 1)})
        (out_dir / "render_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    sheet = make_sheet(out_dir, args.seed, args.promote, prompts, args.preset)
    print(f"[done] sheet: {sheet}")
    if args.promote:
        promoted = FIG / "fig_3_8_calligraphy_with_base_model_capability.png"
        promoted.write_bytes(sheet.read_bytes())
        print(f"[done] promoted: {promoted}")


if __name__ == "__main__":
    main()
