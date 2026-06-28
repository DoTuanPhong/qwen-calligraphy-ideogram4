from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SANS = Path(font_manager.findfont("DejaVu Sans"))
SANS_BOLD = SANS.with_name("DejaVuSans-Bold.ttf")
CALLIGRAPHY_FONT = ROOT / "experiments/gen_dataset/fonts/Thu_Phap_Thanh_Cong_Unicode.ttf"

INK = "#1f2933"
MUTED = "#64748b"
BLUE = "#2563eb"
GREEN = "#15803d"
ORANGE = "#ea580c"
RED = "#b91c1c"
PURPLE = "#7c3aed"
CREAM = "#fffaf0"
PAPER = "#fffdf8"
LINE = "#d6d3d1"


def font(size: int, bold: bool = False, calligraphy: bool = False) -> ImageFont.FreeTypeFont:
    path = CALLIGRAPHY_FONT if calligraphy else (SANS_BOLD if bold else SANS)
    return ImageFont.truetype(str(path), size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        words = raw.split(" ")
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if text_size(draw, trial, fnt)[0] <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    fnt: ImageFont.FreeTypeFont,
    fill: str = INK,
    align: str = "center",
    spacing: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    lines = wrap_text(draw, text, fnt, x2 - x1)
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + max(0, (y2 - y1 - total_h) // 2)
    for line, h in zip(lines, heights):
        w, _ = text_size(draw, line, fnt)
        if align == "left":
            x = x1
        elif align == "right":
            x = x2 - w
        else:
            x = x1 + (x2 - x1 - w) // 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += h + spacing


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str = LINE,
    width: int = 3,
    radius: int = 24,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str = "#334155",
    width: int = 5,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        head = [(x2, y2), (x2 - 18 * direction, y2 - 11), (x2 - 18 * direction, y2 + 11)]
    else:
        direction = 1 if y2 >= y1 else -1
        head = [(x2, y2), (x2 - 11, y2 - 18 * direction), (x2 + 11, y2 - 18 * direction)]
    draw.polygon(head, fill=fill)


def title(draw: ImageDraw.ImageDraw, text: str, subtitle: str | None = None) -> None:
    draw.text((70, 45), text, font=font(46, True), fill=INK)
    if subtitle:
        draw.text((72, 104), subtitle, font=font(24), fill=MUTED)


def save(img: Image.Image, name: str) -> None:
    path = HERE / name
    img.save(path)
    print(path)


def paste_contained(
    canvas: Image.Image,
    img: Image.Image,
    box: tuple[int, int, int, int],
    bg: str = "#ffffff",
) -> None:
    x1, y1, x2, y2 = box
    area = Image.new("RGB", (x2 - x1, y2 - y1), bg)
    contained = ImageOps.contain(img.convert("RGB"), area.size, method=Image.Resampling.LANCZOS)
    px = (area.width - contained.width) // 2
    py = (area.height - contained.height) // 2
    area.paste(contained, (px, py))
    canvas.paste(area, (x1, y1))


def make_digital_font_render() -> Image.Image:
    img = Image.new("RGB", (1024, 1024), "#fbfaf7")
    draw = ImageDraw.Draw(img)
    phrase = "An Khang Th\u1ecbnh V\u01b0\u1ee3ng"
    fnt = font(130, calligraphy=True)
    lines = ["An Khang", "Th\u1ecbnh V\u01b0\u1ee3ng"]
    y = 315
    for line in lines:
        w, h = text_size(draw, line, fnt)
        draw.text(((1024 - w) // 2, y), line, font=fnt, fill="#171717")
        y += h + 40
    draw.text((42, 920), phrase, font=font(28), fill=MUTED)
    return img


def placeholder_panel() -> Image.Image:
    img = Image.new("RGB", (1024, 1024), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((90, 120, 934, 904), radius=28, fill="#ffffff", outline="#cbd5e1", width=5)
    draw_wrapped(
        draw,
        "Commercial black-box\nbaseline image\nnot archived in repo",
        (160, 330, 864, 555),
        font(54, True),
        fill=INK,
    )
    draw_wrapped(
        draw,
        "Replace this cell with the actual\ncommercial-tool render before final submission.",
        (185, 620, 839, 735),
        font(29),
        fill=MUTED,
    )
    return img


def figure_1_1_pipeline() -> None:
    img = Image.new("RGB", (2200, 1100), PAPER)
    draw = ImageDraw.Draw(img)
    title(draw, "Figure 1.1. Overall research pipeline", "From baseline diagnosis to compound-layout Vietnamese calligraphy evaluation")

    boxes = [
        ("Baseline and\ncompetitor survey", "#dbeafe"),
        ("Qwen3-VL / DiT\nsignal probes", "#ede9fe"),
        ("Attention-only\nLoRA plateau", "#fee2e2"),
        ("Wide-target\nDiT LoRA", "#ffedd5"),
        ("Manual gate and\nvariance diagnosis", "#fef3c7"),
        ("Checkpoint soup\nfor stability", "#dcfce7"),
        ("Compound font\nreference data", "#e0f2fe"),
        ("Final compound\nEval28", "#d1fae5"),
    ]
    coords = [
        (90, 235, 510, 390),
        (610, 235, 1030, 390),
        (1130, 235, 1550, 390),
        (1650, 235, 2070, 390),
        (1650, 635, 2070, 790),
        (1130, 635, 1550, 790),
        (610, 635, 1030, 790),
        (90, 635, 510, 790),
    ]
    for (label, color), box in zip(boxes, coords):
        rounded_box(draw, box, color, outline="#94a3b8")
        draw_wrapped(draw, label, (box[0] + 22, box[1] + 20, box[2] - 22, box[3] - 20), font(34, True))

    for i in range(3):
        arrow(draw, (coords[i][2] + 20, 312), (coords[i + 1][0] - 20, 312))
    arrow(draw, (1860, 415), (1860, 610))
    for i in range(4, 7):
        arrow(draw, (coords[i][0] - 20, 712), (coords[i + 1][2] + 20, 712))

    draw.rounded_rectangle((90, 900, 2070, 1010), radius=22, fill="#ffffff", outline="#cbd5e1", width=3)
    draw_wrapped(
        draw,
        "Central finding: the final contribution is an Ideogram4 DiT-LoRA glyph-binding pipeline, not an isolated prompt trick.",
        (140, 918, 2020, 992),
        font(31, True),
        fill=INK,
    )
    save(img, "fig_1_1_research_pipeline.png")


def figure_1_2_competitors() -> None:
    canvas = Image.new("RGB", (1800, 1560), PAPER)
    draw = ImageDraw.Draw(canvas)
    title(draw, "Figure 1.2. Competitor and baseline comparison", "Same Vietnamese calligraphy goal across font rendering, external baselines, open models, and the proposed checkpoint")

    panels: list[tuple[str, Image.Image, str]] = [
        ("Digital font", make_digital_font_render(), "Correct glyphs, static style"),
        ("Nano Banana 2", Image.open(HERE / "Vietnamese_calligraphy_on_paper_202606290215.jpeg"), "Commercial black-box baseline"),
        ("Qwen Image", Image.open(HERE / "qwen_image_base.png"), "Weak Vietnamese baseline"),
        ("ERNIE Image", Image.open(HERE / "ernie_image_base.jpeg"), "Better count, still character errors"),
        ("Base Ideogram4", Image.open(HERE / "ideogram4_base_calligraphy_project_json_prompt_s7000.jpg"), "Project JSON prompt"),
        ("Proposed checkpoint", Image.open(HERE / "final_compound_calligraphy_simple_prompt_s7000.jpg"), "Fine-tuned target style"),
    ]

    cell_w, cell_h = 500, 500
    margin_x, gap_x = 80, 40
    top = 270
    for idx, (name, panel, note) in enumerate(panels):
        row, col = divmod(idx, 3)
        x = margin_x + col * (cell_w + gap_x + 20)
        y = top + row * (cell_h + 210)
        draw.text((x, y - 78), name, font=font(30, True), fill=INK)
        draw.text((x, y - 38), note, font=font(21), fill=MUTED)
        draw.rounded_rectangle((x - 5, y - 5, x + cell_w + 5, y + cell_h + 5), radius=18, fill="#ffffff", outline="#cbd5e1", width=3)
        paste_contained(canvas, panel, (x, y, x + cell_w, y + cell_h))

    draw.text(
        (80, 1490),
        "Prompt family: Vietnamese calligraphy. Nano Banana 2 is a commercial black-box baseline; Base Ideogram4 uses the project JSON prompt.",
        font=font(23),
        fill=MUTED,
    )
    save(canvas, "fig_1_2_competitor_baseline_comparison.png")


def figure_1_3_base_models() -> None:
    canvas = Image.new("RGB", (1800, 850), PAPER)
    draw = ImageDraw.Draw(canvas)
    title(
        draw,
        "Figure 1.3. Base-model capability before fine-tuning",
        "Qwen Image, ERNIE Image, and base Ideogram4 on Vietnamese calligraphy-style prompts",
    )

    panels: list[tuple[str, Image.Image, str]] = [
        ("Qwen Image base", Image.open(HERE / "qwen_image_base.png"), "Direct base render"),
        ("ERNIE Image base", Image.open(HERE / "ernie_image_base.jpeg"), "Direct base render"),
        ("Ideogram4 base", Image.open(HERE / "ideogram4_base_calligraphy_project_json_prompt_s7000.jpg"), "Project JSON prompt"),
    ]

    cell_w, cell_h = 500, 500
    x0, y = 75, 250
    for idx, (name, panel, note) in enumerate(panels):
        x = x0 + idx * 570
        draw.text((x, y - 78), name, font=font(30, True), fill=INK)
        draw.text((x, y - 38), note, font=font(21), fill=MUTED)
        draw.rounded_rectangle((x - 5, y - 5, x + cell_w + 5, y + cell_h + 5), radius=18, fill="#ffffff", outline="#cbd5e1", width=3)
        paste_contained(canvas, panel, (x, y, x + cell_w, y + cell_h))

    draw.text(
        (75, 795),
        "Ideogram4 base is shown with the project JSON prompt rather than the earlier safety-filter placeholder.",
        font=font(23),
        fill=MUTED,
    )
    save(canvas, "fig_1_3_base_model_capability_comparison.png")


def figure_2_1_architecture() -> None:
    img = Image.new("RGB", (2200, 1050), PAPER)
    draw = ImageDraw.Draw(img)
    title(draw, "Figure 2.1. Ideogram4 architecture used in this thesis", "Frozen Qwen3-VL conditioning, DiT LoRA adaptation, and VAE decoding")

    items = [
        ("Prompt /\nJSON prompt", "#dbeafe"),
        ("Qwen3-VL\ntext encoder\nFROZEN", "#ede9fe"),
        ("13 layer taps\nhidden states", "#f3e8ff"),
        ("llm_cond_norm\n+\nllm_cond_proj", "#e0f2fe"),
        ("Ideogram4 DiT\nbase frozen\nLoRA trainable", "#ffedd5"),
        ("VAE decoder\nFROZEN", "#dcfce7"),
        ("1024 x 1024\ncalligraphy image", "#d1fae5"),
    ]
    x = 70
    y = 330
    widths = [220, 250, 230, 280, 320, 220, 270]
    boxes = []
    for (label, color), w in zip(items, widths):
        box = (x, y, x + w, y + 210)
        boxes.append(box)
        rounded_box(draw, box, color, outline="#94a3b8")
        draw_wrapped(draw, label, (box[0] + 18, box[1] + 15, box[2] - 18, box[3] - 15), font(30, True))
        x += w + 30
    for a, b in zip(boxes, boxes[1:]):
        arrow(draw, (a[2] + 8, (a[1] + a[3]) // 2), (b[0] - 8, (b[1] + b[3]) // 2))

    draw.rounded_rectangle((640, 655, 1600, 835), radius=22, fill="#ffffff", outline="#cbd5e1", width=3)
    draw_wrapped(
        draw,
        "Training scope: keep the base model frozen and learn a small LoRA adapter inside the DiT. The text encoder is diagnosed but not fine-tuned.",
        (680, 682, 1560, 808),
        font(29),
        fill=INK,
    )
    save(img, "fig_2_1_ideogram4_architecture.png")


def figure_2_2_conditioning() -> None:
    img = Image.new("RGB", (2200, 1100), PAPER)
    draw = ImageDraw.Draw(img)
    title(draw, "Figure 2.2. Multi-layer Qwen3-VL conditioning", "Vietnamese word spans are represented through layer taps, concatenation, normalization, and projection")

    rounded_box(draw, (60, 230, 410, 420), "#dbeafe", outline="#93c5fd")
    draw_wrapped(draw, "Prompt span\nfor a Vietnamese word", (80, 260, 390, 390), font(29, True))
    arrow(draw, (430, 325), (500, 325))

    x0, y0 = 530, 230
    layer_w, layer_h, gap = 24, 340, 8
    taps = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35}
    for i in range(36):
        x = x0 + i * (layer_w + gap)
        color = "#7c3aed" if i in taps else "#e2e8f0"
        draw.rounded_rectangle((x, y0, x + layer_w, y0 + layer_h), radius=8, fill=color, outline="#cbd5e1", width=1)
        if i in {0, 9, 18, 27, 35}:
            draw.text((x - 5, y0 + layer_h + 18), f"L{i}", font=font(19, True), fill=INK)

    draw_wrapped(draw, "Qwen3-VL layers L0...L35\npurple bars are tapped layers", (560, 640, 1600, 735), font(27), fill=INK)

    extra = [(1710, 230, 1920, 420), (1710, 540, 1920, 700), (1710, 790, 1920, 950), (1970, 790, 2160, 950)]
    labels = ["Concat\n13 taps", "llm_cond\nnorm", "llm_cond\nproj", "DiT\nconditioning"]
    colors = ["#f3e8ff", "#e0f2fe", "#bae6fd", "#ffedd5"]
    for box, label, color in zip(extra, labels, colors):
        rounded_box(draw, box, color, outline="#94a3b8")
        draw_wrapped(draw, label, (box[0] + 15, box[1] + 20, box[2] - 15, box[3] - 20), font(28, True))
    arrow(draw, (1685, 325), (1700, 325))
    arrow(draw, (1815, 430), (1815, 530))
    arrow(draw, (1815, 710), (1815, 780))
    arrow(draw, (1930, 870), (1960, 870))

    draw.rounded_rectangle((80, 840, 1180, 980), radius=22, fill="#ffffff", outline="#cbd5e1", width=3)
    draw_wrapped(
        draw,
        "Probe interpretation: if words are close in tap-space or projection-space, Qwen/conditioning may contribute; if they are far yet images fail, DiT glyph binding is the stronger suspect.",
        (120, 865, 1140, 955),
        font(27),
        fill=INK,
    )
    save(img, "fig_2_2_qwen3vl_multilayer_conditioning.png")


def figure_3_1_lora_targets() -> None:
    img = Image.new("RGB", (2000, 1180), PAPER)
    draw = ImageDraw.Draw(img)
    title(draw, "Figure 3.1. Wide-target LoRA insertion points", "Attention-only LoRA was not enough; the final target set also touches FFN and adaLN modulation")

    block = (260, 210, 1740, 1010)
    draw.rounded_rectangle(block, radius=34, fill="#ffffff", outline="#94a3b8", width=5)
    draw.text((310, 245), "Ideogram4 DiT block", font=font(40, True), fill=INK)

    modules = [
        ("adaLN\nmodulation", (380, 365, 760, 525), "#fef3c7", True),
        ("Attention\nqkv + out", (960, 365, 1420, 525), "#dbeafe", True),
        ("Feed-forward\nw1 / w2 / w3", (960, 650, 1420, 810), "#dcfce7", True),
        ("Residual paths\nand frozen base", (380, 650, 760, 810), "#f8fafc", False),
    ]
    for label, box, color, trainable in modules:
        rounded_box(draw, box, color, outline="#94a3b8")
        draw_wrapped(draw, label, (box[0] + 20, box[1] + 15, box[2] - 20, box[3] - 55), font(31, True))
        tag = "LoRA target" if trainable else "Frozen"
        tag_color = ORANGE if trainable else MUTED
        draw.rounded_rectangle((box[0] + 90, box[3] - 48, box[2] - 90, box[3] - 16), radius=15, fill=tag_color)
        draw_wrapped(draw, tag, (box[0] + 96, box[3] - 45, box[2] - 96, box[3] - 18), font(20, True), fill="#ffffff")

    arrow(draw, (190, 595), (350, 595))
    draw_wrapped(draw, "conditioned\nlatent tokens", (35, 520, 190, 665), font(25, True), fill=INK)
    arrow(draw, (760, 445), (945, 445))
    arrow(draw, (1190, 540), (1190, 635))
    arrow(draw, (945, 735), (780, 735))
    arrow(draw, (1740, 595), (1905, 595))
    draw_wrapped(draw, "updated\nlatent tokens", (1795, 465, 1980, 555), font(25, True), fill=INK)

    draw.rounded_rectangle((350, 900, 1650, 980), radius=20, fill="#fff7ed", outline="#fed7aa", width=3)
    draw_wrapped(
        draw,
        "Wide-target LoRA gives the adapter more routes to affect glyph geometry, not only token-to-token attention.",
        (390, 912, 1610, 968),
        font(28, True),
        fill=INK,
    )
    save(img, "fig_3_1_widetarget_lora_injection_points.png")


def figure_3_4_progression() -> None:
    single_labels = ["r3", "r4", "r5", "r6", "r7", "r8", "soup567", "soup5679"]
    single_scores = [43, 41, 46, 48, 47, 38, 52, 52]
    compound_labels = ["soup567", "e4", "r2", "r3", "soup\nr2r3", "Gold4", "lr3e5", "final\n9:1"]
    compound_errors = [56, 26, 19, 18, 13, 6, 5, 4]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=180)
    fig.patch.set_facecolor(PAPER)

    ax = axes[0]
    ax.set_title("Single-word fragile panel: correct / 60", loc="left", fontsize=15, fontweight="bold")
    ax.axhspan(32, 39, color="#fee2e2", alpha=0.75, label="attention-only plateau 32-39")
    ax.plot(single_labels, single_scores, marker="o", linewidth=3, color=GREEN)
    for i, y in enumerate(single_scores):
        ax.text(i, y + 0.8, str(y), ha="center", fontsize=10)
    ax.set_ylim(30, 55)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    ax.set_ylabel("Correct words")

    ax = axes[1]
    ax.set_title("Compound Eval28: errors / 168 (lower is better)", loc="left", fontsize=15, fontweight="bold")
    ax.plot(compound_labels, compound_errors, marker="o", linewidth=3, color=RED)
    ax.fill_between(range(len(compound_errors)), compound_errors, [0] * len(compound_errors), color="#fee2e2", alpha=0.25)
    for i, y in enumerate(compound_errors):
        ax.text(i, y + 2, str(y), ha="center", fontsize=10)
    ax.set_ylim(0, 62)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("Errors")

    fig.suptitle("Figure 3.4. Result progression", fontsize=21, fontweight="bold", x=0.02, ha="left")
    fig.text(
        0.02,
        0.935,
        "Wide-target LoRA improves the single-word plateau; compound training then reduces multi-word errors from 56 to 4.",
        fontsize=12,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(HERE / "fig_3_4_result_progression.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(HERE / "fig_3_4_result_progression.png")


def main() -> None:
    figure_1_1_pipeline()
    figure_1_2_competitors()
    figure_1_3_base_models()
    figure_2_1_architecture()
    figure_2_2_conditioning()
    figure_3_1_lora_targets()
    figure_3_4_progression()


if __name__ == "__main__":
    main()
