"""V10 — COMPOUND (multi-word 2-line) training dataset builder.

Mục tiêu (user 2026-06-26): chuẩn bị dataset training cho NÚT THẮT compound/layout binding
(single-word 52/60 KHÔNG transfer sang câu: soup567 compound no-bbox = 44/60; lỗi đa số ở
token STABLE bị mất binding-dấu khi nhiều chữ — crowding/layout, KHÔNG phải token fragile).

Thiết kế:
- Ảnh = NHÓM 4/5/7/8 từ, chia 2 DÒNG (line1 = ceil(n/2) từ đầu, line2 = phần còn lại).
- Target font = COMPACT CENTERED 2-line (model-free xấp xỉ của modelmatch_wide: model tự căn-giữa
  một block dẹt-rộng; modelmatch_wide cần model-gen mỗi ảnh nên không dùng được cho training words).
- Prompt = build_no_bbox_prompt(phrase, layout=cols{ceil(n/2)}) — ĐÚNG path no-bbox thắng (44/60),
  `\n` line structure, official JSON.
- COVERAGE: greedy set-cover MỌI diacritic token-id (cả HOA lẫn THƯỜNG) từ lexicon 7184 → mọi token
  dấu tiếng Việt xuất hiện ≥1 lần trong câu. Fragile-core (elim soup567) được OVERSAMPLE.
- 1-từ = bổ trợ ÍT (vài %).

Usage:
  python3 experiments/scripts/v10/build_compound_dataset.py \
      [--frag_oversample 2] [--single_frac 0.08] [--seed 26] \
      [--out metadata_compound_v1.jsonl] [--limit 0]
"""
from __future__ import annotations
import argparse, json, math, random, sys, unicodedata
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / "DiffSynth-Studio"))
sys.path.insert(0, str(R / "experiments/scripts/v10"))
from infer_v10_lora import build_no_bbox_prompt  # noqa: E402
from render_compound6_font_reference import fit_font_for_block, draw_text_centered  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

LEX = R / "experiments/gen_dataset/vietnamesesyllable_7184.txt"
FONT = R / "experiments/gen_dataset/fonts/Thu_Phap_Thanh_Cong_Unicode.ttf"
TOK_DIR = R / "models/ideogram-ai/ideogram-4-fp8/tokenizer"
ELIM = R / "experiments/results/coverage_v10_eval/elim_state_soup567.json"
OUTDIR = R / "data/coverage_v10"
IMGSUB = "images_compound"
SIZE = 1024

TONES = {0x300, 0x301, 0x303, 0x309, 0x323}
MODS = {0x302, 0x306, 0x31b}


def is_diac(s: str) -> bool:
    return any((ord(c) in TONES or ord(c) in MODS or c in "đĐ")
               for c in unicodedata.normalize("NFD", s))


def cap(w: str) -> str:
    return w[:1].upper() + w[1:].lower()


def render_compact_2line(lines: list[str], font_path: Path,
                         w: int = SIZE, h: int = SIZE,
                         width_frac: float = 0.88, height_frac: float = 0.44,
                         y_center: float = 0.49, line_gap: float = 0.12) -> Image.Image:
    """Compact centered 2-line (xấp xỉ layout model: block dẹt-rộng căn giữa)."""
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    font = fit_font_for_block(lines, font_path, w * width_frac, h * height_frac,
                              line_gap_ratio=line_gap)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bxs = [dummy.textbbox((0, 0), ln, font=font) for ln in lines]
    hts = [b[3] - b[1] for b in bxs]
    total_h = sum(hts) + max(0, len(lines) - 1) * font.size * line_gap
    y = y_center * h - total_h / 2
    for ln, b, ht in zip(lines, bxs, hts):
        draw_text_centered(draw, ln, font, (0, y, w, y + ht))
        y += ht + font.size * line_gap
    return img.convert("RGB")


def greedy_set_cover(word_tokens: dict[str, set[int]]) -> list[str]:
    """Phủ MỌI token-id bằng ít từ nhất (greedy), rare-first tie-break."""
    uncovered = set().union(*word_tokens.values()) if word_tokens else set()
    chosen = []
    remaining = dict(word_tokens)
    while uncovered:
        best = max(remaining, key=lambda w: (len(remaining[w] & uncovered), w))
        gain = remaining[best] & uncovered
        if not gain:
            break
        chosen.append(best)
        uncovered -= gain
        del remaining[best]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frag_oversample", type=int, default=2, help="lần lặp thêm từ fragile-core / pass")
    ap.add_argument("--single_frac", type=float, default=0.08, help="tỉ lệ ảnh 1-từ (bổ trợ ít)")
    ap.add_argument("--sizes", default="4,5,7,8", help="kích thước nhóm (2 dòng)")
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--out", default="metadata_compound_v1.jsonl")
    ap.add_argument("--limit", type=int, default=0, help=">0: giới hạn số nhóm (debug)")
    ap.add_argument("--n_compound", type=int, default=0,
                    help=">0: sinh NHIỀU PASS (mỗi pass = random word/token khác) tới đủ N nhóm compound")
    ap.add_argument("--img_sub", default="images_compound", help="thư mục ảnh con trong data/coverage_v10")
    args = ap.parse_args()
    global IMGSUB
    IMGSUB = args.img_sub
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.sizes.split(",")]
    (OUTDIR / IMGSUB).mkdir(parents=True, exist_ok=True)

    print("[1] tokenizer + diacritic-token coverage…")
    tk = AutoTokenizer.from_pretrained(str(TOK_DIR))
    words = [l.strip() for l in open(LEX) if l.strip()]
    rank = {w: i for i, w in enumerate(words)}
    _c = {}
    def dtok(w):
        if w not in _c:
            _c[w] = frozenset(t for t in tk(w, add_special_tokens=False)["input_ids"]
                              if is_diac(tk.decode([t])))
        return _c[w]
    wt_uc, wt_lc = {}, {}
    for w in words:
        cw, lw = cap(w), w.lower()
        if dtok(cw): wt_uc[cw] = set(dtok(cw))
        if dtok(lw): wt_lc[lw] = set(dtok(lw))
    all_tok = set().union(*wt_uc.values()) | set().union(*wt_lc.values())
    cover_uc = greedy_set_cover(wt_uc)
    cover_lc = greedy_set_cover(wt_lc)
    # inverse: token-id → MỌI từ phủ nó (cho random word/token mỗi pass = đa dạng thật)
    tok2uc, tok2lc = defaultdict(list), defaultdict(list)
    for w, ts in wt_uc.items():
        for t in ts: tok2uc[t].append(w)
    for w, ts in wt_lc.items():
        for t in ts: tok2lc[t].append(w)
    print(f"    diacritic token-ids total={len(all_tok)} | cover_uc={len(cover_uc)} cover_lc={len(cover_lc)}")

    # fragile-core words (oversample)
    s = json.load(open(ELIM))
    frag_words = []
    for t, d in s.items():
        if d.get("status") == "fragile":
            if d.get("fail_word"): frag_words.append(cap(d["fail_word"]))
            for w in d.get("words", [])[:2]: frag_words.append(cap(w))
    frag_words = sorted(set(frag_words))
    print(f"    fragile-core words = {len(frag_words)} (oversample ×{args.frag_oversample})")

    def make_groups(p, si0):
        """Chunk thành nhóm size cycled; KHÔNG bỏ từ — tail <4 gộp vào nhóm trước."""
        gs, i, si = [], 0, si0
        while i < len(p):
            n = sizes[si % len(sizes)]; si += 1
            g = p[i:i + n]; i += n
            if len(g) < 4 and gs:
                gs[-1].extend(g)            # gộp tail vào nhóm trước (không mất từ)
            elif len(g) >= 1:
                gs.append(g)
        return gs, si

    def pass_pools(pass_seed, use_random):
        """1 pass = phủ TOÀN BỘ token (cả 2 case) 1 lần; use_random → chọn từ NGẪU NHIÊN/token
        (đa dạng giữa các pass) thay vì set-cover cố định. + fragile oversample (uppercase)."""
        pr = random.Random(pass_seed)
        if use_random:
            ucp = [pr.choice(tok2uc[t]) for t in tok2uc]
            lcp = [pr.choice(tok2lc[t]) for t in tok2lc]
        else:
            ucp = list(cover_uc); lcp = list(cover_lc)
        ucp = ucp + frag_words * args.frag_oversample
        pr.shuffle(ucp); pr.shuffle(lcp)
        return ucp, lcp

    # CASE-CONSISTENT groups; nếu --n_compound>0 → nhiều PASS (mỗi pass random word/token khác)
    groups = []
    n_target = args.n_compound if args.n_compound > 0 else 0
    p_idx = 0
    while True:
        ucp, lcp = pass_pools(args.seed + 1000 * p_idx, use_random=(n_target > 0))
        ucg, si = make_groups(ucp, p_idx)
        lcg, _ = make_groups(lcp, si)
        groups += ucg + lcg
        p_idx += 1
        if n_target == 0 or len(groups) >= n_target:
            break
    if n_target:
        groups = groups[:n_target]
    rng.shuffle(groups)
    if args.limit:
        groups = groups[:args.limit]
    print(f"    passes={p_idx} | compound groups={len(groups)} (random-word/token={'yes' if n_target else 'no'})")

    # 1-word supplement (bổ trợ ít): fragile words, single line
    n_single = int(len(groups) * args.single_frac / max(1 - args.single_frac, 1e-6))
    singles = rng.sample(frag_words, min(n_single, len(frag_words)))

    print(f"[2] render {len(groups)} compound groups + {len(singles)} single (1-từ bổ trợ)…")
    meta = []
    def emit(words_list, tag, idx):
        n = len(words_list)
        cols = max(1, math.ceil(n / 2)) if n > 1 else 1
        if n == 1:
            lines = [words_list[0]]; layout = "single"
        else:
            lines = [" ".join(words_list[:cols]), " ".join(words_list[cols:])]
            layout = f"cols{cols}"
        phrase = " ".join(words_list)
        img = render_compact_2line(lines, FONT) if n > 1 else render_compact_2line([words_list[0]], FONT, height_frac=0.30)
        fname = f"{tag}_{idx:05d}.jpg"
        img.save(OUTDIR / IMGSUB / fname, quality=95)
        meta.append({"image": f"{IMGSUB}/{fname}",
                     "prompt": build_no_bbox_prompt(phrase, layout)})

    for k, g in enumerate(groups, 1):
        emit(g, "cmp", k)
        if k % 50 == 0: print(f"    {k}/{len(groups)} groups")
    for k, w in enumerate(singles, 1):
        emit([w], "sgl", k)

    rng.shuffle(meta)
    mp = OUTDIR / args.out
    mp.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in meta), encoding="utf-8")

    # coverage audit: every diacritic token appears in some emitted phrase?
    seen = set()
    for m in meta:
        # phrase is in prompt high_level_description; re-derive from image words is overkill — audit pool instead
        pass
    cov_pool = set()
    for g in groups:
        for w in g: cov_pool |= dtok(w)
    miss = all_tok - cov_pool
    print(f"[DONE] {len(meta)} samples ({len(groups)} compound + {len(singles)} single) → {args.out}")
    print(f"       group-size dist: " + ", ".join(f"{n}:{sum(1 for g in groups if len(g)==n)}" for n in sizes))
    print(f"       token coverage in pool: {len(cov_pool)}/{len(all_tok)} | MISSING {len(miss)} {sorted(miss)[:10]}")
    print(f"       images → data/coverage_v10/{IMGSUB}/")


if __name__ == "__main__":
    main()
