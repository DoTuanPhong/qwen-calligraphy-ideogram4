"""V10 Phase 4 — dựng mask VÙNG-DẤU per-từ cho region-masked loss (v2: per-char Unicode-gated).

v1 (gap+global-zone) FAIL: bỏ sót ê (mũ dính thân), ơ/ư (horn vai-phải), hỏi trong từ có chữ HOA cao.
v2: xử lý TỪNG ký tự; CHỈ ký-tự-CÓ-dấu mới tô zone, theo ĐÚNG loại dấu (biết từ NFD), trong ink RIÊNG
của ký tự đó (per-char box → tự xử lý chữ-hoa-cao & hoa/thường). KHÔNG bỏ sót dấu là tiêu chí số 1.

Zone theo loại dấu (trong box ink của ký tự):
  - TOP-mark (thanh-trên huyền/sắc/hỏi/ngã HOẶC mũ/trăng): top 45% ink.
  - HORN (ơ/ư): thêm cột phải 35% × rows trên 60% (vai-phải-trên).
  - NẶNG: thêm rows dưới 22% (chấm dưới).
Union → dilate → lưu PNG L-mode 1024², key = NFC-word.

Usage:
  python3 experiments/scripts/v10/build_diacritic_masks.py            # toàn bộ từ có GT trong images/
  python3 experiments/scripts/v10/build_diacritic_masks.py --verify   # + overlay nhóm dấu (_check/)
"""
import argparse
import sys
import unicodedata
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FONT_PATH = str(PROJECT_ROOT / "assets/fonts/Thu_Phap_Thanh_Cong_Unicode.ttf")
SIZE = (1024, 1024)
_SZ = SIZE[0]

def calculate_optimal_font(text: str, font_path: str, layout: str, img_size: tuple[int, int], max_fs: int = 1200) -> ImageFont.FreeTypeFont:
    W, H = img_size
    m = 0.15  # safe margin
    mw, mh = W * (1 - m * 2), H * (1 - m * 2)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    fs = max_fs

    while fs > 50:
        try:
            font = ImageFont.truetype(font_path, fs)
        except Exception:
            return ImageFont.load_default()

        if layout == "vertical":
            words = text.split()
            th, maxww = 0, 0
            sp = fs * 0.2
            for w in words:
                b = dummy.textbbox((0, 0), w, font=font)
                th += b[3] - b[1]
                maxww = max(maxww, b[2] - b[0])
            if len(words) > 1:
                th += (len(words) - 1) * sp
            if th < mh and maxww < mw:
                return font
        else:
            b = dummy.textbbox((0, 0), text, font=font)
            if (b[2] - b[0]) < mw and (b[3] - b[1]) < mh:
                return font
        fs -= 10
    return ImageFont.load_default()

def draw_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, img_size: tuple[int, int], layout: str, color: str) -> None:
    W, H = img_size
    if layout == "vertical" and len(text.split()) > 1:
        words = text.split()
        ls = font.size * 0.2
        th = sum(
            draw.textbbox((0, 0), w, font=font)[3] - draw.textbbox((0, 0), w, font=font)[1]
            for w in words
        )
        th += (len(words) - 1) * ls
        cy = (H - th) / 2
        for w in words:
            bbox = draw.textbbox((0, 0), w, font=font)
            ww = bbox[2] - bbox[0]
            x = (W - ww) / 2 - bbox[0]
            y = cy - bbox[1]
            draw.text((x, y), w, font=font, fill=color)
            cy += (bbox[3] - bbox[1]) + ls
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (W - w) / 2 - bbox[0]
        y = (H - h) / 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=color)

def render(word: str, font_path: str = DEFAULT_FONT_PATH) -> Image.Image:
    img = Image.new("RGB", SIZE, color="#FFFFFF")
    draw = ImageDraw.Draw(img)
    font = calculate_optimal_font(word, font_path, "horizontal", SIZE)
    draw_text(draw, word, font, SIZE, "horizontal", "#000000")
    return img

TONE_TOP = {0x0300, 0x0301, 0x0309, 0x0303}   # huyền sắc hỏi ngã (đặt trên)
MODIF_TOP = {0x0302, 0x0306}                  # mũ(circumflex) trăng(breve) — đặt trên
HORN = 0x031B                                 # móc/horn — vai phải
NANG = 0x0323                                 # nặng — chấm dưới
FONT = DEFAULT_FONT_PATH


def char_marks(ch: str):
    """(has_top, has_horn, has_nang) từ NFD của 1 ký tự."""
    cps = {ord(c) for c in unicodedata.normalize("NFD", ch)}
    has_top = bool(cps & (TONE_TOP | MODIF_TOP))
    return has_top, (HORN in cps), (NANG in cps)


def _render_font(text: str, font) -> np.ndarray:
    """Centered horizontal render (KHỚP draw_text dataset) với font cho trước."""
    im = Image.new("L", (_SZ, _SZ), 255); d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((_SZ - tw) / 2 - bb[0], (_SZ - th) / 2 - bb[1]), text, font=font, fill=0)
    return np.asarray(im, np.uint8)


def build_mask(word: str, dilate_px: int = 8, thr: int = 128) -> np.ndarray:
    """Per-char Unicode-gated zone mask {0,255}.

    Window RỘNG RÃI (ưu tiên phủ-đủ hơn ôm-sát, do font brush dấu chảy ngang):
      - right-extension cho top-mark (huyền/ngã kéo sang ký tự sau),
      - min-width (dấu trên 'i' hẹp không bị cắt),
      - top-zone cao (0.55) để không cắt dấu chồng / drift placement.
    """
    font = calculate_optimal_font(word, FONT, "horizontal", (_SZ, _SZ))
    A = _render_font(word, font)
    ink = A < thr
    out = np.zeros((_SZ, _SZ), bool)
    if not ink.any():
        return out.astype(np.uint8)
    d = ImageDraw.Draw(Image.new("L", (_SZ, _SZ)))
    bb = d.textbbox((0, 0), word, font=font)
    x0 = (_SZ - (bb[2] - bb[0])) / 2 - bb[0]
    fsz = getattr(font, "size", 120)
    min_w = 0.75 * fsz                                   # đảm bảo bề rộng tối thiểu (cho 'i')

    for k, ch in enumerate(word):
        if ch.isspace():
            continue
        has_top, has_horn, has_nang = char_marks(ch)
        if not (has_top or has_horn or has_nang):
            continue                                     # ký tự không dấu → bỏ
        xa = x0 + font.getlength(word[:k])
        xb = x0 + font.getlength(word[:k + 1])
        cw = max(xb - xa, 1.0)
        # window: pad trái + right-extension (dấu chảy phải) + min-width
        left = xa - 0.12 * cw
        ext = 1.0 * cw if has_top else (0.7 * cw if has_horn else 0.3 * cw)  # top & horn chảy/cong phải
        right = xb + ext
        if right - left < min_w:
            extra = (min_w - (right - left))
            left -= extra * 0.35; right += extra * 0.65   # nới phải nhiều hơn (dấu lệch phải)
        xs0 = max(0, int(left)); xs1 = min(_SZ, int(right))
        if xs1 - xs0 < 2:
            continue
        sub = ink[:, xs0:xs1]
        rows = np.where(sub.any(1))[0]
        if len(rows) == 0:
            continue
        cy0, cy1 = int(rows.min()), int(rows.max()); h = max(cy1 - cy0, 1)
        if has_top:                                      # top 55% ink (cao hơn → không cắt chồng/đuôi)
            r1 = cy0 + int(0.55 * h)
            out[cy0:r1, xs0:xs1] |= sub[cy0:r1]
        if has_horn:                                     # vai phải-trên: nới TRÁI, bớt CAO (ôm sát curl)
            base_ch = "".join(c for c in unicodedata.normalize("NFD", ch)
                              if not unicodedata.combining(c)).lower()
            horn_ry = 0.68 if base_ch == "o" else 0.50    # ơ (base tròn) thấp hơn ư
            rx0 = xs0 + int(0.32 * (xs1 - xs0)); ry1 = cy0 + int(horn_ry * h)
            out[cy0:ry1, rx0:xs1] |= ink[cy0:ry1, rx0:xs1]
        if has_nang:                                     # chấm dưới 25%
            r0 = cy1 - int(0.25 * h)
            out[r0:cy1 + 1, xs0:xs1] |= sub[r0:cy1 + 1]

    m = Image.fromarray((out.astype(np.uint8)) * 255)
    if dilate_px > 0:
        m = m.filter(ImageFilter.MaxFilter(dilate_px * 2 + 1))
    return np.asarray(m, np.uint8)


def _words_from_images(imgdir: Path):
    seen, words = set(), []
    for f in sorted(imgdir.iterdir()):
        if f.suffix.lower() not in (".jpg", ".png"):
            continue
        b = f.stem
        w = unicodedata.normalize("NFC", b.split("_", 1)[1] if "_" in b else b)
        if w not in seen:
            seen.add(w); words.append(w)
    return words


# nhóm overlay kiểm tra đúng các ca fail + đại diện mọi loại dấu
CHECK = {
    "mu_e_o (ê/ô)": ["Bêu", "Dôi", "Cồ", "Diệm"],
    "horn (ơ/ư)": ["Cưu", "Bưng", "Chưởng", "Dượng", "Cữ"],
    "hoi_capital (Hỉ)": ["Hỉ", "Hiểm", "Cỏi", "Chẻ"],
    "nang": ["Ạt", "Cược", "Cật", "Chực"],
    "nga": ["Cưỡi", "Gẫy", "Ngỗng", "Giễu"],
    "trang_a (ă)": ["Chằng", "Gặng", "Bặt"],
    "stacked": ["Cưỡi", "Dẫu", "Khuỷu", "Ngỗng"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(PROJECT_ROOT / "data/coverage_v10/images"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/coverage_v10/masks_diacritic"))
    ap.add_argument("--dilate", type=int, default=6)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    words = _words_from_images(Path(args.images))
    print(f"[build] {len(words)} unique words (scope = images/)")

    empty_with_diacritic = []
    cov = []
    for i, w in enumerate(words):
        m = build_mask(w, args.dilate)
        Image.fromarray(m).save(outdir / f"{w}.png")
        cov.append((m > 0).mean())
        # audit: từ CÓ dấu mà mask rỗng = MISS
        if m.max() == 0 and any(any(char_marks(c)) for c in w):
            empty_with_diacritic.append(w)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(words)}")
    print(f"[DONE] {len(words)} masks → {outdir}")
    print(f"  coverage frac: mean={np.mean(cov):.4f} median={np.median(cov):.4f} max={np.max(cov):.4f}")
    print(f"  ⚠ MISS (có dấu nhưng mask rỗng): {len(empty_with_diacritic)} "
          f"{empty_with_diacritic[:15]}")

    if args.verify:
        vdir = outdir / "_check"; vdir.mkdir(exist_ok=True)
        for grp, ws in CHECK.items():
            for w in ws:
                try:
                    m = build_mask(w, args.dilate)
                    arr = np.asarray(render(w).convert("RGB")).copy()
                    edge = np.asarray(Image.fromarray(m).filter(ImageFilter.FIND_EDGES), np.uint8) > 0
                    arr[edge] = [255, 0, 0]
                    Image.fromarray(arr).save(vdir / f"{w}.png")
                except Exception as e:
                    print("  skip", w, e)
        print(f"  overlays → {vdir}  (groups: {list(CHECK)})")


if __name__ == "__main__":
    main()
