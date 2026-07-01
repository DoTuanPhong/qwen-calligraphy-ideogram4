# V10 Guide: 2-Instance Migration qua Google Drive

**Phiên bản:** 6.0, compound/final era.
**Cập nhật:** 2026-06-28.
**Thay thế:** v5.0 wide-target/early-compound era. Đã bổ sung đầy đủ kết quả thực nghiệm cuối cùng, các run phụ và lưu ý an toàn rclone.

## 0. Trạng Thái Hiện Tại (CHỐT)

Mục tiêu migration vẫn là tách workload:

- **Instance A:** training, checkpoint conversion, soup, research probes.
- **Instance B:** inference, visual eval, production image generation.
- **Google Drive:** chỉ làm cầu nối artifact. Không sync base model.

Dự án có **HAI gold checkpoint cuối**, dùng CHUNG kiến trúc rank64 wide-target (6 module):

```text
# Single-word gold (panel fragile60, seed 7000)
experiments/checkpoints/coverage_v10_widetarget_soup567/step-soup_infer.safetensors        # 52/60

# Compound gold = CHECKPOINT PRODUCTION CUỐI (panel Eval28, seed 7000)
experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors   # 4 lỗi / 168 ≈ 97.6%
```

**Eval28 là gate chính cho mục tiêu cuối (câu/cụm nhiều từ).** Single-word 52/60 KHÔNG transfer đủ tốt sang
bố cục nhiều từ, nên panel chính cuối cùng là **compound Eval28** (28 ảnh / 168 từ, 7 ảnh mỗi cỡ 4/5/7/8, 2 dòng).

Tiến trình compound bridge (Eval28, /168 từ, seed 7000, manual word-level count):

| Checkpoint / candidate | lỗi / 168 | Ghi chú |
|---|---:|---|
| `coverage_v10_widetarget_soup567` (single-word gold) | 56 | 52/60 một-từ nhưng compound còn lỗi nhiều |
| `coverage_v10_compound_bridge` (e4) | 26 | VALIDATED: train trực tiếp nhiều-từ → tổng quát hóa layout-binding |
| `coverage_v10_compound_bridge_r2` | 19 | warm-continue, fresh compound seed |
| `coverage_v10_compound_bridge_r3` | 18 | giảm chậm, gần plateau |
| `coverage_v10_compound_soup_e4r2r3` | 13 | soup > từng branch (triệt nhiễu riêng vòng) |
| `coverage_v10_compound_bridge_r4_from_soup` | 15 | solo từ soup không bằng soup |
| **`coverage_v10_compound_soup_e4r2r3r4` (Gold4)** | **6** | 4-way soup, mốc ổn định |
| `coverage_v10_compound_lr2e5_from_gold4` | 6 | LR 2e-5 quá thấp: giữ nguyên mức Gold4, không unlock được plateau |
| `coverage_v10_compound_soup_gold4_lr2e5_3to1` | 7 | soup 75% lr2e5 + 25% Gold4 làm xấu nhẹ |
| `coverage_v10_compound_lr3e5_from_gold4` | 5 | LR 3e-5 = sweet spot, solo new best |
| `coverage_v10_compound_soup_lr3e5_gold4_3to1` | 6 | soup 75/25 kéo lại một số lỗi cũ (`Huyên` x2) |
| **`coverage_v10_compound_soup_lr3e5_gold4_9to1` FINAL** | **4** | 90% lr3e5 + 10% Gold4 = best cuối cùng cho luận văn |

Lỗi còn lại của compound gold (4): `Hấn→Hẩn`, `Chịt→Chít`, `Huyên→Huyện`, `Dôi→Dồi`.

Held-out test sau khi chốt checkpoint:

- Panel: `experiments/results/coverage_v10_eval/compound_test28_heldout_seed20260630_groups.json`
- Render: `experiments/results/coverage_v10_eval/compound_test28_heldout_seed20260630_soup_lr3e5_gold4_9to1/`
- Kết quả chấm tay: **3 lỗi / 168 ≈ 98.2%** (`Hôn→Hon`, `Nữ→Nử`, `Ghẹ→Ghệ`).
- Vai trò: **Test28 = held-out test**, còn Eval28 là validation/model-selection panel chính trong quá trình chọn checkpoint.

Kết luận vận hành:

- **Pivot đúng = train trực tiếp compound (4/5/7/8 từ, 2 dòng căn giữa, prompt no-bbox).** Baseline compound 56 lỗi → final 4 lỗi.
- **Soup (checkpoint averaging) là lever chính cuối:** trung-bình các branch cùng basin → giữ skill chung, triệt nhiễu
  riêng từng vòng. Nhưng chỉ tốt khi tỉ lệ đúng: Gold4 25% kéo về lỗi cũ; Gold4 10% đủ làm regularizer nhẹ.
- **LR 3e-5 là sweet spot vùng cuối** (5e-5 quá mạnh cho follow-up gold, 2e-5 quá yếu/không unlock).
- **Count manual là gate.** Eval28 cho compound, fragile60 cho single-word. OCR/classifier chỉ advisory.

Nguồn bối cảnh:

- `docs/v10_widetarget_session_log.md` (§2d compound bottleneck, §2e final compound gold)
- `docs/thesis/Final_Thesis_V10_VI.md` (luận văn chốt)

---

## 1. TL;DR Hằng Ngày

Sau khi A train xong một branch compound:

```bash
cd /workspace/qwen_calligraphy_lora

# 1. Convert official training checkpoint -> inference checkpoint (rank64).
python3 experiments/scripts/v10/convert_official_lora_to_infer.py \
  experiments/checkpoints/<branch>/step-11820.safetensors \
  experiments/checkpoints/<branch>/step-11820_infer.safetensors \
  --rank 64 --alpha 64

# 2. Push gold cuối lên GDrive.
RDST=gdrive:qwen_calligraphy_v10
rclone copy experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/ \
  $RDST/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/ --progress
```

Trên B (inference bằng compound gold):

```bash
cd /workspace/qwen_calligraphy_lora
RSRC=gdrive:qwen_calligraphy_v10

rclone copy $RSRC/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/ \
  experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/ --progress

python3 experiments/scripts/v10/infer_v10_lora.py \
  --lora_ckpt experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors \
  --lora_targets attention.qkv attention.o feed_forward.w1 feed_forward.w2 feed_forward.w3 adaln_modulation \
  --lora_rank 64 --lora_alpha 64 \
  --hard_cases "Tâm Đức Trí Nhân" "Phúc Thọ Khang Vượng" "An Khang Thịnh Vượng" \
  --output_dir experiments/results/infer_compound_gold \
  --num_inference_steps 48 --cfg_scale 7.0 --seed 7000
```

Checkpoint wide-target/compound đều ~**424MB** (rank64, 6 module), không phải LoRA attn-only ~120MB.

---

## 2. Base Model Và Prerequisite

Không sync base model qua GDrive.

| Instance | Cần model | Path | Ghi chú |
|---|---|---|---|
| A training | BF16 local | `models/ideogram-4-bf16-local/` | dùng bởi `run_widetarget_gentle.sh` |
| B inference/eval | FP8 Ideogram4 | `models/ideogram-ai/ideogram-4-fp8/` | đủ cho infer/probe/render ảnh |

Instance B cần đủ 5 thư mục:

```bash
ls models/ideogram-ai/ideogram-4-fp8/{transformer,unconditional_transformer,text_encoder,vae,tokenizer}
```

Nếu thiếu:

```bash
hf download ideogram-ai/ideogram-4-fp8 --local-dir models/ideogram-ai/ideogram-4-fp8
```

Inference full CFG cần cả `transformer` và `unconditional_transformer`. Không dùng `--skip_unconditional` cho
production (CFG sẽ lệch; từng làm raw Vượng→Hán trong smoke).

---

## 3. Artifact Cần Sync Qua GDrive

Drive root đề xuất: `gdrive:qwen_calligraphy_v10/`

> [!WARNING]
> **KHÔNG DÙNG `rclone sync`:** Do các checkpoint lịch sử (ví dụ `v10_rgdpo_round*`) đã được dọn dẹp ở local để tiết kiệm đĩa, việc chạy `rclone sync` sẽ xóa vĩnh viễn các checkpoint này trên Google Drive. Luôn luôn sử dụng `rclone copy` để tránh mất mát dữ liệu!

| Artifact | Local path | Tier | Vai trò |
|---|---|---|---|
| **compound gold (infer)** | `experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors` | bắt buộc | B inference/eval production |
| compound gold (official) | `…/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup.safetensors` | nên có | resume/soup tiếp |
| single-word gold | `experiments/checkpoints/coverage_v10_widetarget_soup567/step-soup{,_infer}.safetensors` | nên có | so single-word, ingredient soup |
| compound branch ingredients | `…/coverage_v10_compound_bridge{,_r2,_r3}/step-11820.safetensors`, `…_soup_e4r2r3r4/step-soup.safetensors`, `…_lr3e5_from_gold4/step-11820.safetensors`, `…_soup_lr3e5_gold4_3to1/step-soup.safetensors`, `…_soup_gold4_lr2e5_3to1/step-soup.safetensors` | nên có | tái soup / phân tích |
| Eval28 panel | `experiments/results/coverage_v10_eval/compound_eval28_groups.json` | bắt buộc | gate compound chuẩn |
| session log | `docs/v10_widetarget_session_log.md` | nên có | handoff trạng thái |
| luận văn | `docs/thesis/Final_Thesis_V10_VI.md` | nên có | tài liệu chốt |
| eval folder cuối | `experiments/results/coverage_v10_eval/compound_eval28_soup_lr3e5_gold4_9to1/` | optional | ảnh chấm tay final |
| **dataset packaged** | `data/coverage_v10.tar.gz` | nên có | toàn bộ ảnh/metadata dataset V10 đã đóng gói |
| **eval results packaged** | `experiments/results/coverage_v10_eval.tar.gz` | nên có | toàn bộ thư mục kết quả đánh giá đã đóng gói |

Code (scripts) sync bằng **git**, không bằng GDrive.

---

## 4. Checkpoint Format Và Scale

Hai dạng checkpoint:

| Dạng | File | Dùng cho |
|---|---|---|
| official training | `step-*.safetensors` (training: step-11820; soup: step-soup) | warm/resume training, convert, soup |
| inference | `step-*_infer.safetensors` | `infer_v10_lora.py`, `render_compound_eval.py`, `probe_seed_variance.py` |

Lý do phải convert:

- Training official dùng standard LoRA scale `alpha/rank = 1.0`.
- Inference wrapper dùng rsLoRA scale `alpha/sqrt(rank)`.
- Convert scale `lora_B` để effective delta khớp (rank64 → ×0.125).

```bash
python3 experiments/scripts/v10/convert_official_lora_to_infer.py \
  <IN.safetensors> <OUT_infer.safetensors> --rank 64 --alpha 64
```

Cả single-word soup, compound branch, lẫn compound soup đều rank64/alpha64 → convert giống nhau. Soup checkpoint
(`step-soup.safetensors`) cũng là official-format (key `.default.weight`, scale 1.0) nên warm-train/convert như branch thường.

---

## 5. Inference Trên Instance B

Wide-target/compound phải inject đủ **6 module**:

```text
attention.qkv
attention.o
feed_forward.w1
feed_forward.w2
feed_forward.w3
adaln_modulation
```

Smoke compound gold:

```bash
python3 experiments/scripts/v10/infer_v10_lora.py \
  --lora_ckpt experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors \
  --lora_targets attention.qkv attention.o feed_forward.w1 feed_forward.w2 feed_forward.w3 adaln_modulation \
  --lora_rank 64 --lora_alpha 64 \
  --hard_cases "Tâm Đức Trí Nhân" "Phúc Thọ Khang Vượng" \
  --output_dir experiments/results/smoke_compound_gold \
  --num_inference_steps 48 --cfg_scale 7.0 --seed 7000
```

Quan trọng:

- **Prompt production là no-bbox JSON** qua `build_no_bbox_prompt(phrase, layout)`. Nhiều từ → `layout="cols{ceil(n/2)}"`
  (2 dòng: 4→cols2, 5/6→cols3, 7/8→cols4). bbox per-word/per-row KÉM (DiffSynth local không parse bbox thành hard-constraint).
- `--use_bbox` chỉ debug legacy.
- Manual visual review là ground truth; OCR/classifier chỉ advisory với dấu nhỏ/nét thư pháp.

---

## 6. Eval Trên Instance B

### 6.1. Compound Eval28 (GATE CHÍNH)

Panel chuẩn: `experiments/results/coverage_v10_eval/compound_eval28_groups.json`
(28 nhóm = 7 mỗi cỡ 4/5/7/8 từ, vocab fragile60, seed 999; render seed 7000 → 168 từ).

```bash
cd /workspace/qwen_calligraphy_lora
python3 experiments/scripts/v10/render_compound_eval.py \
  --groups experiments/results/coverage_v10_eval/compound_eval28_groups.json \
  --lora_ckpt experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors \
  --out_dir experiments/results/coverage_v10_eval/compound_eval28_<tag> \
  --seed 7000
```

Ảnh ra tên `NN_n{size}_{words}_s7000.jpg` → đếm lỗi cấp từ /168. Gold cuối = **4 lỗi**.

### 6.2. Single-word fragile60 (gate phụ)

Vẫn dùng được cho single-word, nhưng KHÔNG còn là gate cuối (single-word không transfer đủ sang câu dài):

```bash
WORDS=$(python3 -c "from pathlib import Path;print(','.join(w.strip() for l in Path('experiments/results/coverage_v10_eval/fragile_subset60.txt').read_text().splitlines() for w in l.split(',') if w.strip()))")
SEED_BASE=7000 SEEDS=1 MAKE_LINKS=1 SORT_WORDS=1 PROBE_WORDS="$WORDS" \
LORA_TARGETS="attention.qkv,attention.o,feed_forward.w1,feed_forward.w2,feed_forward.w3,adaln_modulation" \
LORA_STRICT=0 LORA_RANK=64 LORA_ALPHA=64 \
LORA_CKPT=experiments/checkpoints/coverage_v10_widetarget_soup567/step-soup_infer.safetensors \
PROBE_OUT=experiments/results/coverage_v10_eval/soup567_scan_B \
PYTHONPATH=$PWD/DiffSynth-Studio \
python3 experiments/scripts/v10/probe_seed_variance.py
```

Sanity (cả 2 path): `injected 205 modules | missing=0 unexpected=0`.

---

## 7. Training Tiếp Trên Instance A (compound bridge + soup)

### 7.1. Tạo dữ liệu compound

```bash
python3 experiments/scripts/v10/build_compound_dataset.py \
  --n_compound 2808 --frag_oversample 1 --single_frac 0.05 --sizes 4,5,7,8 \
  --seed <SEED_MỚI_MỖI_VÒNG> --img_sub images_compound_<tag> \
  --out metadata_compound_2808_<tag>.jsonl
```

- 2808 nhóm 4/5/7/8 từ × 2 dòng + ~5% single, phủ **406/406 diacritic token-id cả HOA+THƯỜNG**, multi-pass random
  word/token (đa dạng thật). Target = compact-centered 2-line font (khớp layout model, đã verify). `frag_oversample=1`
  (nhẹ — lỗi compound chủ yếu ở token STABLE/layout, không phải fragile).

### 7.2. Train branch (warm-continue, gated)

```bash
# Warm từ OFFICIAL checkpoint (step-11820 hoặc step-soup), KHÔNG dùng _infer để train.
bash experiments/scripts/v10/run_widetarget_gentle.sh \
  experiments/checkpoints/coverage_v10_compound_soup_e4r2r3r4/step-soup.safetensors \
  experiments/checkpoints/coverage_v10_compound_<tag> \
  4 3e-5 metadata_compound_2808_<tag>.jsonl
```

- 4 epoch, **LR 3e-5** (sweet spot vùng cuối; 5e-5 cho các vòng đầu khi còn xa đỉnh). Lưu 9 intermediate.
- Convert → `render_compound_eval.py` trên Eval28 → đếm.

### 7.3. Soup checkpoint (lever chính cuối)

Trộn branch mới với base soup theo `(BASE_WEIGHT*base + new)/(BASE_WEIGHT+1)`:

```bash
# BASE_WEIGHT=0.1111 ≈ 90% new + 10% base (đã cho final = 4 lỗi)
BASE_WEIGHT=0.1111111111 experiments/scripts/v10/postprocess_compound_branch.sh \
  <train_tmux_session_hoặc_no_training_session_for_manual_soup> \
  experiments/checkpoints/coverage_v10_compound_lr3e5_from_gold4/step-11820.safetensors \
  experiments/checkpoints/coverage_v10_compound_soup_e4r2r3r4/step-soup.safetensors \
  experiments/checkpoints/coverage_v10_compound_soup_<tag> \
  experiments/results/coverage_v10_eval/compound_eval28_soup_<tag>
```

Soup nhiều branch cùng basin (e4+r2+r3 = 13; +r4 = Gold4 = 6) cũng làm tay bằng cách average đều
`step-11820.safetensors` của các branch.

### 7.4. Quy tắc quyết định (gate Eval28)

- Branch mới `< best`: nhận / soup vào pool.
- `= best` hoặc hơi cao: giữ soup, thử weighted-soup (Gold4 10-25%) tìm tỉ lệ tốt.
- Cao rõ hoặc nặng-attractor bùng (Huyên/Hoắc +nặng): rollback, KHÔNG chain mù.
- **Không chain `soup→r4→r5→…` mà chưa eval từng vòng** — LR 5e-5 đủ mạnh kéo checkpoint lệch.

### 7.5. Lỗi môi trường

```text
RuntimeError: Configured CUDA binary not found ... libbitsandbytes_cuda132.so
```

Là warning bitsandbytes lúc khởi động, KHÔNG fatal (training vẫn chạy/ lưu checkpoint bình thường).

---

## 8. Bối Cảnh Chẩn Đoán (vì sao hướng này)

- **Single-word gold soup567 = 52/60** đạt bằng weight-soup các vòng wide-target (warm-continue high-variance:
  46/48/47 → soup 52). Warm-continue = sinh SAMPLE; soup = PRODUCTION.
- **Compound bottleneck:** lỗi câu dài đa số ở token STABLE (mất binding-dấu khi crowding), KHÔNG phải fragile.
  ⇒ nút thắt = multi-word/layout glyph-binding; single-word chỉ là proxy.
- **Compound bridge VALIDATED:** train trực tiếp nhiều-từ tổng quát hóa layout-binding trên Eval28 held-out
  (56→26 ngay vòng đầu, không trùng exact group với train).
- Qwen3-VL probe: KHÔNG train Qwen global; nút thắt nghiêng DiT/glyph-binding (wide-target + replay + soup).

---

## 9. Gotchas Bắt Buộc Nhớ

1. **Tuyệt đối không dùng `rclone sync`:** Hãy luôn sử dụng `rclone copy` để tránh việc xóa nhầm các checkpoint lịch sử đã dọn dẹp ở local nhưng vẫn cần lưu trữ trên Drive.
2. **Wide-target/compound = 6 module**, không phải attn-only. Luôn truyền đủ khi infer/eval.
3. **Inference dùng `_infer.safetensors`; training/soup dùng official `step-*.safetensors`.** Không lẫn.
4. **Gate cuối là compound Eval28** (manual /168), không phải single-word.
5. **Soup chỉ tốt khi đúng tỉ lệ.** Gold4 25% kéo về lỗi cũ; 10% là regularizer nhẹ. LR 3e-5 là sweet spot cuối.
6. **Prompt production = no-bbox JSON**, layout `cols{ceil(n/2)}` cho nhiều từ. Đừng đổi prompt khi so checkpoint.
7. **Seed phải ghi rõ.** Mọi so sánh cuối dùng seed 7000. (Lịch sử: wide r1-4 eval seed 1000, r5+/soup seed 7000.)
8. **Base model không sync.** GDrive chỉ chứa LoRA/artifact/docs.
9. **Manual count là gate.** OCR/classifier có blind spot dấu nhỏ.
10. **B cần 48GB+ VRAM cho full CFG.** Full load ~35GB; skip unconditional chỉ để debug.
11. **`--output_path` của `run_widetarget_gentle.sh` là tương đối** → luôn prefix `experiments/checkpoints/` kẻo ghi vào repo-root.
12. **Cách unpack tệp đóng gói trên Instance B:**
    ```bash
    # Giải nén dataset
    tar -xzf data/coverage_v10.tar.gz -C data/
    # Giải nén kết quả eval
    tar -xzf experiments/results/coverage_v10_eval.tar.gz -C experiments/results/
    ```

---

## 10. Pointers

- Session log: `docs/v10_widetarget_session_log.md`
- Luận văn chốt: `docs/thesis/Final_Thesis_V10_VI.md`
- Inference: `experiments/scripts/v10/infer_v10_lora.py`
- Convert official→infer: `experiments/scripts/v10/convert_official_lora_to_infer.py`
- Train (wide-target & compound): `experiments/scripts/v10/run_widetarget_gentle.sh`
- Tạo dữ liệu compound: `experiments/scripts/v10/build_compound_dataset.py`
- Render eval compound: `experiments/scripts/v10/render_compound_eval.py`
- Soup + eval branch: `experiments/scripts/v10/postprocess_compound_branch.sh`
- Probe fragile60 single-word: `experiments/scripts/v10/probe_seed_variance.py`
- Eval28 panel: `experiments/results/coverage_v10_eval/compound_eval28_groups.json`
- Fragile panel: `experiments/results/coverage_v10_eval/fragile_subset60.txt`
