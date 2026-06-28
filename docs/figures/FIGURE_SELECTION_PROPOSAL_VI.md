# De Xuat Hinh Cho Luan Van

Tai lieu nay doi chieu truc tiep voi danh muc hinh trong `docs/thesis/Final_Thesis_V10_EN.md`. Ban cu cua file nay moi tap trung vao cac anh so sanh ket qua, nen thieu mot nhom hinh dang so do/diagram da duoc dua vao thesis: Hinh 1.1, 2.1, 2.2, 3.1 va 3.4. Ngoai ra, Hinh 3.2 va 3.3 khong con la hinh bo sung tuy chon nua, vi ban thesis hien tai da dung chung trong noi dung chinh.

## Doi Chieu Voi Thesis Hien Tai

| Hinh | Ten file | Trang thai | Vai tro trong thesis |
|---|---|---|---|
| Hinh 1.1 | `fig_1_1_research_pipeline.png` | READY | So do tong quan quy trinh nghien cuu tu baseline, probe, LoRA, soup den compound training |
| Hinh 1.2 | `fig_1_2_competitor_baseline_comparison.png` | DRAFT | So sanh font may, cong cu black-box, Qwen Image, ERNIE Image, Ideogram4 goc va checkpoint de xuat; o commercial hien la placeholder |
| Hinh 1.3 | `fig_1_3_base_model_capability_comparison.png` | READY | So sanh nang luc goc cua Qwen Image, ERNIE Image va Ideogram4 truoc fine-tuning |
| Hinh 2.1 | `fig_2_1_ideogram4_architecture.png` | READY | So do kien truc Ideogram4 dung trong thesis: Qwen3-VL, DiT LoRA va VAE |
| Hinh 2.2 | `fig_2_2_qwen3vl_multilayer_conditioning.png` | READY | So do conditioning nhieu layer Qwen3-VL vao DiT |
| Hinh 3.1 | `fig_3_1_widetarget_lora_injection_points.png` | READY | So do cac diem gan LoRA trong DiT |
| Hinh 3.2 | `fig_3_2_no_bbox_vs_bbox_layout_comparison.png` | READY | So sanh no-bbox, bbox rong va bbox theo dong cho compound prompt |
| Hinh 3.3 | `fig_3_3_font_reference_vs_model_generation.jpg` | READY | So sanh font-rendered reference voi layout anh model sinh |
| Hinh 3.4 | `fig_3_4_result_progression.png` | READY | Bieu do tien trinh ket qua tu attention-only den final compound checkpoint |
| Hinh 3.5 | `fig_3_5_single_word_before_after_examples.png` | READY | Vi du truoc/sau tren cac tu tieng Viet co dau kho |
| Hinh 3.6 | `fig_3_6_compound_eval28_before_after.png` | READY | So sanh Eval28 giua `soup567` va checkpoint cuoi |
| Hinh 3.7 | `fig_3_7_remaining_error_cases.png` | READY | Bon loi con lai cua checkpoint compound tot nhat |
| Hinh 3.8 | `fig_3_8_calligraphy_with_base_model_capability.png` | READY | Kiem tra kha nang ket hop chu thu phap voi nang luc sinh anh nen cua Ideogram4 |

## Nhom Hinh Vua Tao Bang Script

Cac hinh trong nhom nay da duoc tao nhanh bang script `docs/thesis/figures/make_missing_figures.py`. Neu can thay doi style, so lieu hoac anh nguon, nen sua script va render lai de giu tinh tai lap.

### Hinh 1.1. So do tong quan quy trinh nghien cuu

**Ten file hien co:** `fig_1_1_research_pipeline.png`

Hinh nay nen la mot pipeline ngang hoac doc, dung de giai thich logic cua toan bo thesis. Noi dung nen gom cac khoi:

`Baseline observation -> tokenizer/signal probes -> attention-only LoRA -> wide-target LoRA -> checkpoint soup -> compound dataset -> compound bridge -> final Eval28`.

Nen them mot nhanh nho the hien viec pivot tu Qwen Image/ERNIE Image sang Ideogram4 dua tren bang chung thuc nghiem. Hinh nay khong can anh render that; co the ve bang PowerPoint, draw.io, Mermaid roi export PNG. Vi no nam o Chuong 1, hinh phai doc duoc ngay ca voi nguoi chua biet cac checkpoint r5/r6/r7.

### Hinh 1.2. So sanh cac mo hinh va cong cu nen

**Ten file hien co:** `fig_1_2_competitor_baseline_comparison.png`

Hinh nay can giai thich vi sao de tai khong chi dung truc tiep font may, cong cu thuong mai, Qwen Image, ERNIE Image hoac Ideogram4 goc. Bo cuc tot nhat la bang 2 hang x 3 cot, gom:

`font may`, `black-box commercial tool`, `Qwen Image`, `ERNIE Image`, `Ideogram4 goc`, `checkpoint compound cuoi`.

Ban hien tai da ghep anh co san trong repo cho font may, Qwen Image, ERNIE Image, Ideogram4 goc va checkpoint de xuat. Rieng o commercial black-box hien la placeholder vi repo chua co anh nguon that; truoc khi nop ban cuoi nen thay o nay bang anh render that neu ban muon dung doi thu thuong mai trong panel hinh.

### Hinh 2.1. So do kien truc Ideogram4 trong thesis

**Ten file hien co:** `fig_2_1_ideogram4_architecture.png`

Hinh nay nen trinh bay cac thanh phan chinh:

`Prompt/JSON prompt -> Qwen3-VL text encoder (frozen) -> llm_cond_norm + llm_cond_proj -> Ideogram4 DiT with LoRA adapters -> VAE -> image`.

Nen danh dau phan nao frozen, phan nao co LoRA, va dau ra nao la anh 1024x1024. Muc tieu cua hinh la giup nguoi doc hieu vi sao thesis khong train Qwen3-VL ma tap trung vao DiT/glyph binding.

### Hinh 2.2. Conditioning nhieu layer Qwen3-VL vao DiT

**Ten file hien co:** `fig_2_2_qwen3vl_multilayer_conditioning.png`

Hinh nay nen tap trung vao phan text-conditioning: Qwen3-VL co nhieu hidden-state taps, cac layer duoc concat, sau do qua normalization/projection truoc khi vao DiT. Nen ghi ro day la ly do thesis co the probe `tap-space` va `proj-space` de kiem tra tin hieu dau tieng Viet.

Mot bo cuc de doc:

`Prompt span -> Qwen3-VL layers L0...L35 -> 13 tap layers -> concat -> llm_cond_norm -> llm_cond_proj -> DiT conditioning tokens`.

### Hinh 3.1. Diem gan wide-target LoRA trong DiT

**Ten file hien co:** `fig_3_1_widetarget_lora_injection_points.png`

Hinh nay nen la so do mot DiT block, trong do highlight cac module:

`attention.qkv`, `attention.o`, `feed_forward.w1`, `feed_forward.w2`, `feed_forward.w3`, `adaln_modulation`.

Nen dung mau khac nhau de tach attention, feed-forward va adaLN. Caption trong thesis da noi rang attention-only khong du, nen hinh nay can lam ro viec wide-target LoRA tac dong vao nhieu kenh hon de anh huong glyph geometry.

### Hinh 3.4. Bieu do tien trinh ket qua

**Ten file hien co:** `fig_3_4_result_progression.png`

Hinh nay nen la bieu do duong hoac bieu do cot ket hop, tom tat hai truc ket qua:

- Single-word fragile60: attention-only plateau 32-39, wide r3/r4/r5/r6/r7/r8, `soup567=52`, `soup5679=52`.
- Compound Eval28: `soup567 baseline=56 errors`, e4=26, r2=19, r3=18, `soup_e4r2r3=13`, Gold4=6, lr3e5=5, final `soup_lr3e5_gold4_9to1=4`.

Nen can than ghi ro don vi cua hai phan: single-word la "correct/60" con compound la "errors/168". De tranh gay nham lan, co the tach thanh hai subplot: subplot tren la single-word accuracy, subplot duoi la compound error count.

## Nhom Hinh Da Co Va Dang Dung Trong Thesis

### Hinh 1.3. So sanh nang luc goc cua ba mo hinh open-weight

**Ten file hien co:** `fig_1_3_base_model_capability_comparison.png`

Hinh nay tra loi cau hoi vi sao Ideogram4 duoc chon lam base model cuoi thay vi tiep tuc voi Qwen Image hoac ERNIE Image. Hinh hien dung prompt ngan:

`Traditional Vietnamese calligraphy written in black ink on white paper. The text says "An Khang Thịnh Vượng".`

Trong lan render prompt ngan giong nhau, Ideogram4 goc tra ve placeholder cua safety filter, vi vay khi dua vao thesis nen chu thich ro day la ket qua raw voi prompt ngan, khong phai loi ghep anh.

### Hinh 3.2. No-bbox so voi bbox

**Ten file hien co:** `fig_3_2_no_bbox_vs_bbox_layout_comparison.png`

Hinh nay hien la hinh chinh trong thesis, khong con la optional. No giai thich vi sao pipeline compound cuoi cung uu tien prompt no-bbox va de target image day cho model cach can giua dong. Nguon anh lien quan:

`experiments/results/coverage_v10_eval/soup567_compound6x10_seed7000/`

`experiments/results/coverage_v10_eval/soup567_compound6x10_bbox_rows_seed7000/`

`experiments/results/coverage_v10_eval/soup567_compound6x10_bbox_yx_wide_probe_seed7000/`

### Hinh 3.3. Font reference so voi anh model sinh

**Ten file hien co:** `fig_3_3_font_reference_vs_model_generation.jpg`

Hinh nay cung la hinh chinh trong thesis. No cho thay font-rendered target du gan voi layout model sinh de dung lam du lieu giam sat, dong thoi van giu dung chu va dau tieng Viet. Nguon chinh:

`experiments/results/coverage_v10_eval/soup567_compound6x10_fontref_no_bbox_modelmatch_wide_seed7000/contact_sheet_compare.jpg`

### Hinh 3.5. Cai thien sinh tung tu

**Ten file hien co:** `fig_3_5_single_word_before_after_examples.png`

Hinh nay chung minh giai doan single-word da cai thien truoc khi chuyen sang compound training. Cac tu dai dien nen tiep tuc la `Bậc`, `Cữu`, `Chưởng`, `Gẫy`, `Ghế`, `Huyên`, vi chung bao phu nhieu kieu loi: mat dau thanh, doi dau, nham nguyen am co dau va nham cum nguyen am.

### Hinh 3.6. Compound Eval28 truoc/sau

**Ten file hien co:** `fig_3_6_compound_eval28_before_after.png`

Day la hinh ket qua quan trong nhat cua Chuong 3. No can bam sat ket qua: baseline `soup567` co 56 loi tren 168 tu, checkpoint cuoi `soup_lr3e5_gold4_9to1` con 4 loi tren 168 tu. Nguon anh:

`experiments/results/coverage_v10_eval/compound_eval28_soup567_baseline/`

`experiments/results/coverage_v10_eval/compound_eval28_soup_lr3e5_gold4_9to1/`

### Hinh 3.7. Loi con lai sau checkpoint tot nhat

**Ten file hien co:** `fig_3_7_remaining_error_cases.png`

Hinh nay giup thesis trung thuc hon: checkpoint da cai thien manh nhung chua hoan hao. Nen giu bon loi cuoi:

`Hấn -> Hẩn`, `Chịt -> Chút`, `Huyên -> Huyện`, `Dôi -> Dồi`.

### Hinh 3.8. Chu thu phap trong boi canh sinh anh nen

**Ten file hien co:** `fig_3_8_calligraphy_with_base_model_capability.png`

Hinh nay kiem tra yeu cau rat quan trong: fine-tuning khong chi lam chu dung hon tren nen trang, ma con phai giu duoc kha nang dung bo cuc, nen, anh sang, vat the va khong khi thiet ke cua Ideogram4. Ban hien tai dung hai prompt seed 7002:

- `Phúc Thọ / An Khang`
- `Tâm Đức Trí Nhân / Phúc Thọ Khang Vượng`

Bo cuc chinh nen giu 3 cot: Ideogram4 goc, `soup567`, checkpoint compound cuoi. Cum `Lộc` khong nen dung trong hinh promote vi de mat mu `ô`; `Vượng` cho ket qua chap nhan duoc hon.

## Thu Tu Uu Tien Thuc Te

Neu can nop nhanh, nen uu tien lam du cac hinh thesis dang tham chieu. Thu tu uu tien:

1. Cac hinh ket qua da co: 1.3, 3.2, 3.3, 3.5, 3.6, 3.7, 3.8.
2. Hinh 1.2 vi day la hinh so sanh doi thu/nen, giup bao ve ly do chon Ideogram4.
3. Cac diagram de ve nhanh: 1.1, 2.1, 2.2, 3.1.
4. Hinh 3.4 vi can tong hop so lieu thanh chart, nhung no rat huu ich de nguoi doc thay trajectory thay vi doc tung bang.

## Luu Y Khi Chon Anh Cuoi

Tat ca anh so sanh trong cung mot hinh nen dung cung prompt, cung seed va cung bo cuc neu co the. Khong nen lay anh dep nhat cua checkpoint nay so voi anh xau nhat cua checkpoint khac. Voi cac anh loi, nen khoanh hoac crop vung loi thay vi bat nguoi doc tu tim trong anh lon. Ten file cuoi nen giu dung ma hinh trong thesis de khi chuyen sang Word hoac LaTeX khong bi nham.
