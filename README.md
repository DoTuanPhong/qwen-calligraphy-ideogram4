# Fine-tuning Ideogram4 for Vietnamese Calligraphy Text Rendering

This repository contains the official restructured codebase and Master's thesis documentation for the **Vietnamese Calligraphy Generation** project. 

Our pipeline implements a surgical Parameter-Efficient Fine-Tuning (PEFT) strategy on top of **Ideogram4 (9.3B)** to resolve the Vietnamese diacritic plateau and enable high-fidelity text rendering of compound (multi-word) calligraphy layouts.

## Core Features & Contributions
- **Surgical Wide-Target LoRA:** Injects trainable BF16 LoRA adapters into 6 targeted DiT modules (`attention.qkv`, `attention.o`, `feed_forward.w1`, `feed_forward.w2`, `feed_forward.w3`, `adaln_modulation`) while freezing the base FP8 Ideogram4 weights.
- **Diacritic Masked Loss:** Implements a custom pixel/patch-masked MSE loss that boosts the gradient weight of fine-grained diacritical marks (e.g. `â`, `ă`, `ê`, `ô`, `ơ`, `ư` and tone markers) by a factor of `10.0`, preventing character mutations.
- **No-Bounding-Box Compound Layout:** Adapts training to centered two-line multi-word sentences directly, bypassing layout limitations of strict bounding boxes.
- **Checkpoint Averaging (Souping):** Averages late-stage compatible checkpoints to reduce parameter variance, achieving **97.6% rendering accuracy** (only 4 errors out of 168 words on the compound Eval28 panel).

## Visual Results & Comparisons

### 1. Baseline vs. Fine-Tuned (Eval28 Compound Panel)
Comparison between the base Ideogram4 model and our final compound souped checkpoint across complex multi-word Vietnamese calligraphy phrases:
![Before and After Comparison](docs/figures/fig_3_6_compound_eval28_before_after.png)

### 2. Integration with Complex Decorative Scenes
Testing calligraphy text rendering combined with full holiday still-life backgrounds (Tet holiday theme):
![Calligraphy in Tet holiday scenes](docs/figures/fig_3_8_calligraphy_with_base_model_capability.png)

## Thesis Documentation & Supplementary Reports

All supplementary documents, diagnostic reports, and error audits referenced in the main thesis are included under the `docs/` and `experiments/` directories:
- **Master's Thesis:** [English Version](docs/Final_Thesis_V10_EN.md) | [Vietnamese Version](docs/Final_Thesis_V10_VI.md)
- **Tokenizer Investigation Reports:** 
  - [ERNIE Image Tokenization Report](docs/ernie_image_tokenization_report.vi.md) (Mistral3 byte-level BPE tokenizer issues)
  - [Qwen3-VL Tokenizer Robustness Report](docs/v10_tokenizer_robustness_report.vi.md) (detailed analysis of Vietnamese diacritical characters)
- **Qwen3-VL Signal & Family Probes:**
  - [Signal Probe Report](docs/probe_reports/_QWEN3_VL_SIGNAL_PROBE_2026-06-23.md) (tap space diagnostic probe)
  - [Cuu Family Probe Report](docs/probe_reports/_QWEN3_VL_CUU_FAMILY_PROBE_2026-06-24.md) (tone/vowel binding diagnostic probe)
- **Evaluation Baseline Reports:**
  - [Baseline Results (4-Word)](docs/eval_baseline_results_4w.md) (details on the 16.27% CER baseline measurement using Vintern-3B-R-beta)
  - [Baseline Framework Specifications](docs/eval_baseline_framework_4w.md)
- **Manual Stress Probe Logs:** [manual_error_summary_20260701.json](experiments/results/coverage_v10_eval/scene_slant_probe_test28_seed7000/manual_error_summary_20260701.json) (manual word-level scoring for the Test28 stress probe)

---

## Directory Structure

The repository is organized as a clean, modular Python codebase alongside active research reports:

```text
├── README.md                 # Project introduction, setup, and usage guides
├── requirements.txt          # Python package dependencies
├── src/                      # Core codebase
│   ├── peft/                 
│   │   ├── hybrid_peft.py    # LoRA wrapper injection into DiT blocks
│   │   └── utils.py          # Central VAE mu-only encoding & input preparation helpers
│   ├── loss/                 
│   │   ├── diacritic_loss.py # Diacritic Masked MSE flow-matching loss
│   │   └── mask_builder.py   # Self-contained Unicode-gated character mask builder
│   ├── dataset/              
│   │   ├── loader.py         # Custom dataset loader for Ideogram4 training
│   │   └── builder.py        # Greedy set-cover compound dataset generator
│   └── pipeline/             
│   │   ├── train.py          # Cosine-annealed training loop for FP8 base + BF16 LoRA
│   │   └── infer.py          # Unified inference script for calligraphy text generation
├── scripts/                  
│   ├── soup_checkpoints.py   # Checkpoint averaging and rsLoRA conversion script
│   └── evaluate.py           # Batch evaluation runner on custom prompt panels
├── experiments/              # Experiment logs and scripts (from thesis)
│   ├── scripts/v10/          # Specific v10 data/training scripts
│   └── results/              # Target evaluation error logs (json)
├── assets/                   
│   ├── fonts/                # Target calligraphy font directory
│   ├── vietnamesesyllable_7184.txt # Vietnamese syllable lexicon list
│   └── elim_state_soup567.json    # Diagnostic evaluation error logs
└── docs/                     
    ├── Final_Thesis_V10_EN.md # Master Thesis (English Version)
    ├── Final_Thesis_V10_VI.md # Master Thesis (Vietnamese Version)
    ├── ernie_image_tokenization_report.vi.md # ERNIE tokenizer analysis
    ├── v10_tokenizer_robustness_report.vi.md # Qwen tokenizer analysis
    ├── eval_baseline_results_4w.md # Baseline diacritic-accuracy report
    ├── eval_baseline_framework_4w.md # Evaluation framework specs
    ├── probe_reports/        # Qwen3-VL tap & projection probe reports
    └── figures/              # Embedded research diagrams and result crops
```

---

## Installation & Setup

1. **Clone this repository & modelscope/DiffSynth-Studio submodule:**
   ```bash
   git clone --recursive <repo-url>
   cd qwen-calligraphy-ideogram4
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Base Model Weights:**
   Download the public weights of `ideogram-ai/ideogram-4-fp8` from Hugging Face into a local directory:
   ```bash
   huggingface-cli download ideogram-ai/ideogram-4-fp8 --local-dir models/ideogram-ai/ideogram-4-fp8
   ```

4. **Prepare Calligraphy Font:**
   Place the target TrueType font (e.g. `Thu_Phap_Thanh_Cong_Unicode.ttf`) in `assets/fonts/`.

---

## Code Walkthrough & Usage

### 1. Build Training Dataset & Masks
Generate compound phrases and corresponding character-level diacritic masks to prepare the training data:
```bash
# 1. Package compound metadata and horizontal font-target renders
python3 -m src.dataset.builder

# 2. Build Unicode-gated binary diacritic masks for the generated dataset
python3 -m src.loss.mask_builder --images data/coverage_v10/images_compound --out data/coverage_v10/masks_diacritic
```

### 2. Fine-tuning with Masked Loss
Run the main trainer using FP8 base weights, BF16 LoRA parameters, and the custom diacritic-masked loss:
```bash
python3 -m src.pipeline.train \
    --model_dir models/ideogram-ai/ideogram-4-fp8 \
    --metadata_path data/coverage_v10/metadata.jsonl \
    --image_base_dir data/coverage_v10 \
    --max_steps 12000 \
    --save_steps 1000 \
    --output_dir experiments/checkpoints/v10_compound_run \
    --use_8bit_adam
```

### 3. Checkpoint Souping & rsLoRA Conversion
Average late-stage checkpoints (e.g., Gold4 and a subsequent learning rate branch) and convert the standard training checkpoint scale (`alpha/rank`) to the optimized rsLoRA inference scale (`alpha/sqrt(rank)`):
```bash
python3 scripts/soup_checkpoints.py \
    --base experiments/checkpoints/gold4/step-soup.safetensors \
    --new experiments/checkpoints/lr3e5_branch/step-11820.safetensors \
    --out experiments/checkpoints/soup_9to1/step-soup.safetensors \
    --weight 9.0
```
This automatically produces `experiments/checkpoints/soup_9to1/step-soup_infer.safetensors` pre-scaled for inference.

### 4. Inference
Run single-word or multi-word calligraphy generation:
```bash
# General multi-word phrase with centered layout
python3 -m src.pipeline.infer \
    --model_dir models/ideogram-ai/ideogram-4-fp8 \
    --lora_ckpt experiments/checkpoints/soup_9to1/step-soup_infer.safetensors \
    --lora_targets attention.qkv attention.o feed_forward.w1 feed_forward.w2 feed_forward.w3 adaln_modulation \
    --hard_cases "Tâm Đức Trí Nhân" "Phúc Thọ Khang Vượng" \
    --output_dir experiments/results/generation \
    --seed 7000
```
*Note: Public pre-trained weights for our final model are available at Hugging Face: [phong09021998/vietnamese-calligraphy-ideogram4-lora](https://huggingface.co/phong09021998/vietnamese-calligraphy-ideogram4-lora).*

---

## Citation
If you use this codebase or checkpoint in your research, please cite our Master's thesis:
```bibtex
@mastersthesis{dopt2026vietnamesecalligraphy,
  author       = {Đỗ Tuấn Phong},
  title        = {Fine-tuning Ideogram4 for Vietnamese Calligraphy Text Rendering},
  school       = {FPT University},
  address      = {Hanoi, Vietnam},
  year         = {2026},
  type         = {{M.Sc.}},
  month        = jun,
  note         = {MSE-AI program}
}
```
