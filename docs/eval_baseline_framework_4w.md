# Qwen-Image-2512 Baseline Evaluation Framework
## "Vietnamese-Calligraphy-4W" — Pre-LoRA Component Audit

| Field | Value |
|-------|-------|
| **Document ID** | `EVAL-VNCAL-4W-v1.0` |
| **Date** | 2026-03-27 |
| **Scope** | Base Qwen-Image-2512 model, 1–4 Vietnamese words |
| **Target Font** | Thu Phap Thanh Cong Unicode |
| **Reference Dataset** | 8,028 synthetic calligraphy images (`dataset_8028_calligraphy_1font`) |
| **Objective** | Establish quantitative baseline before LoRA fine-tuning |

---

## Table of Contents

1. [Component-Level Audit Strategy](#1-component-level-audit-strategy)
2. [Vietnamese-Calligraphy-4W Evaluation Metrics](#2-vietnamese-calligraphy-4w-evaluation-metrics)
3. [Concrete Test Cases (Prompt Matrix)](#3-concrete-test-cases-prompt-matrix)
4. [Gap Analysis & Predicted Failure Modes](#4-gap-analysis--predicted-failure-modes)
5. [Appendices](#5-appendices)

---

## 1. Component-Level Audit Strategy

This section defines specific testing methodologies for each of Qwen-Image's three core components, calibrated to the constraints of the 1–4 word Vietnamese calligraphy task.

### 1.1 Qwen2.5-VL (MLLM / Text Encoder) — Diacritic Comprehension Audit

**Role in Pipeline:** Qwen2.5-VL (7B params: 32-layer ViT + 28-layer LLM) serves as the condition encoder. It processes the full `<|im_start|>` conversational prompt and produces the last-layer hidden state that conditions the MMDiT backbone. For calligraphy generation, the encoder must correctly tokenize, disambiguate, and represent every Vietnamese diacritical character in the prompt.

**Why This Is Critical:** Vietnamese uses 5 tone marks (sắc `´`, huyền `` ` ``, hỏi `?-hook`, ngã `~`, nặng `.`) applied over base vowels that may already carry circumflex or breve modifiers (e.g., `ă`, `â`, `ê`, `ô`, `ơ`, `ư`). This creates "stacked" diacritic combinations like `ậ` (â + nặng), `ễ` (ê + ngã), `ỗ` (ô + ngã). If the encoder conflates or drops any modifier, the downstream diffusion will render incorrect characters.

#### Test Methodology

| Test ID | Method | What We Measure | Pass Criteria |
|---------|--------|-----------------|---------------|
| **ENC-01** | **Token-level inspection** — Extract Qwen2.5-VL's tokenization of each test prompt. Verify that stacked-diacritic characters (e.g., `Nhẫn`, `ườ`, `ức`) are tokenized into correct, distinct token IDs vs. their unaccented counterparts. | Token ID uniqueness for diacritic variants | `ẫ` ≠ `a`, `ườ` ≠ `uo`, `ức` ≠ `uc` at token level |
| **ENC-02** | **Hidden-state cosine similarity** — Compute cosine similarity between the final hidden state of minimal-pair prompts: `"Nhẫn"` vs. `"Nhan"`, `"Phúc"` vs. `"Phuc"`. | Semantic separation of diacritic-bearing vs. bare forms | cos_sim < 0.90 (sufficient discrimination) |
| **ENC-03** | **Prompt-format fidelity** — Feed the exact `<\|im_start\|>system...assistant` template from `metadata.jsonl`. Confirm the encoder does not truncate or re-encode the special tokens, and that the assistant's Vietnamese content between the last `<\|im_start\|>assistant` and `<\|im_end\|>` is fully preserved in the hidden state. | Output latent dimension, no NaN/Inf gradients | Hidden state shape = expected; no anomalies |
| **ENC-04** | **NFC normalization check** — Confirm all test prompts are in Unicode NFC form and that the tokenizer handles both NFC and NFD inputs identically (Vietnamese text frequently appears in both forms). | Token output identity for NFC vs. NFD variants of the same string | Identical token IDs for NFC(`"Nhẫn"`) and NFD(`"Nhẫn"`) |

#### Procedure
```python
# ENC-02: Hidden-state cosine similarity
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
import torch.nn.functional as F

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B")

pairs = [("Nhẫn", "Nhan"), ("Phúc", "Phuc"), ("đức", "duc"), ("trường", "truong")]
for vn, bare in pairs:
    h_vn = model.encode(tokenizer(vn))   # pseudo-code: extract last hidden
    h_bare = model.encode(tokenizer(bare))
    sim = F.cosine_similarity(h_vn, h_bare, dim=-1).mean()
    assert sim < 0.90, f"Diacritic discrimination too low: {vn} vs {bare} = {sim:.4f}"
```

---

### 1.2 MMDiT Backbone & MSRoPE — Spatial Layout & Character Count Control

**Role in Pipeline:** The 20B-parameter MMDiT (60 layers, 24 heads, head_size=128) performs the core denoising diffusion conditioned on the Qwen2.5-VL hidden states. Multimodal Scalable RoPE (MSRoPE) handles positional encoding by conceptualizing text tokens along the diagonal of the image grid, maintaining 1D-RoPE equivalence for text while enabling 2D resolution scaling for images.

**Why This Is Critical for 1–4 Words:** The dataset imposes strict character-count boundaries. With only 1–4 words, the spatial "budget" is minimal. The model must:
- Place exactly N words (no hallucinated extra characters)
- Respect the `vertical` or `horizontal` layout directive
- Scale character sizes proportionally (1 word = larger, 4 words = smaller)
- Center content appropriately on a "clean background"

#### Test Methodology

| Test ID | Method | What We Measure | Pass Criteria |
|---------|--------|-----------------|---------------|
| **SPT-01** | **Character-count fidelity** — Generate images for prompts specifying 1, 2, 3, and 4 words. Apply Vintern-3B OCR to count recognized words. | OCR word count vs. prompt word count | Count_detected == Count_prompt for ≥ 80% of samples |
| **SPT-02** | **Hallucination detection** — Generate 10 images per word-count category. Flag any image where OCR detects MORE characters than specified in the prompt. | Hallucination rate (extra characters / strokes) | Hallucination rate ≤ 10% |
| **SPT-03** | **Vertical layout compliance** — For prompts specifying `vertical layout centered`, measure the bounding-box aspect ratio of detected text. Vertical layout → height >> width. | Aspect ratio of text bounding box | Height/Width ratio > 2.0 for vertical layout |
| **SPT-04** | **Centering accuracy** — Compute centroid of all detected text bounding boxes. Measure offset from image center. | Centroid deviation from image center (pixels) | Offset < 15% of image dimension |
| **SPT-05** | **Scale proportionality** — For 1-word vs. 4-word prompts at the same resolution, compare average character height. Single words should occupy more canvas. | Relative character height: `h(1_word) / h(4_word)` | Ratio > 1.5 |

#### Procedure
```python
# SPT-01 + SPT-02: Character count fidelity & hallucination detection
from experiments.scripts.eval_checkpoints import VLMOcrEngine

engine = VLMOcrEngine(model_name="vintern_3b")
test_prompts = {
    1: "Nhẫn",
    2: "An Nhiên",
    3: "Vạn Sự An",
    4: "Phúc đức trường tồn"
}

for expected_count, content in test_prompts.items():
    # Generate image with base model (pseudo-code)
    image = qwen_image_generate(prompt=format_prompt(content, layout="vertical"))
    detected_text = engine.predict(image)
    detected_words = detected_text.strip().split()
    
    count_match = len(detected_words) == expected_count
    hallucinated = len(detected_words) > expected_count
    
    print(f"[{expected_count}W] Expected: {expected_count}, Got: {len(detected_words)}, "
          f"Match: {count_match}, Hallucinated: {hallucinated}")
```

---

### 1.3 VAE Decoder — Calligraphic Stroke Fidelity

**Role in Pipeline:** The VAE decoder (15 layers, 73M params, 8×8 scale factor, 16 channels) reconstructs the generated image from the MMDiT's latent space. Qwen-Image's VAE was specifically fine-tuned on text-rich images to improve small-text reconstruction (PSNR 36.63 dB, SSIM 0.9839 on text benchmarks — see Technical Report Table 2).

**Why This Is Critical:** The "Thu Phap Thanh Cong Unicode" font has distinctive calligraphic characteristics:
- **Thin, variable-width strokes** — pressure-sensitive brush simulation
- **Ink bleeding / feathering** at stroke edges — natural ink-on-paper effect
- **Sharp stroke terminals** — distinct endings vs. gradual fade-outs
- **Connected vs. disconnected radicals** — some strokes merge naturally in calligraphy

The VAE must reconstruct these micro-features without introducing artifacts, blurring stroke edges, or merging adjacent strokes.

#### Test Methodology

| Test ID | Method | What We Measure | Pass Criteria |
|---------|--------|-----------------|---------------|
| **VAE-01** | **Encode-decode roundtrip on ground truth** — Take 20 reference images from `dataset_8028` (5 per word-count bucket), encode through the VAE encoder, decode, and measure reconstruction quality. | PSNR, SSIM between original and reconstructed | PSNR ≥ 35 dB, SSIM ≥ 0.97 |
| **VAE-02** | **Stroke-width preservation** — In reconstructed images, measure mean stroke width using skeletonization and distance-transform. Compare to ground truth. | Stroke width deviation (pixels) | Mean deviation ≤ 2px |
| **VAE-03** | **Ink bleeding fidelity** — Analyze the gradient profile at stroke edges in LAB color space. Compute the mean transition width (number of pixels from full ink to background). | Edge gradient profile width (px) | Within ±30% of ground truth profile |
| **VAE-04** | **Thin stroke survival** — Identify strokes thinner than 3px in ground truth. Measure what fraction survives encode-decode roundtrip (is still detectable as connected component after binarization). | Thin-stroke survival rate | ≥ 85% survival |
| **VAE-05** | **Diacritic mark preservation** — Crop the region containing tone marks from ground truth and reconstructed images. Measure SSIM on cropped diacritic regions only. | Diacritic-region SSIM | ≥ 0.92 |

#### Procedure
```python
# VAE-01: Encode-decode roundtrip
import torch
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

vae = load_qwen_image_vae()  # Load VAE encoder + decoder
sample_images = load_dataset_samples("dataset_8028_calligraphy_1font", n=20)

for img_path, img in sample_images:
    z = vae.encode(img)          # Latent
    img_recon = vae.decode(z)    # Reconstructed
    
    s = ssim(img, img_recon, channel_axis=-1)
    p = psnr(img, img_recon)
    
    print(f"{img_path}: PSNR={p:.2f} dB, SSIM={s:.4f}")
    assert p >= 35.0, f"PSNR below threshold: {p:.2f}"
    assert s >= 0.97, f"SSIM below threshold: {s:.4f}"
```

---

## 2. Vietnamese-Calligraphy-4W Evaluation Metrics

This section defines the composite scoring system for the `VNCAL-4W` benchmark. All metrics are evaluated via Vintern-3B (gold-standard VLM OCR) unless otherwise specified.

### 2.1 Metric Definitions

#### 2.1.1 Character Accuracy (CA)

**Definition:** Normalized edit-distance-based accuracy between OCR-detected text and ground-truth content.

```
CA = 1 - (Levenshtein(OCR_output, ground_truth) / max(len(OCR_output), len(ground_truth)))
```

| Rating | CA Score | Interpretation |
|--------|---------|----------------|
| **Excellent** | ≥ 0.95 | Near-perfect character rendering |
| **Good** | 0.85 – 0.94 | Minor substitutions (e.g., `ạ` → `a`) |
| **Acceptable** | 0.70 – 0.84 | Visible errors but text partially readable |
| **Fail** | < 0.70 | Majority of characters incorrect or missing |

> [!IMPORTANT]
> For the **base model** (pre-LoRA), we predict CA scores in the **Fail** range (< 0.70) for Vietnamese calligraphy. This establishes the baseline gap that LoRA training must close.

---

#### 2.1.2 Diacritic Integrity Score (DIS)

**Definition:** Measures how accurately tone marks and base-vowel modifiers are rendered, independent of the base character. Uses character-level decomposition.

```
DIS = (correctly_rendered_diacritic_chars / total_diacritic_chars_in_ground_truth)
```

**Failure taxonomy:**

| Failure Type | Example | Severity |
|--------------|---------|----------|
| **Dropped tone** | `Nhẫn` → `Nhân` (ngã mark lost) | Critical |
| **Wrong tone** | `Nhẫn` → `Nhận` (ngã → nặng) | Critical |
| **Fused diacritic** | `ễ` rendered as single blob (ê + ngã merge) | High |
| **Floating diacritic** | Tone mark detached from vowel body | Medium |
| **Misplaced diacritic** | Tone on wrong vowel in multi-vowel syllable | High |

| Rating | DIS Score | Interpretation |
|--------|----------|----------------|
| **Excellent** | ≥ 0.90 | Most diacritics correct |
| **Good** | 0.70 – 0.89 | Some diacritic errors but words recognizable |
| **Acceptable** | 0.50 – 0.69 | Frequent diacritic failures |
| **Fail** | < 0.50 | Systematic diacritic failure |

---

#### 2.1.3 Layout Adherence Score (LAS)

**Definition:** Composite score measuring compliance with the specified layout directive.

```
LAS = 0.4 × Orientation_Score + 0.3 × Centering_Score + 0.3 × Spacing_Score
```

| Sub-metric | Measurement | Scoring |
|------------|-------------|---------|
| **Orientation** | Text bounding-box aspect ratio vs. expected (vertical: H/W > 2, horizontal: W/H > 2) | Binary: 1.0 if correct, 0.0 if wrong |
| **Centering** | Centroid offset from image center / image dimension | `1 - (offset / max_offset)` clamped [0, 1] |
| **Spacing** | Coefficient of variation of inter-character gaps | `1 - min(CV, 1.0)` (lower CV = more uniform) |

| Rating | LAS Score | Interpretation |
|--------|----------|----------------|
| **Excellent** | ≥ 0.85 | Layout matches specification |
| **Good** | 0.70 – 0.84 | Mostly correct orientation/centering |
| **Acceptable** | 0.50 – 0.69 | Orientation correct but centering/spacing off |
| **Fail** | < 0.50 | Layout incorrect or chaotic |

---

#### 2.1.4 Stylistic Consistency Score (SCS)

**Definition:** Measures visual similarity to the target "Thu Phap Thanh Cong Unicode" font style.

```
SCS = 0.5 × CLIP_Similarity + 0.3 × Stroke_Profile_Match + 0.2 × Background_Cleanliness
```

| Sub-metric | Measurement | Scoring |
|------------|-------------|---------|
| **CLIP Similarity** | Cosine sim between generated image CLIP embedding and average embedding of 50 dataset reference images | Raw cosine similarity [0, 1] |
| **Stroke Profile** | Compare stroke width distribution (via skeletonization) to reference | `1 - KL_divergence` (clamped [0, 1]) |
| **Background** | Percentage of background pixels that are "clean" (within ±10 L* of mean background) | Direct percentage [0, 1] |

| Rating | SCS Score | Interpretation |
|--------|----------|----------------|
| **Excellent** | ≥ 0.80 | Visually matches target calligraphy font |
| **Good** | 0.65 – 0.79 | Calligraphic style present, some deviations |
| **Acceptable** | 0.50 – 0.64 | Generic brush script, not font-specific |
| **Fail** | < 0.50 | No calligraphic characteristics |

---

### 2.2 Composite Score

```
VNCAL-4W = 0.35 × CA + 0.25 × DIS + 0.20 × LAS + 0.20 × SCS
```

The weighting prioritizes textual accuracy (CA + DIS = 60%) over visual/layout qualities (LAS + SCS = 40%), as the primary objective is correct Vietnamese text rendering.

---

## 3. Concrete Test Cases (Prompt Matrix)

> [!NOTE]
> As of the latest update, the evaluation suite has been **comprehensively expanded** from the original 4 core test cases to a centralized repository of **25 diverse test cases** located in `experiments/scripts/eval_test_cases.py`. This central registry is used by all components (Encoder, MMDiT `spt_tests.py`, VAE, and End-to-End metrics) to ensure complete synchronization.

All test cases use the **exact `<|im_start|>` format** from the project's `metadata.jsonl`. Each prompt is ready to be fed directly to the base Qwen-Image-2512 model. Below are the 4 foundational test cases, followed by a summary of the expanded suite.

### Test Case 1: 1-Word — Complex Stacked Diacritics

**Target:** `Nhẫn` (Patience) — Contains `ẫ` = `â` (circumflex) + `~` (ngã tone)

```json
{
  "file_name": "test/1_word/test_baseline_001.jpg",
  "text": "<|im_start|>system\nGenerate a high-quality image based on the given text description. Pay close attention to all the details mentioned, including objects, scenes, actions, colors, styles, and any specific requirements. Ensure the generated image accurately reflects the provided description with precise rendering of any text content.<|im_end|>\n<|im_start|>user\n<|user_text|>Vietnamese calligraphy artwork of the text 'Nhẫn'. It features a vertical layout centered on a clean background.<|im_end|>\n<|im_start|>assistant",
  "metadata": {
    "font": "Thu Phap Thanh Cong Unicode",
    "layout": "vertical",
    "content": "Nhẫn",
    "category": "short",
    "word_count": 1
  }
}
```

| Audit Focus | What to Check |
|-------------|---------------|
| **ENC** | Tokenizer separates `ẫ` from `â` and `a`; hidden state encodes ngã tone |
| **MMDiT/MSRoPE** | Single large character centered; no extra glyphs hallucinated |
| **VAE** | Circumflex and ngã mark both sharp; no fusion into single mark |

**Expected Base-Model Failures:** The stacked `ẫ` will likely be rendered as `â` (dropped ngã) or as a smudged blob. The model has no font-specific prior for Vietnamese calligraphic diacritics.

---

### Test Case 2: 2-Words — Mixed Tone Marks

**Target:** `An Nhiên` (Serenity) — Contains `ê` (circumflex on e) and no tone mark on `An`, demonstrating the model must render BOTH a bare vowel and an accented vowel correctly.

```json
{
  "file_name": "test/2_words/test_baseline_002.jpg",
  "text": "<|im_start|>system\nGenerate a high-quality image based on the given text description. Pay close attention to all the details mentioned, including objects, scenes, actions, colors, styles, and any specific requirements. Ensure the generated image accurately reflects the provided description with precise rendering of any text content.<|im_end|>\n<|im_start|>user\n<|user_text|>Vietnamese calligraphy artwork of the text 'An Nhiên'. It features a vertical layout centered on a clean background.<|im_end|>\n<|im_start|>assistant",
  "metadata": {
    "font": "Thu Phap Thanh Cong Unicode",
    "layout": "vertical",
    "content": "An Nhiên",
    "category": "short",
    "word_count": 2
  }
}
```

| Audit Focus | What to Check |
|-------------|---------------|
| **ENC** | Correct distinction between `ê` and `e`; proper digraph `Nh` handling |
| **MMDiT/MSRoPE** | Exactly 2 words separated with correct vertical spacing; proper proportional sizing |
| **VAE** | Clean separation between `An` and `Nhiên`; circumflex on `ê` is distinct and sharp |

**Expected Base-Model Failures:** `Nhiên` may render as `Nhien` (dropped circumflex), or inter-word spacing will be wrong for vertical calligraphy layout.

---

### Test Case 3: 3-Words — Vertical Layout Control

**Target:** `Vạn Sự An` (Ten-Thousand Things Peaceful) — Contains `ạ` (nặng dot below) and `ự` (ư + nặng). This tests vertical layout with 3 semantically independent words.

```json
{
  "file_name": "test/3_words/test_baseline_003.jpg",
  "text": "<|im_start|>system\nGenerate a high-quality image based on the given text description. Pay close attention to all the details mentioned, including objects, scenes, actions, colors, styles, and any specific requirements. Ensure the generated image accurately reflects the provided description with precise rendering of any text content.<|im_end|>\n<|im_start|>user\n<|user_text|>Vietnamese calligraphy artwork of the text 'Vạn Sự An'. It features a vertical layout centered on a clean background.<|im_end|>\n<|im_start|>assistant",
  "metadata": {
    "font": "Thu Phap Thanh Cong Unicode",
    "layout": "vertical",
    "content": "Vạn Sự An",
    "category": "medium",
    "word_count": 3
  }
}
```

| Audit Focus | What to Check |
|-------------|---------------|
| **ENC** | Dot-below diacritic (`ạ`, `ự`) correctly encoded; ư-horn correctly tokenized |
| **MMDiT/MSRoPE** | 3 words stacked vertically with even spacing; H/W aspect ratio > 2.0; centered on canvas |
| **VAE** | Nặng dots are visible below the baseline; ư-hook is not merged with the letter body |

**Expected Base-Model Failures:** The vertical 3-word layout is the highest-risk scenario. The base model will likely default to horizontal layout (its training bias for Latin scripts). `ự` (ư + nặng) is a triple-stacked form that will almost certainly be corrupted.

---

### Test Case 4: 4-Words — Maximum-Length Stress Test

**Target:** `Phúc đức trường tồn` (Blessings and Virtue Endure Forever) — Contains 4 diacritic-heavy words with `ú` (sắc), `ức` (sắc), `ườ` (ơ-hook + huyền), `ồ` (ô + huyền). This is the maximum complexity allowed within the 4-word limit.

```json
{
  "file_name": "test/4_words/test_baseline_004.jpg",
  "text": "<|im_start|>system\nGenerate a high-quality image based on the given text description. Pay close attention to all the details mentioned, including objects, scenes, actions, colors, styles, and any specific requirements. Ensure the generated image accurately reflects the provided description with precise rendering of any text content.<|im_end|>\n<|im_start|>user\n<|user_text|>Vietnamese calligraphy artwork of the text 'Phúc đức trường tồn'. It features a vertical layout centered on a clean background.<|im_end|>\n<|im_start|>assistant",
  "metadata": {
    "font": "Thu Phap Thanh Cong Unicode",
    "layout": "vertical",
    "content": "Phúc đức trường tồn",
    "category": "medium",
    "word_count": 4
  }
}
```

| Audit Focus | What to Check |
|-------------|---------------|
| **ENC** | All 4 distinct tone types correctly tokenized; `đ` (d-bar) not confused with `d` |
| **MMDiT/MSRoPE** | 4 words vertically stacked; no words truncated or merged; all text fits within canvas without overflow |
| **VAE** | Smaller character sizes (proportional to 4-word layout) still preserve all diacritics; stroke details not lost to downscaling |

**Expected Base-Model Failures:** With 4 words, the model will likely exhibit at least one of: (a) word truncation (only 3 words rendered), (b) character hallucination (extra strokes), (c) complete layout breakdown (text wraps or overlaps), (d) widespread diacritic loss at reduced character size.

---

### 3.1 Test Matrix Summary (Expanded 25-Case Suite)

The full suite of 25 test cases (TC1 to TC25) is categorized into Core Structural tests and Edge-Case Stress tests:

#### Core Diacritic & Length Baselines (TC1 - TC17)
Tests absolute fundamental capability to render Vietnamese characters across different lengths and combinations.
| TC Range | Focus Area | Example Words | Primary Risk |
|----|-------|---------|-------------|
| **TC1 - TC8** | 1-Word (Micro-diacritics) | *Nhẫn, Tâm, Phúc, Lộc, Trí* | Stacked diacritic fusion (ẫ), subtle accents (í) |
| **TC9 - TC12** | 2-Words (Tone Combos) | *An Nhiên, Hạnh Phúc* | Mixed accented/bare vowel (ê/e) |
| **TC13 - TC15** | 3-Words (Vertical Layout) | *Vạn Sự An, Thuận Tự Nhiên* | Vertical layout compliance, dot below (ạ, ự) |
| **TC16 - TC17** | 4-Words (Max Length) | *Phúc đức trường tồn* | Max-length capacity + proportional scaling |

#### Edge Cases & Stress Tests (TC18 - TC25)
Tests prompt adherence, negative constraints, and out-of-domain resistance.
| TC Range | Focus Area | Example Constraint | Primary Risk |
|----|-------|---------|-------------|
| **TC18 - TC19** | Layout Overrides | `layout="horizontal"` | MMDiT failing to break vertical prior |
| **TC20 - TC21** | Style/Color Overrides | `red ink on gold paper` | Color bleed, structural decay due to texture |
| **TC22** | Negative Prompts | `no red stamps or seals` | Hallucinating objects against constraint |
| **TC23 - TC25** | Out-of-Domain / Length | `8028`, ultra-long prompts | Hallucination, ignoring the core text |

---

## 4. Gap Analysis & Predicted Failure Modes

### 4.1 Capability Gap Summary

| Capability | Base Qwen-Image-2512 | Dataset Requirement | Gap Severity |
|------------|----------------------|---------------------|-------------|
| **Vietnamese diacritics** | Trained on English + Chinese text; Vietnamese is out-of-distribution for the text renderer | Perfect NFC Vietnamese with all 5 tones + 9 base modifiers | 🔴 Critical |
| **Calligraphy font style** | No exposure to "Thu Phap Thanh Cong Unicode"; trained on diverse fonts/styles | Single-font consistent style across 8,028 images | 🔴 Critical |
| **Vertical CJK-style layout for Latin-script text** | MSRoPE trained on vertical Chinese/Japanese text but NOT vertical Vietnamese (Latin-based script) | Strict vertical layout for romanized Vietnamese words | 🟠 High |
| **1–4 word brevity** | Optimized for complex, multi-element prompts (slides, couplets, paragraphs) | Minimal-content images with 1–4 words on clean background | 🟡 Medium |

---

### 4.2 Top 3 Predicted Failure Modes

#### Failure Mode 1: Systematic Diacritic Corruption (Predicted Severity: CRITICAL)

**Prediction:** The base model will fail to correctly render Vietnamese tone marks in ≥ 70% of test cases.

**Root Cause Analysis:**
- Qwen-Image was trained on English and Chinese text rendering (Technical Report §3.4). Vietnamese is a Latin-script language with diacritical complexity that rivals Chinese radical complexity but uses entirely different visual patterns.
- The VAE decoder was fine-tuned on "text-rich images" (§2.3), but these are predominantly English documents and Chinese characters. Vietnamese diacritics occupy a unique visual niche: small marks above/below Latin letters at specific positions.
- The MMDiT has never seen the mapping: `"text 'Nhẫn'"` prompt → calligraphic rendering of `ẫ` with correct circumflex + ngã placement. Without this association in training data, it will either:
  - Drop the diacritic entirely (render `Nhan` or `Nhân`)
  - Render a generic "accent-like" mark that doesn't match any Vietnamese tone
  - Fuse stacked diacritics into a single visual artifact

**How LoRA Fixes This:**
The 8,028-image dataset provides ~2,000+ unique diacritic-bearing characters across all 5 tones × 12 base vowel forms. LoRA fine-tuning of the MMDiT's cross-attention layers (the layers that attend to the Qwen2.5-VL condition signal) will create new associations between Vietnamese text tokens and their corresponding calligraphic visual patterns. The targeted weight updates (`rank=32–64`) are sufficient to encode this mapping without catastrophically forgetting the model's existing English/Chinese capabilities.

---

#### Failure Mode 2: Layout Default to Horizontal (Predicted Severity: HIGH)

**Prediction:** The base model will render text horizontally in ≥ 60% of cases despite the prompt explicitly requesting `"vertical layout centered"`.

**Root Cause Analysis:**
- The model's vertical text rendering capability (Technical Report §5, Figure 20) was trained on Chinese characters, which are inherently square and naturally stack vertically.
- Vietnamese words are rendered in Latin script, which Qwen-Image has only seen in horizontal configurations. The instruction "vertical layout" for Latin-script text is a novel, out-of-distribution request.
- MSRoPE encodes 2D spatial positions, but during training it was only exposed to vertical positioning for Chinese logographic tokens, not Latin-script tokens. The positional encoding has no prior for vertically stacking multi-letter Latin words.
- Even if the model attempts vertical layout, word spacing will be calibrated for Chinese characters (roughly square) rather than Vietnamese words (variable width, much wider than tall).

**How LoRA Fixes This:**
The 8,028 images in the dataset are labeled with `"layout": "vertical"` metadata and the prompts consistently specify vertical arrangement. LoRA training will:
1. Recalibrate the MMDiT's spatial attention to associate `"vertical layout"` + Latin text with top-to-bottom word arrangement
2. Learn the appropriate inter-word spacing for Vietnamese calligraphic words in vertical configurations
3. Adjust MSRoPE's effective behavior (via attention weight modification) to correctly map Latin-script words into vertical positions

---

#### Failure Mode 3: Style Hallucination / Font Mismatch (Predicted Severity: HIGH)

**Prediction:** The base model will generate text in a generic brush-style or a Chinese calligraphy style, NOT in the specific "Thu Phap Thanh Cong Unicode" style.

**Root Cause Analysis:**
- The term "calligraphy" in the prompt will activate the model's Chinese calligraphy prior (strongly represented in training data — see Technical Report Figure 2, bottom-right). This will produce:
  - Ink wash effects characteristic of Chinese brush calligraphy
  - Stroke dynamics mimicking Chinese cursive/semi-cursive scripts (草书/行书)
  - Possibly Chinese-style seal stamps or decorative elements
- "Thu Phap Thanh Cong Unicode" is a specific Vietnamese calligraphic font with distinctive characteristics:
  - Thinner, more uniform stroke widths than Chinese calligraphy
  - Less dramatic thick-thin variation
  - Subtle ink bleeding (not dramatic wash effects)
  - Clean, minimal backgrounds (not textured paper/scroll)
- The model has zero examples of this specific font in its training data.

**How LoRA Fixes This:**
The 8,028 images are ALL rendered in the same "Thu Phap Thanh Cong Unicode" font. LoRA fine-tuning will:
1. Overwrite the generic "calligraphy" prior with font-specific stroke patterns, width distributions, and ink bleeding profiles
2. Establish the mapping: "Vietnamese calligraphy artwork" → the specific visual characteristics of this font
3. Lock in the clean background aesthetic (the dataset uses `"clean background"` consistently, not textured paper)
4. The single-font constraint is actually an advantage for LoRA — it reduces the style space to a single point, making convergence faster and more reliable

---

### 4.3 Predicted Baseline Scores (Pre-LoRA)

| Metric | Test Case 1 (1W) | Test Case 2 (2W) | Test Case 3 (3W) | Test Case 4 (4W) | Average |
|--------|-------------------|-------------------|-------------------|-------------------|---------|
| **CA** (Character Accuracy) | 0.40 | 0.35 | 0.25 | 0.20 | **0.30** |
| **DIS** (Diacritic Integrity) | 0.20 | 0.25 | 0.15 | 0.10 | **0.18** |
| **LAS** (Layout Adherence) | 0.60 | 0.45 | 0.30 | 0.25 | **0.40** |
| **SCS** (Stylistic Consistency) | 0.35 | 0.30 | 0.30 | 0.25 | **0.30** |
| **VNCAL-4W Composite** | 0.39 | 0.34 | 0.25 | 0.20 | **0.30** |

> [!CAUTION]
> These are **predicted** scores based on architectural analysis. Actual baseline scores will be measured when running the 4 test cases against the base model. The predicted composite of **0.30** represents a model that cannot reliably produce Vietnamese calligraphy — confirming the need for LoRA fine-tuning.

### 4.4 Post-LoRA Target Scores

| Metric | Target (Post-LoRA 8K steps) | Justification |
|--------|----------------------------|---------------|
| **CA** | ≥ 0.85 | 8,028 training images provide dense character coverage |
| **DIS** | ≥ 0.80 | Dataset includes all 5 tones across 1–4 word configurations |
| **LAS** | ≥ 0.85 | Consistent vertical layout labels throughout dataset |
| **SCS** | ≥ 0.90 | Single-font dataset provides extremely strong style prior |
| **VNCAL-4W** | ≥ 0.85 | Overall target reflecting production-ready quality |

---

## 5. Appendices

### Appendix A: Evaluation Toolchain

| Tool | Purpose | Reference |
|------|---------|-----------|
| **Vintern-3B-R-beta** | Gold-standard VLM OCR for Vietnamese calligraphy | `get_ocr_engine("vintern_3b")` |
| **Qwen2.5-VL-7B** | Secondary OCR engine for cross-validation | `get_ocr_engine("qwen")` |
| **Gemini 3 Flash** | Cross-validation oracle (most accurate, rate limited) | `get_ocr_engine("gemini")` |
| **PaddleOCR** | Fast sanity check, character detection baseline | `get_ocr_engine("paddle")` |
| **GOT-OCR 2.0** | End-to-end VLM perspective | `get_ocr_engine("got")` |
| **PARSeq** | Latin scene text verification | `get_ocr_engine("parseq")` |
| **CLIP (ViT-L/14)** | Stylistic consistency scoring | `openai/clip-vit-large-patch14` |
| **scikit-image** | SSIM, PSNR for VAE roundtrip tests | `skimage.metrics` |
| **rapidfuzz** | CA and DIS character/word distance computation (faster than python-Levenshtein, SIMD-optimized) | `pip install rapidfuzz` |
| **scipy** | Distance transform for stroke-width analysis | `scipy.ndimage.distance_transform_edt` |

> See `eval_baseline_results_4w.md` Section 12 for the full expanded OCR engine survey (8 new engines evaluated 2026-03-28).

---

### Appendix D: Implementation Scripts

Scripts implementing this evaluation framework are located in `experiments/scripts/`:

#### D.1 Script Overview

| Script | Purpose | CLI Entry Point |
|--------|---------|-----------------|
| `eval_baseline_4w.py` | Main orchestration — generates images, runs OCR, computes all metrics, SPT tests | `python eval_baseline_4w.py --model_root ... --dataset_root ...` |
| `metrics_4w.py` | VNCAL-4W metric library (CA, DIS, LAS, SCS, composite) | `python metrics_4w.py` (self-test) |
| `enc_tests.py` | Encoder component tests ENC-01 → ENC-04 | `python enc_tests.py --tokenizer_path ...` |
| `vae_tests.py` | VAE decoder component tests VAE-01 → VAE-05 | `python vae_tests.py --model_root ... --dataset_root ...` |
| `ocr_engines.py` | Unified OCR engine collection (14 backends) with factory function | `python ocr_engines.py` (self-test) |
| `test_new_engines.py` | Integration test for new OCR engines | `python test_new_engines.py [engine_names...]` |

Shell launcher: `experiments/configs/eval_baseline_4w.sh`

#### D.2 Quick Start

```bash
# Verify metrics module (no GPU required)
python experiments/scripts/metrics_4w.py

# Quick baseline — generate 4 test images + OCR + VNCAL-4W metrics
bash experiments/configs/eval_baseline_4w.sh --quick

# Full evaluation including ENC and VAE component tests
bash experiments/configs/eval_baseline_4w.sh --full

# Encoder tests only (tokenizer-level, no GPU required for ENC-01/04)
bash experiments/configs/eval_baseline_4w.sh --enc-only

# VAE tests only (requires GPU + dataset)
bash experiments/configs/eval_baseline_4w.sh --vae-only
```

Environment variables for the shell launcher:

```bash
MODEL_ROOT=/path/to/Qwen-Image-2512-Local
DATASET_ROOT=/path/to/dataset_8028_calligraphy_1font
OUTPUT_DIR=./experiments/results/baseline_4w
OCR_ENGINE=vintern_3b
DEVICE=cuda:0
VRAM_LIMIT=75.0
```

#### D.3 Output Files

After a full run, `OUTPUT_DIR` contains:

| File | Contents |
|------|----------|
| `baseline_4w_results.json` | Full structured results — all metrics, SPT tests, ENC/VAE results |
| `baseline_4w_summary.csv` | Per-test-case metric scores (CA, DIS, LAS, SCS, composite) |
| `TC1_Nhẫn.jpg` | Generated image — TC1 (1 word) |
| `TC2_An_Nhiên.jpg` | Generated image — TC2 (2 words) |
| `TC3_Vạn_Sự_An.jpg` | Generated image — TC3 (3 words) |
| `TC4_Phúc_đức_trường_tồn.jpg` | Generated image — TC4 (4 words) |
| `enc_tests_results.json` | ENC-01→04 results (if `--run_enc`) |
| `vae_tests_results.json` | VAE-01→05 results (if `--run_vae`) |

#### D.4 Metric Module API

```python
from metrics_4w import (
    compute_character_accuracy,      # CA
    compute_diacritic_integrity,     # DIS — returns dict with failures list
    compute_layout_adherence,        # LAS — needs bounding boxes + image_size
    compute_stylistic_consistency,   # SCS — needs reference images for CLIP
    compute_vncal4w_composite,       # 0.35×CA + 0.25×DIS + 0.20×LAS + 0.20×SCS
    full_report,                     # Combined report dict
)

# Example
ca = compute_character_accuracy(ocr_text="Nhan", ground_truth="Nhẫn")
dis = compute_diacritic_integrity(ocr_text="Nhan", ground_truth="Nhẫn")
# dis["score"], dis["failures"], dis["total_diacritic_chars"]

composite = compute_vncal4w_composite(ca=0.40, dis=0.20, las=0.60, scs=0.35)
```

#### D.5 Dependencies

```
# Core
torch>=2.0
Pillow
rapidfuzz
numpy

# Spatial / VAE tests
scikit-image
scipy

# Style scoring
transformers  # for CLIP (openai/clip-vit-large-patch14)

# OCR (gold standard)
# Vintern-3B-R-beta — loaded via ocr_engines.py
```

### Appendix B: Unicode Reference — Vietnamese Diacritics Used in Test Cases

| Character | Unicode | Decomposition | Test Case |
|-----------|---------|---------------|-----------|
| ẫ | U+1EAB | â (U+00E2) + combining tilde (U+0303) | TC1 |
| ê | U+00EA | e (U+0065) + combining circumflex (U+0302) | TC2 |
| ạ | U+1EA1 | a (U+0061) + combining dot below (U+0323) | TC3 |
| ự | U+1EF1 | ư (U+01B0) + combining dot below (U+0323) | TC3 |
| ú | U+00FA | u (U+0075) + combining acute (U+0301) | TC4 |
| ức | — | ư (U+01B0) + combining acute (U+0301) + c | TC4 |
| ườ | — | ư (U+01B0) + combining grave (U+0300) + ơ (U+01A1) | TC4 |
| ồ | U+1ED3 | ô (U+00F4) + combining grave (U+0300) | TC4 |
| đ | U+0111 | Latin small letter d with stroke | TC4 |

### Appendix C: Dataset Distribution (1–4 Words Subset)

Based on the `dataset_8028_calligraphy_1font` structure:

| Word Count | Directory | Estimated Sample Count | Proportion |
|------------|-----------|----------------------|------------|
| 1 | `short/1_word/` | ~1,500 | ~19% |
| 2 | `short/2_words/` | ~2,000 | ~25% |
| 3 | `medium/3_words/` | ~1,800 | ~22% |
| 4 | `medium/4_words/` | ~1,200 | ~15% |
| 5–8 | `medium/5–8_words/` | ~1,528 | ~19% |
| **Total** | | **8,028** | **100%** |

> [!NOTE]
> The 1–4 word subset (~6,500 images, ~81% of the dataset) provides strong coverage for this evaluation scope. The LoRA will be primarily optimized for this range.

---

*Document prepared for: qwen_calligraphy_lora project — LoRA V4 pre-training baseline assessment*
*Reference: [Qwen-Image Technical Report](file:///workspace/qwen_calligraphy_lora/docs/Qwen-Image_Technical_Report.md) | [Dataset Guide](file:///workspace/qwen_calligraphy_lora/docs/guide_dataset_structure.md) | [Evaluation Metrics](file:///workspace/qwen_calligraphy_lora/docs/guide_evaluation_metrics.md)*
