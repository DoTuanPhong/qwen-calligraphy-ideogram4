# VNCAL-4W Baseline Evaluation Results
## Qwen-Image-2512 — Pre-LoRA Baseline Assessment

| Field | Value |
|-------|-------|
| **Document ID** | `EVAL-RESULTS-VNCAL-4W-v1.0` |
| **Date** | 2026-03-30 |
| **Reference** | `EVAL-VNCAL-4W-v1.0` (eval_baseline_framework_4w.md) |
| **Model** | Qwen-Image-2512 (base, no LoRA) |
| **OCR Engine** | Vintern-3B-R-beta (gold standard) |
| **Test Suite** | 25 test cases (TC1–TC25) from `eval_test_cases.py` |
| **Hardware** | NVIDIA A100 80GB PCIe |
| **Output** | `experiments/results/baseline_4w/` |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [VNCAL-4W Composite Results](#2-vncal-4w-composite-results)
3. [Per-Test-Case Detailed Results](#3-per-test-case-detailed-results)
4. [Metric Breakdown Analysis](#4-metric-breakdown-analysis)
5. [SPT Spatial Tests](#5-spt-spatial-tests)
6. [Encoder Tests (ENC-01 to ENC-04)](#6-encoder-tests-enc-01-to-enc-04)
7. [VAE Tests (VAE-01 to VAE-05)](#7-vae-tests-vae-01-to-vae-05)
8. [Failure Mode Analysis](#8-failure-mode-analysis)
9. [Predicted vs Actual Comparison](#9-predicted-vs-actual-comparison)
10. [Conclusions & LoRA Training Implications](#10-conclusions--lora-training-implications)
11. [OCR Engine Accuracy Assessment](#11-ocr-engine-accuracy-assessment)
12. [Expanded OCR Engine Survey](#12-expanded-ocr-engine-survey)
13. [Comprehensive OCR Accuracy Evaluation](#13-comprehensive-ocr-accuracy-evaluation)
    - 13.1 [Motivation](#131-motivation)
    - 13.2 [New Engines Added](#132-new-engines-added)
    - 13.3 [Evaluation Methodology](#133-evaluation-methodology)
    - 13.4 [Phase 1 — Smoke Test Results](#134-phase-1--smoke-test-results)
    - 13.5 [Phase 2 — Generated Images (TC1–TC25)](#135-phase-2--generated-images-tc1tc25-human-verified-gt)
    - 13.6 [Phase 3 — Dataset Samples](#136-phase-3--dataset-samples-metadata-gt)
    - 13.7 [Vintern-3B Comprehensive Results](#137-vintern-3b-comprehensive-results-added-2026-03-28)
    - 13.8 [Chandra OCR Results](#138-chandra-ocr-results-added-2026-03-28)
    - 13.9 [Cross-Engine Phase 2 Comparison Matrix](#139-cross-engine-phase-2-comparison-matrix-human-verified-gt)
    - 13.9.1 [Gemini 3 Flash Preview — by prompt](#1391-gemini-3-flash-preview--ocr-output-by-prompt-phase-2-human-verified-gt)
    - 13.9.2 [Gemini 2.5 Flash — by prompt](#1392-gemini-25-flash--ocr-output-by-prompt-phase-2-human-verified-gt)
    - 13.9.3 [Gemini 2.5 Flash Lite — by prompt](#1393-gemini-25-flash-lite--ocr-output-by-prompt-phase-2-human-verified-gt)
    - 13.10 [Comprehensive Engine Ranking](#1310-comprehensive-engine-ranking-updated)
    - 13.10.1 [Gemini — metrics by model × prompt](#13101-gemini--full-metrics-by-model-and-prompt-phase-2--dataset)
    - 13.11 [Key Findings](#1311-key-findings-updated)
    - 13.12 [Cross-Validation Strategy](#1312-updated-multi-engine-cross-validation-strategy)
    - 13.13 [Installation Requirements](#1313-installation-requirements)
    - 13.14 [Scripts & Reproducibility](#1314-scripts--reproducibility)
14. [Prompt Comparison Study](#14-prompt-comparison-study-default-vs-strict-anti-autocorrect-2026-03-28)
    - 14.1 [Motivation](#141-motivation)
    - 14.2 [Engines Tested](#142-engines-tested)
    - 14.3 [Results — Phase 2 (Generated Images)](#143-results--phase-2-generated-images-tc1tc25)
    - 14.4 [Results — Phase 3 (Dataset Samples)](#144-results--phase-3-dataset-samples)
    - 14.5 [Analysis](#145-analysis)
    - 14.6 [Conclusion](#146-conclusion)
    - 14.7 [Expanded Prompt Comparison — Gemini 3.1 Flash Lite](#147-expanded-prompt-comparison-for-gemini-31-flash-lite-2026-03-30)
    - 14.8 [Five-Prompt OCR Sweep — Gemini 3 Preview, 2.5 Flash, 2.5 Flash Lite](#148-five-prompt-ocr-sweep--gemini-3-preview-25-flash-25-flash-lite-2026-03-30)
15. [Corrected VNCAL-4W Scores (Human-Verified Ground Truth)](#15-corrected-vncal-4w-scores-human-verified-ground-truth)
    - 15.1 [Motivation](#151-motivation)
    - 15.2 [Methodology](#152-methodology)
    - 15.3 [Corrected Overall Scores](#153-corrected-overall-scores)
    - 15.4 [Per-TC Corrected Results](#154-per-tc-corrected-results)
    - 15.5 [Most Affected Test Cases](#155-most-affected-test-cases)
    - 15.6 [Corrected Averages by Word Count](#156-corrected-averages-by-word-count)
    - 15.7 [Key Findings](#157-key-findings)

---

## 1. Executive Summary

| Metric | Score | Rating | Target (Post-LoRA) |
|--------|-------|--------|---------------------|
| **CA** (Character Accuracy) | **0.7535** | Acceptable | >= 0.85 |
| **DIS** (Diacritic Integrity) | **0.4640** | Fail | >= 0.80 |
| **LAS** (Layout Adherence) | **0.4822** | Fail | >= 0.85 |
| **SCS** (Stylistic Consistency) | **0.7381** | Good | >= 0.90 |
| **VNCAL-4W Composite** | **0.6238** | Acceptable | >= 0.85 |

**Overall Verdict:** The base Qwen-Image-2512 achieves an **Acceptable** baseline (0.6238), significantly better than the predicted Fail (0.30). However, two critical metrics — **Diacritic Integrity (0.46)** and **Layout Adherence (0.48)** — remain firmly in Fail territory, confirming that **LoRA fine-tuning is essential** for production-quality Vietnamese calligraphy.

### Rating Distribution (25 Test Cases)

| Rating | Count | Percentage |
|--------|-------|------------|
| Excellent | 1 | 4% |
| Good | 8 | 32% |
| Acceptable | 9 | 36% |
| Fail | 7 | 28% |

---

## 2. VNCAL-4W Composite Results

### 2.1 Full Results Table

| TC | Words | Content | CA | DIS | LAS | SCS | VNCAL-4W | Rating |
|----|-------|---------|----|-----|-----|-----|----------|--------|
| TC1 | 1 | Nhẫn | 0.7500 | 0.0000 | 0.4049 | 0.7920 | 0.5019 | Acceptable |
| TC2 | 2 | An Nhiên | 1.0000 | 1.0000 | 0.4277 | 0.8009 | 0.8457 | Good |
| TC3 | 3 | Vạn Sự An | 0.7778 | 0.0000 | 0.4400 | 0.8098 | 0.5222 | Acceptable |
| TC4 | 4 | Phúc đức trường tồn | 0.8421 | 0.4000 | 0.4392 | 0.7446 | 0.6315 | Acceptable |
| TC5 | 1 | Tâm | 1.0000 | 1.0000 | 0.4184 | 0.7320 | 0.8301 | Good |
| TC6 | 1 | Phúc | 1.0000 | 1.0000 | 0.4304 | 0.7144 | 0.8290 | Good |
| TC7 | 1 | Lộc | 0.0000 | 0.0000 | 0.4386 | 0.7371 | 0.2351 | Fail |
| TC8 | 1 | Trí | 0.5000 | 0.0000 | 0.4210 | 0.7739 | 0.4140 | Fail |
| TC9 | 2 | Bình An | 0.5714 | 0.0000 | 0.4386 | 0.7389 | 0.4355 | Fail |
| TC10 | 2 | Hạnh Phúc | 1.0000 | 1.0000 | 0.4365 | 0.7885 | 0.8450 | Good |
| TC11 | 2 | Tri Ân | 1.0000 | 1.0000 | 0.4421 | 0.7561 | 0.8396 | Good |
| TC12 | 2 | An Khang | 1.0000 | 1.0000 | 0.4330 | 0.7642 | 0.8394 | Good |
| TC13 | 3 | Tâm Bất Biến | 0.6667 | 0.3333 | 0.4356 | 0.7328 | 0.5503 | Acceptable |
| TC14 | 3 | Phúc Mãn Đường | 0.7857 | 0.5000 | 0.4479 | 0.7590 | 0.6414 | Acceptable |
| TC15 | 3 | Thuận Tự Nhiên | 0.7143 | 0.3333 | 0.4441 | 0.7699 | 0.5761 | Acceptable |
| TC16 | 4 | Vạn sự như ý | 0.3571 | 0.0000 | 0.8447 | 0.7325 | 0.4404 | Fail |
| TC17 | 4 | An khang thịnh vượng | 0.5238 | 0.0000 | 0.4468 | 0.7517 | 0.4230 | Fail |
| TC18 | 1 | Nhẫn (horizontal) | 0.7500 | 0.0000 | 0.8403 | 0.8080 | 0.5922 | Acceptable |
| TC19 | 4 | Phúc đức trường tồn (horiz.) | 0.5500 | 0.2000 | 0.4444 | 0.7151 | 0.4744 | Fail |
| TC20 | 2 | Tài Lộc (red ink) | 0.8571 | 0.5000 | 0.4409 | 0.7452 | 0.6622 | Acceptable |
| TC21 | 3 | Tâm Bất Biến (gold/wood) | 0.6923 | 0.3333 | 0.4500 | 0.5025 | 0.5161 | Acceptable |
| TC22 | 1 | Tâm (no stamps) | 1.0000 | 1.0000 | 0.4148 | 0.7573 | 0.8344 | Good |
| TC23 | 2 | 8028 (numerals) | 1.0000 | 1.0000 | 0.7836 | 0.6925 | 0.8952 | Excellent |
| TC24 | 2 | Thanh Tâm (complex prompt) | 1.0000 | 1.0000 | 0.4444 | 0.7442 | 0.8377 | Good |
| TC25 | 3 | Nhẫn Nhường Nhịn | 0.5000 | 0.0000 | 0.4462 | 0.5900 | 0.3822 | Fail |
| **AVG** | | | **0.7535** | **0.4640** | **0.4822** | **0.7381** | **0.6238** | **Acceptable** |

### 2.2 Averages by Word Count

| Word Count | #TCs | CA | DIS | LAS | SCS | VNCAL-4W | Trend |
|------------|------|----|-----|-----|-----|----------|-------|
| 1-Word | 8 | 0.8438 | 0.5000 | 0.4733 | 0.7646 | 0.6421 | Best overall |
| 2-Word | 8 | 0.9286 | 0.7500 | 0.4809 | 0.7538 | 0.7750 | Strong CA+DIS |
| 3-Word | 5 | 0.6714 | 0.2000 | 0.4452 | 0.6710 | 0.5130 | DIS collapse |
| 4-Word | 4 | 0.5683 | 0.1500 | 0.5438 | 0.7360 | 0.4923 | Weakest |

> **Key Insight:** Performance degrades sharply beyond 2 words. At 3–4 words, DIS drops to 0.15–0.20 (systematic diacritic failure), confirming the base model cannot reliably render Vietnamese diacritics at reduced character sizes.

---

## 3. Per-Test-Case Detailed Results

### 3.1 Core Diacritic & Length Tests (TC1–TC17)

#### TC1: `Nhẫn` (1W — Stacked Diacritics `ẫ`)
- **OCR Output:** `Nhấn` — Wrong tone: ngã (ẫ) rendered as sắc (ấ)
- **DIS Failure:** `ẫ` -> `ấ` (wrong_tone) at position 2
- CA=0.7500, DIS=0.0000, LAS=0.4049, SCS=0.7920 | **VNCAL=0.5019 (Acceptable)**
- Layout: orientation=0.00 (horizontal instead of vertical), centering=0.8496
- SCS: clip=0.5949, stroke=0.9817, bg=1.0000

#### TC2: `An Nhiên` (2W — Mixed Tones `ê`)
- **OCR Output:** `An Nhiên` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.4277, SCS=0.8009 | **VNCAL=0.8457 (Good)**
- Layout: orientation=0.00 (horizontal), centering=0.9258
- SCS: clip=0.6018, stroke=1.0000, bg=1.0000

#### TC3: `Vạn Sự An` (3W — Dot Below `ạ`, `ự`)
- **OCR Output:** `Vân Sư An` — Diacritics dropped
- **DIS Failures:**
  - `ạ` -> `â` (multiple_diacritic_errors) at position 1
  - `ự` -> `ư` (dropped_tone) at position 5
- CA=0.7778, DIS=0.0000, LAS=0.4400, SCS=0.8098 | **VNCAL=0.5222 (Acceptable)**

#### TC4: `Phúc đức trường tồn` (4W — Maximum Stress)
- **OCR Output:** `Phúc đức trông tôn`
- **DIS Failures:**
  - `ư` -> (dropped_character) at position 11
  - `ờ` -> `ô` (multiple_diacritic_errors) at position 12
  - `ồ` -> `ô` (dropped_tone) at position 17
- CA=0.8421, DIS=0.4000, LAS=0.4392, SCS=0.7446 | **VNCAL=0.6315 (Acceptable)**

#### TC5: `Tâm` (1W — Base Vowel `â`)
- **OCR Output:** `Tâm` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.4184, SCS=0.7320 | **VNCAL=0.8301 (Good)**

#### TC6: `Phúc` (1W — Acute `ú`)
- **OCR Output:** `Phúc` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.4304, SCS=0.7144 | **VNCAL=0.8290 (Good)**

#### TC7: `Lộc` (1W — Dot Below `ộ`)
- **OCR Output:** `上善` — Completely wrong, rendered as Chinese characters
- **DIS Failure:** `ộ` -> `上` (dropped_all_diacritics)
- CA=0.0000, DIS=0.0000, LAS=0.4386, SCS=0.7371 | **VNCAL=0.2351 (Fail)**
- **Root cause:** Model's Chinese calligraphy prior completely overrode the Vietnamese text instruction

#### TC8: `Trí` (1W — Acute Narrow `í`)
- **OCR Output:** `Trái` — Base vowel corrupted
- **DIS Failure:** `í` -> `á` (unknown error type)
- CA=0.5000, DIS=0.0000, LAS=0.4210, SCS=0.7739 | **VNCAL=0.4140 (Fail)**

#### TC9: `Bình An` (2W — Grave `ì`)
- **OCR Output:** `Biển An` — `ình` -> `iển`
- **DIS Failure:** `ì` -> `i` (dropped_tone)
- CA=0.5714, DIS=0.0000, LAS=0.4386, SCS=0.7389 | **VNCAL=0.4355 (Fail)**

#### TC10: `Hạnh Phúc` (2W — Multiple Tones `ạ`, `ú`)
- **OCR Output:** `Hạnh Phúc` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.4365, SCS=0.7885 | **VNCAL=0.8450 (Good)**

#### TC11: `Tri Ân` (2W — Circumflex Cap `Â`)
- **OCR Output:** `Tri Ân` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.4421, SCS=0.7561 | **VNCAL=0.8396 (Good)**

#### TC12: `An Khang` (2W — Pure Baseline, no diacritics)
- **OCR Output:** `An Khang` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.4330, SCS=0.7642 | **VNCAL=0.8394 (Good)**

#### TC13: `Tâm Bất Biến` (3W — Combos `â`, `ấ`, `ế`)
- **OCR Output:** `Tấm Bất Biển`
- **DIS Failures:**
  - `â` -> `ấ` (wrong_tone) at position 1
  - `ế` -> `ể` (wrong_tone) at position 10
- CA=0.6667, DIS=0.3333, LAS=0.4356, SCS=0.7328 | **VNCAL=0.5503 (Acceptable)**

#### TC14: `Phúc Mãn Đường` (3W — Tilde and Horn `ã`, `ờ`)
- **OCR Output:** `Phúc Mãn Doùng`
- **DIS Failures:**
  - `ư` -> `o` (dropped_modifier) at position 10
  - `ờ` -> `ù` (wrong_modifier) at position 11
- CA=0.7857, DIS=0.5000, LAS=0.4479, SCS=0.7590 | **VNCAL=0.6414 (Acceptable)**

#### TC15: `Thuận Tự Nhiên` (3W — Dot Below `ậ`, `ự`)
- **OCR Output:** `Thuấn Tú Nhiên`
- **DIS Failures:**
  - `ậ` -> `ấ` (wrong_tone) at position 3
  - `ự` -> `ú` (multiple_diacritic_errors) at position 7
- CA=0.7143, DIS=0.3333, LAS=0.4441, SCS=0.7699 | **VNCAL=0.5761 (Acceptable)**

#### TC16: `Vạn sự như ý` (4W — Stress `ạ`, `ự`, `ư`, `ý`)
- **OCR Output:** `Vẫn bùi háu ốy` — Severely corrupted
- **DIS Failures:** All 4 diacritic characters failed (multiple_diacritic_errors)
- CA=0.3571, DIS=0.0000, LAS=0.8447, SCS=0.7325 | **VNCAL=0.4404 (Fail)**
- Note: Only TC where LAS orientation=1.0 (vertical detected) — possibly because severe corruption aligned vertically

#### TC17: `An khang thịnh vượng` (4W — Stress `ị`, `ượ`)
- **OCR Output:** `An kâng thành vulgóng` — Severely corrupted
- **DIS Failures:** `ị` -> `à`, `ư` -> `u`, `ợ` -> `l`
- CA=0.5238, DIS=0.0000, LAS=0.4468, SCS=0.7517 | **VNCAL=0.4230 (Fail)**

### 3.2 Edge Case & Stress Tests (TC18–TC25)

#### TC18: `Nhẫn` — Horizontal Layout Override
- **OCR Output:** `Nhấn` — Same wrong_tone error as TC1
- CA=0.7500, DIS=0.0000, LAS=0.8403, SCS=0.8080 | **VNCAL=0.5922 (Acceptable)**
- LAS orientation=1.0 — Horizontal layout successfully detected (vs TC1 vertical failed)

#### TC19: `Phúc đức trường tồn` — Horizontal 4-Word
- **OCR Output:** `Phúc dịch ctrông lổn` — Severely corrupted
- **DIS Failures:** `ứ`->`ị`, `ư`->dropped, `ờ`->`ô`, `ồ`->`ổ`
- CA=0.5500, DIS=0.2000, LAS=0.4444, SCS=0.7151 | **VNCAL=0.4744 (Fail)**

#### TC20: `Tài Lộc` — Red Ink on Gold Paper
- **OCR Output:** `Tái Lộc` — Subtle tone swap
- **DIS Failure:** `à` -> `á` (wrong_tone)
- CA=0.8571, DIS=0.5000, LAS=0.4409, SCS=0.7452 | **VNCAL=0.6622 (Acceptable)**

#### TC21: `Tâm Bất Biến` — Gold Ink on Wooden Plaque
- **OCR Output:** `Tâm Phật Biển`
- **DIS Failures:** `ấ` -> `h` (dropped_all), `ế` -> `ể` (wrong_tone)
- CA=0.6923, DIS=0.3333, LAS=0.4500, SCS=0.5025 | **VNCAL=0.5161 (Acceptable)**
- SCS background=0.1164 — Wooden plaque texture destroyed background cleanliness

#### TC22: `Tâm` — Negative Constraint (no stamps/seals)
- **OCR Output:** `Tâm` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.4148, SCS=0.7573 | **VNCAL=0.8344 (Good)**
- Model correctly followed negative constraint

#### TC23: `8028` — Out-of-Domain (Arabic Numerals)
- **OCR Output:** `8028` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.7836, SCS=0.6925 | **VNCAL=0.8952 (Excellent)**
- Best overall score — numerals are well within the model's training distribution
- LAS orientation=1.0 (layout detected correctly)

#### TC24: `Thanh Tâm` — Complex Descriptive Prompt
- **OCR Output:** `Thanh Tâm` — Perfect match
- CA=1.0000, DIS=1.0000, LAS=0.4444, SCS=0.7442 | **VNCAL=0.8377 (Good)**
- Model handled prompt distractors well

#### TC25: `Nhẫn Nhường Nhịn` — Alliteration Stress (Nh×3)
- **OCR Output:** `Nhấn Nhuận Nín` — All diacritics corrupted
- **DIS Failures:** `ẫ`->`ấ`, `ư`->`u`, `ờ`->`ậ`, `ị`->`í`
- CA=0.5000, DIS=0.0000, LAS=0.4462, SCS=0.5900 | **VNCAL=0.3822 (Fail)**
- Background cleanliness=0.000 — Complete background contamination

---

## 4. Metric Breakdown Analysis

### 4.1 Character Accuracy (CA) — 0.7535 (Acceptable)

| Rating | Count | Test Cases |
|--------|-------|------------|
| Excellent (>=0.95) | 10 | TC2, TC5, TC6, TC10, TC11, TC12, TC22, TC23, TC24 (+ TC12 no diacritics) |
| Good (0.85–0.94) | 2 | TC4, TC20 |
| Acceptable (0.70–0.84) | 5 | TC1, TC3, TC14, TC15, TC18 |
| Fail (<0.70) | 8 | TC7, TC8, TC9, TC13, TC16, TC17, TC19, TC25 |

**Key finding:** CA is high (>=0.85) when words are simple (1–2 words, common diacritics) but degrades sharply for complex or long content.

### 4.2 Diacritic Integrity (DIS) — 0.4640 (Fail)

| Rating | Count | Test Cases |
|--------|-------|------------|
| Excellent (>=0.90) | 10 | TC2, TC5, TC6, TC10, TC11, TC12, TC22, TC23, TC24 (all perfect) |
| Acceptable (0.50–0.69) | 3 | TC14, TC20 (+ TC4 at 0.40) |
| Fail (<0.50) | 12 | TC1, TC3, TC7, TC8, TC9, TC13, TC15, TC16, TC17, TC18, TC19, TC25 |

**Diacritic failure taxonomy (observed):**

| Failure Type | Count | Examples |
|-------------|-------|---------|
| **wrong_tone** | 8 | `ẫ`->`ấ` (TC1, TC18, TC25), `â`->`ấ` (TC13), `à`->`á` (TC20) |
| **dropped_tone** | 3 | `ự`->`ư` (TC3), `ì`->`i` (TC9), `ồ`->`ô` (TC4) |
| **multiple_diacritic_errors** | 9 | `ạ`->`â` (TC3), `ự`->`ú` (TC15), all 4W cases |
| **dropped_modifier** | 2 | `ư`->`o` (TC14), `ư`->`u` (TC17, TC25) |
| **dropped_all_diacritics** | 2 | `ộ`->`上` (TC7), `ấ`->`h` (TC21) |
| **dropped_character** | 1 | `ư`->`` (TC4, TC19) |

**Critical pattern:** The stacked diacritic `ẫ` (â + ngã) is consistently rendered as `ấ` (â + sắc) — the model confuses the ngã (~) and sắc (´) tone marks. This is the single most predictable failure.

### 4.3 Layout Adherence (LAS) — 0.4822 (Fail)

| Sub-metric | Average | Rating |
|------------|---------|--------|
| **Orientation** | 0.12 | Fail — Only 3/25 cases correctly detected orientation |
| **Centering** | 0.9494 | Excellent — Consistently well-centered |
| **Spacing** | 0.4916 | Fail — Uniform spacing not achieved |

**LAS detail: Orientation failures**

| Orientation | Requested | Correct | Incorrect | Pass Rate |
|-------------|-----------|---------|-----------|-----------|
| Vertical | 21 | 1 (TC16) | 20 | 4.8% |
| Horizontal | 2 (TC18, TC19) | 2 | 0 | 100% |
| N/A | 2 (TC23 detected correct) | — | — | — |

> **Critical finding:** The base model almost **never** produces vertical layout for Vietnamese Latin-script text (4.8% pass rate). It only succeeds at horizontal layout. This confirms the framework's prediction that MSRoPE has no prior for vertical Latin text.

### 4.4 Stylistic Consistency (SCS) — 0.7381 (Good)

| Sub-metric | Average | Rating |
|------------|---------|--------|
| **CLIP Similarity** | 0.5917 | Moderate — General calligraphic style detected |
| **Stroke Profile** | 0.8556 | Good — Stroke widths somewhat match reference |
| **Background Cleanliness** | 0.9247 | Excellent — Clean backgrounds in most cases |

**Outliers:**
- TC21 (gold/wood plaque): bg=0.1164 — textured background heavily penalized
- TC25 (alliteration): bg=0.0000 — complete background contamination
- TC23 (numerals): clip=0.4786 — lowest CLIP, expected since numerals don't match calligraphy reference

---

## 5. SPT Spatial Tests

### 5.1 SPT-01: Character-Count Fidelity
- Evaluated via OCR word count vs expected word count
- **Overall:** Since OCR frequently fails on diacritics, some counts match even when text is wrong

### 5.2 SPT-02: Hallucination Detection
- **Result:** No hallucinated extra words detected in any test case
- **Pass rate:** 25/25 (100%)

### 5.3 SPT-03: Layout Compliance (Vertical Aspect Ratio > 2.0)
- **Result:** Almost all vertical-requested images have aspect ratio < 2.0
- **Pass rate:** ~3/25 — Consistent horizontal bias confirmed

### 5.4 SPT-04: Centering Accuracy (Offset < 15%)
- **Result:** All test cases well-centered
- **Pass rate:** 25/25 (100%)
- Average offset: ~3–8% of image dimension

### 5.5 SPT-05: Scale Proportionality (Cross-case)
- **Result:** h(1W) / h(4W) = **2.43** (threshold > 1.5)
- **PASS** — The model correctly scales character size: 1-word characters (~508px) are 2.43x taller than 4-word characters (~209px)

---

## 6. Encoder Tests (ENC-01 to ENC-04)

| Test | Status | Detail |
|------|--------|--------|
| **ENC-01** Token-level Inspection | **PASS** (6/6) | All diacritic characters have unique token IDs distinct from bare forms |
| **ENC-04** NFC Normalization | **PASS** (8/8) | NFC and NFD produce identical token IDs |
| ENC-02 Hidden-state Similarity | Skipped | Requires full encoder model loading |
| ENC-03 Prompt-format Fidelity | Skipped | Requires full encoder model loading |

### ENC-01 Token ID Verification

| Diacritic | Token ID | Bare Form | Token ID | Distinct? |
|-----------|----------|-----------|----------|-----------|
| `ẫ` | [124503] | `a` | [64] | PASS |
| `ườ` | [51990] | `uo` | [23137] | PASS |
| `ức` | [94963] | `uc` | [1754] | PASS |
| `ạ` | [20229] | `a` | [64] | PASS |
| `ồ` | [70680] | `o` | [78] | PASS |
| `ệ` | [25232] | `e` | [68] | PASS |

**Conclusion:** The tokenizer correctly distinguishes Vietnamese diacritics at the token level. The generation failures are downstream (MMDiT/VAE), not encoder-level.

---

## 7. VAE Tests (VAE-01 to VAE-05)

### 7.1 Summary

| Test | Pass/Fail | Samples | Key Metric |
|------|-----------|---------|------------|
| **VAE-01** Roundtrip (PSNR/SSIM) | **PASS** (20/20) | 5 per word-count | Mean PSNR=50.31 dB, SSIM=0.9986 |
| **VAE-02** Stroke-width | **PASS** (20/20) | 5 per word-count | Max deviation=0.14px (threshold <=2px) |
| **VAE-03** Ink Bleeding | **FAIL** (15/20) | 5 per word-count | 5 failures (ratio < 0.70) |
| **VAE-04** Thin Stroke Survival | **PASS** (20/20) | 5 per word-count | Mean survival=99.05% (threshold >=85%) |
| **VAE-05** Diacritic Preservation | **PASS** (18/18) | 18 with diacritics | Mean SSIM=0.9982 (threshold >=0.92) |

### 7.2 VAE-01: Encode-Decode Roundtrip

| Word Count | PSNR Range (dB) | SSIM Range | All Pass? |
|------------|-----------------|------------|-----------|
| 1W (5 samples) | 50.11 – 52.10 | 0.9984 – 0.9987 | Yes |
| 2W (5 samples) | 48.78 – 51.45 | 0.9983 – 0.9987 | Yes |
| 3W (5 samples) | 48.51 – 50.37 | 0.9985 – 0.9988 | Yes |
| 4W (5 samples) | 48.97 – 51.61 | 0.9985 – 0.9988 | Yes |

All values far exceed thresholds (PSNR >= 35 dB, SSIM >= 0.97). The VAE reconstructs calligraphic images with very high fidelity.

### 7.3 VAE-03: Ink Bleeding Fidelity (Failed Test)

5 samples failed the ±30% tolerance on gradient ratio:

| Sample | Word Count | Ratio | Status |
|--------|-----------|-------|--------|
| syn_02293.jpg | 2W | 0.665 | FAIL |
| syn_04310.jpg | 3W | 0.600 | FAIL |
| syn_05270.jpg | 4W | 0.548 | FAIL |
| syn_00253.jpg | 4W | 0.567 | FAIL |
| syn_06118.jpg | 4W | 0.527 | FAIL |

**Pattern:** Failures concentrate at higher word counts (3W–4W) where characters are smaller and edge gradients are thinner. The VAE slightly smooths edge transitions at small scales.

### 7.4 VAE-04: Thin Stroke Survival

| Word Count | Survival Range | Mean | Thin Pixels (avg) |
|------------|---------------|------|-------------------|
| 1W | 98.6% – 100% | 99.0% | 164 |
| 2W | 97.8% – 100% | 99.0% | 255 |
| 3W | 98.1% – 99.7% | 99.1% | 459 |
| 4W | 98.9% – 99.8% | 99.2% | 529 |

Excellent thin stroke preservation across all word counts.

---

## 8. Failure Mode Analysis

### 8.1 Confirmed Failure Modes (vs Framework Predictions)

#### Failure Mode 1: Diacritic Corruption — **CONFIRMED (Severity: CRITICAL)**

- **Predicted:** >= 70% of test cases would have diacritic failures
- **Actual:** 12/25 test cases (48%) have DIS=0 (complete diacritic failure); only 10/25 achieve DIS=1.0
- **Most common error:** Tone mark confusion (ngã ↔ sắc), horn/modifier dropping
- **Pattern:** Stacked diacritics (ẫ, ậ, ự) fail most reliably; simple diacritics (â, ú, ê) often succeed

#### Failure Mode 2: Layout Default to Horizontal — **CONFIRMED (Severity: CRITICAL)**

- **Predicted:** >= 60% horizontal default despite vertical instruction
- **Actual:** 95.2% (20/21) of vertical-requested images rendered horizontally
- **The model has zero capability for vertical Latin text layout**

#### Failure Mode 3: Style Hallucination / Font Mismatch — **PARTIALLY CONFIRMED (Severity: MEDIUM)**

- **Predicted:** Generic brush/Chinese calligraphy style
- **Actual:** SCS averages 0.7381 (Good) — better than expected
  - TC7 (`Lộc` -> `上善`) confirmed Chinese calligraphy hallucination
  - Most other cases maintain a general calligraphic appearance
  - Background cleanliness is generally excellent (SCS bg=0.92 avg)

### 8.2 Newly Discovered Failure Modes

#### Failure Mode 4: Chinese Character Substitution (NEW)
- TC7: Vietnamese `Lộc` was entirely replaced by Chinese characters `上善`
- The model's Chinese calligraphy prior can completely override Vietnamese text instructions
- Risk factor: Words that phonetically or semantically overlap with Chinese concepts

#### Failure Mode 5: Alliteration Sensitivity (NEW)
- TC25: Repeated `Nh` initial consonant caused cascading diacritic errors across all 3 words
- The model struggles when similar phonetic patterns repeat, compounding errors

#### Failure Mode 6: Texture-induced Quality Degradation (NEW)
- TC21 (wooden plaque): SCS dropped to 0.5025 (bg=0.1164)
- Non-standard backgrounds significantly degrade both OCR and style metrics

---

## 9. Predicted vs Actual Comparison

| Metric | Predicted (Framework) | Actual | Delta | Analysis |
|--------|----------------------|--------|-------|----------|
| **CA** | 0.30 | 0.7535 | +0.45 | Model renders base characters better than expected |
| **DIS** | 0.18 | 0.4640 | +0.28 | Still Fail, but less catastrophic than predicted |
| **LAS** | 0.40 | 0.4822 | +0.08 | Close to prediction, centering saves the score |
| **SCS** | 0.30 | 0.7381 | +0.44 | Model has better calligraphic priors than assumed |
| **VNCAL-4W** | 0.30 | 0.6238 | +0.32 | Acceptable vs predicted Fail |

### Why Actual Scores Exceeded Predictions

1. **Character rendering capability underestimated:** The base model can render simple Vietnamese characters (Tâm, Phúc, An Nhiên) correctly. The model has some Vietnamese text exposure.
2. **Centering is inherently good:** The MMDiT naturally centers generated content, providing a consistent LAS boost via the centering sub-metric.
3. **Background cleanliness:** The "clean background" prompt is well-understood by the model, keeping SCS bg scores high.
4. **2-word sweet spot:** The 2-word category performs unexpectedly well (VNCAL=0.78), suggesting the model has reasonable capability for short Vietnamese phrases.

### Why LoRA Is Still Essential

Despite the better-than-predicted baseline:
- **DIS=0.46 (Fail):** Diacritics are wrong in most complex cases. Vietnamese without correct diacritics is not Vietnamese.
- **LAS orientation=0.12:** Vertical layout is completely broken (95% failure rate).
- **3-4 word degradation:** CA drops from 0.93 (2W) to 0.57 (4W); DIS drops from 0.75 to 0.15.
- **Critical use cases fail:** TC1 (Nhẫn), the flagship stacked-diacritic test, gets DIS=0.

---

## 10. Conclusions & LoRA Training Implications

### 10.1 Priority Areas for LoRA Fine-tuning

| Priority | Area | Current | Target | Gap |
|----------|------|---------|--------|-----|
| **P0** | Diacritic rendering (DIS) | 0.46 | 0.80 | 0.34 |
| **P0** | Vertical layout (LAS orientation) | 0.12 | 0.90 | 0.78 |
| **P1** | 3-4 word accuracy (CA@3-4W) | 0.62 | 0.85 | 0.23 |
| **P2** | Font style matching (SCS) | 0.74 | 0.90 | 0.16 |

### 10.2 LoRA Training Recommendations

1. **Stacked diacritic emphasis:** The ẫ/ậ/ễ/ỗ/ự forms need heavy representation in training. Current consistent failure (ẫ→ấ) indicates the MMDiT cross-attention layers lack the fine-grained conditioning for tone mark distinction.

2. **Vertical layout conditioning:** The 8,028-image dataset with consistent `layout="vertical"` metadata should create the missing vertical-Latin association. This is likely the single largest improvement LoRA will deliver.

3. **4-word scaling:** Longer text at reduced character size compounds diacritic errors. Training should include proportional 4-word examples to calibrate the MMDiT's spatial scaling.

4. **Anti-hallucination:** The Chinese character substitution (TC7) should be eliminated by LoRA exposure to Vietnamese-specific calligraphy. The single-font dataset provides a strong style anchor.

### 10.3 Expected Post-LoRA Improvement

Based on the gap analysis and dataset coverage (8,028 images, ~6,500 in 1-4W range):

| Metric | Baseline | Expected Post-LoRA | Confidence |
|--------|----------|-------------------|------------|
| CA | 0.7535 | >= 0.90 | High |
| DIS | 0.4640 | >= 0.80 | Medium-High |
| LAS | 0.4822 | >= 0.85 | High (layout labels present) |
| SCS | 0.7381 | >= 0.90 | High (single font, strong prior) |
| VNCAL-4W | 0.6238 | >= 0.85 | Medium-High |

---

## Appendix: Output Files

| File | Description |
|------|-------------|
| `baseline_4w_results.json` | Full structured results — 25 TCs with all metrics, SPT, DIS failures |
| `baseline_4w_summary.csv` | Per-test-case metric scores for analysis |
| `enc_tests_results.json` | ENC-01, ENC-04 results (token IDs, NFC checks) |
| `vae_tests_results.json` | VAE-01→05 results (PSNR, SSIM, stroke, survival, diacritic) |
| `TC1_Nhẫn.jpg` ... `TC25_Nhẫn_Nhường_Nhịn.jpg` | 25 generated test images |

### Reproducibility

```bash
# Metrics self-test (no GPU)
python experiments/scripts/metrics_4w.py

# Encoder tests
python experiments/scripts/enc_tests.py \
    --tokenizer_path Qwen-Image-2512-Local/tokenizer \
    --output_dir experiments/results/baseline_4w

# Full baseline (requires A100 80GB)
python experiments/scripts/eval_baseline_4w.py \
    --model_root Qwen-Image-2512-Local \
    --dataset_root data/raw/dataset_8028_calligraphy_1font \
    --output_dir experiments/results/baseline_4w \
    --ocr_engine vintern_3b --device cuda:0

# VAE tests
python experiments/scripts/vae_tests.py \
    --model_root Qwen-Image-2512-Local \
    --dataset_root data/raw/dataset_8028_calligraphy_1font \
    --device cuda:0 --n_samples 20 \
    --output_dir experiments/results/baseline_4w
```

### Dependencies Note

Vintern-3B-R-beta requires `timm` package and a compatibility patch for transformers >= 5.x:
- `pip install timm`
- Patch `modeling_intern_vit.py`: Replace `[x.item() for x in torch.linspace(...)]` with `torch.linspace(..., device="cpu").tolist()`
- Patch `modeling_internvl_chat.py`: Add `_tied_weights_keys = []` to `InternVLChatModel` class

---

## 11. OCR Engine Accuracy Assessment

### 11.1 Motivation

During the baseline evaluation (Sections 2–4), all VNCAL-4W metrics relied on Vintern-3B OCR output. However, analysis revealed that the OCR engine exhibits **autocorrect behavior** — it silently normalizes unusual characters into well-formed Vietnamese words. For example, the base model actually rendered `Phụ́c` (with combining acute on ụ), but Vintern read it as `Phúc`. This means some baseline metric scores (particularly CA and DIS) may be **inflated**, because OCR "fixed" generation errors before they were measured.

To quantify this effect, a separate OCR accuracy evaluation was conducted using **human-verified ground truth** — the user visually inspected all 25 generated images and recorded the exact text rendered by the base model.

### 11.2 Human-Verified Ground Truth

The following table shows the actual text visible in each generated image (as verified by human inspection), compared to what Vintern-3B OCR reported:

| TC | Prompt GT | Human-Verified (actual image) | Vintern OCR Output | OCR Correct? |
|----|-----------|-------------------------------|-------------------|-------------|
| TC1 | Nhẫn | Nhấn | Nhấn | Yes |
| TC2 | An Nhiên | An Nhiên | An Nhiên | Yes |
| TC3 | Vạn Sự An | Vấn Sú An | Vân Sư An | No (CER=22%) |
| TC4 | Phúc đức trường tồn | Phúc dúch ctrông tón | Phúc đức trông tôn | No (CER=25%) |
| TC5 | Tâm | Tâm | Tâm | Yes |
| TC6 | Phúc | Phụ́c | Phúc | No (CER=40%) — autocorrected |
| TC8 | Trí | Trâi | Trái | No (CER=25%) |
| TC9 | Bình An | Bián An | Biển An | No (CER=14%) |
| TC10 | Hạnh Phúc | Hấnh Phụ̃c | Hạnh Phúc | No (CER=30%) — autocorrected |
| TC11 | Tri Ân | Tri Ân | Tri Ân | Yes |
| TC12 | An Khang | An Khang | An Khang | Yes |
| TC13 | Tâm Bất Biến | Tẫm Bấṭ Biân | Tấm Bất Biển | No (CER=25%) |
| TC14 | Phúc Mãn Đường | Phụ́c Mấn Doŭng | Phúc Mãn Doùng | No (CER=27%) |
| TC15 | Thuận Tự Nhiên | Thuấn Tú Nhiễn | Thuấn Tú Nhiên | No (CER=7%) |
| TC16 | Vạn sự như ý | Vấn đùs hău ốy | Vẫn bùi háu ốy | No (CER=29%) |
| TC17 | An khang thịnh vượng | An kâng thành vauglóng | An kâng thành vulgóng | No (CER=14%) |
| TC18 | Nhẫn | Nhấn | Nhấn | Yes |
| TC19 | Phúc đức trường tồn | Phụ́c diĉh ctnóng lổn | Phúc dịch ctrông lổn | No (CER=29%) |
| TC20 | Tài Lộc | Tâi Lộc | Tái Lộc | No (CER=14%) |
| TC21 | Tâm Bất Biến | Tấm Bhất Biềń | Tâm Phật Biển | No (CER=38%) |
| TC22 | Tâm | Tâm | Tâm | Yes |
| TC24 | Thanh Tâm | Thanh Tâm | Thanh Tâm | Yes |
| TC25 | Nhẫn Nhường Nhịn | Nhầ́n Nhŭå̀n Niín | Nhấn Nhuận Nín | No (CER=35%) |

### 11.3 OCR Accuracy Results

#### Vintern-3B on Generated Images (vs human-verified GT)

| Metric | Score |
|--------|-------|
| Exact Match Rate | **34.8%** (8/23) |
| Mean CER | **16.27%** |

#### Vintern-3B on Dataset Images (vs metadata GT)

| Metric | Score |
|--------|-------|
| Exact Match Rate | **80.0%** (24/30) |
| Mean CER | **1.53%** |
| 1W Exact | 100% (7/7) |
| 2W Exact | 86% (6/7) |
| 3W Exact | 86% (6/7) |
| 4-8W Exact | 73% (5/9) |

**Key finding:** Vintern-3B is highly accurate on clean calligraphy images from the dataset (CER 1.53%), but struggles on AI-generated images with distorted characters (CER 16.27%). The gap confirms that OCR errors on generated images are significant and must be accounted for.

### 11.4 OCR Autocorrect Behavior

The most impactful OCR behavior observed is **autocorrect** — Vintern-3B silently normalizes unusual or malformed characters into valid Vietnamese words:

| TC | Actual Image Text | OCR Output | What Happened |
|----|-------------------|------------|---------------|
| TC6 | Phụ́c (combining acute on ụ) | Phúc | Normalized unusual Unicode to standard form |
| TC10 | Hấnh Phụ̃c (combining tilde on ụ) | Hạnh Phúc | Auto-corrected to real Vietnamese words |
| TC21 | Tấm Bhất Biềń | Tâm Phật Biển | "Bhất" → "Phật" (recognized as Buddhist term) |
| TC4 | Phúc dúch ctrông tón | Phúc đức trông tôn | Corrected multiple misspellings |

**Impact on baseline metrics:** This autocorrect inflates CA and DIS scores. For example, TC10 scored CA=1.0 and DIS=1.0 in the baseline evaluation because OCR reported `Hạnh Phúc` (matching the prompt), but the image actually shows `Hấnh Phụ̃c` (wrong diacritics). The true generation quality is worse than the metrics suggest.

### 11.5 Prompt Engineering for OCR Accuracy

To test whether OCR autocorrect can be mitigated, a strict literal prompt was designed and tested against the default Vietnamese prompt on both Vintern-3B and Qwen2.5-VL-7B.

#### Prompts Tested

| ID | Language | Prompt |
|----|----------|--------|
| **Default** | Vietnamese | `Trích xuất chính xác các chữ thư pháp trong ảnh. Chỉ trả về văn bản, không mô tả.` |
| **Strict** | English | Detailed prompt with rules: NO AUTOCORRECT, sắc vs ngã geometry rule, hallucinated underdots warning, Unicode combining instruction. (~600 tokens) |

#### Results: Vintern-3B

| Prompt | Gen CER | Gen Exact | Dataset CER | Dataset Exact |
|--------|---------|-----------|-------------|---------------|
| Default (VI) | **16.3%** | **35%** (8/23) | **0.7%** | **93%** (14/15) |
| Strict Literal (EN) | **15.0%** | **35%** (8/23) | 1.0% | 87% (13/15) |

**Per-TC improvements with Strict prompt (Vintern-3B):**

| TC | Default CER | Strict CER | Delta | What Changed |
|----|-------------|------------|-------|-------------|
| TC4 | 25.0% | 15.0% | -10.0% | `đức` → `dích` (closer to actual `dúch`) |
| TC10 | 30.0% | 20.0% | -10.0% | `Hạnh Phúc` → `Hạnh Phục` (less autocorrect) |
| TC17 | 13.6% | 4.5% | -9.1% | `vulgóng` → `vuglóng` (closer to actual) |
| TC21 | 38.5% | 30.8% | -7.7% | Less aggressive word substitution |
| TC16 | 28.6% | 35.7% | +7.1% | `háu` → `hầu` (overcorrected differently) |

**Net effect:** Strict prompt reduces Generated CER by 1.3% but slightly hurts Dataset accuracy. Same exact match count.

#### Results: Qwen2.5-VL-7B

| Prompt | Gen CER | Gen Exact | Dataset CER | Dataset Exact |
|--------|---------|-----------|-------------|---------------|
| Default (VI) | 15.2% | 35% (8/23) | 17.6% | 27% (4/15) |
| Strict Literal (EN) | 18.0% | 26% (6/23) | 7.9% | 40% (6/15) |

**Key observations:**
- Strict prompt **worsened** Qwen on generated images (+2.85% CER, lost 2 exact matches including TC18 and TC22 which were previously perfect)
- Strict prompt **improved** Qwen on dataset images (-9.76% CER), likely because it stopped hallucinating Latin characters (`Trí`→`Trix`, `Sơn`→`Soon` no longer occurred)

### 11.6 Engine Comparison Summary

| Criterion | Vintern-3B | Qwen2.5-VL-7B | Winner |
|-----------|------------|----------------|--------|
| **Dataset OCR accuracy** | CER 0.7%, Exact 93% | CER 17.6%, Exact 27% | **Vintern-3B** (by far) |
| **Generated image OCR** | CER 16.3%, Exact 35% | CER 15.2%, Exact 35% | Tie |
| **Autocorrect tendency** | Moderate | Low-Moderate | — |
| **Hallucination risk** | Low | High (`Trí`→`Trix`) | **Vintern-3B** |
| **Prompt sensitivity** | Low (stable) | High (unstable) | **Vintern-3B** |
| **VRAM usage** | ~7 GB | ~16 GB | **Vintern-3B** |

### 11.7 Conclusions

1. **Vintern-3B remains the best OCR engine** for this project — excellent on clean calligraphy (CER 0.7%), stable across prompts, low VRAM.

2. **Qwen2.5-VL-7B is not suitable** as primary OCR — high hallucination rate on clean images, unstable with prompt changes, 2x VRAM cost.

3. **Strict literal prompts provide marginal improvement** on Vintern-3B generated images (CER 16.3% → 15.0%) but are not transformative. The OCR autocorrect behavior is deeply embedded in the model's weights and cannot be fully overridden by prompting alone.

4. **Baseline metric inflation:** The VNCAL-4W scores in Section 2 are likely **optimistically biased** by ~5-10% due to OCR autocorrect. True generation quality is worse than measured — strengthening the case for LoRA fine-tuning.

5. **Recommendation for future evaluations:** Use the human-verified ground truth approach (as in `eval_ocr_accuracy.py`) for critical assessments, and be aware that OCR-based metrics represent an upper bound on actual quality.

### 11.8 Scripts & Reproducibility

```bash
# OCR accuracy evaluation (single engine)
python experiments/scripts/eval_ocr_accuracy.py \
    --ocr_engine vintern_3b \
    --device cuda:0 \
    --dataset_root data/raw/dataset_8028_calligraphy_1font \
    --n_dataset_samples 30

# Prompt comparison (Vintern + Qwen)
python experiments/scripts/eval_ocr_prompt_compare.py \
    --engines vintern_3b,qwen \
    --device cuda:0 \
    --dataset_root data/raw/dataset_8028_calligraphy_1font \
    --n_dataset_samples 15
```

**Output files:**
- `experiments/results/ocr_accuracy/ocr_accuracy_results.json`
- `experiments/results/ocr_accuracy/ocr_prompt_compare_results.json`

---

## 12. Expanded OCR Engine Survey

### 12.1 Motivation

Section 11 concluded that Vintern-3B is the best OCR engine for this project but also revealed significant OCR autocorrect behavior that inflates baseline metrics. To find engines that are more literal (less autocorrect) and to prepare a multi-engine cross-validation pipeline, a broader survey of open-source and API-based OCR engines was conducted.

### 12.2 Engines Evaluated

#### Open-source / Local Models

| Engine | Model ID | Type | Size | Languages |
|--------|----------|------|------|-----------|
| **PaddleOCR** | PP-OCRv4 (via `paddleocr 2.10.0`) | Detection + Recognition | ~50 MB | vi, en, zh |
| **PaddleOCR-VL-1.5** | `PaddlePaddle/PaddleOCR-VL-1.5` | Vision-Language (ERNIE-4.5-0.3B) | ~0.9 GB | vi, en, zh |
| **GOT-OCR 2.0** | `stepfun-ai/GOT-OCR-2.0-hf` | End-to-end VLM | ~0.5 GB | vi, en |
| **SVTRv2** | PP-OCRv4 backend (SVTRv2 recognition) | Detection + Recognition | ~50 MB | vi |
| **PARSeq** | `baudm/parseq` (torch.hub) | Scene Text Recognition | ~91 MB | en only |
| **dots.ocr** | `rednote-hilab/dots.ocr` | Document Layout VLM | ~2.8 GB | vi, en |

#### Free / Freemium APIs

| Engine | Model | Type | Cost | Languages |
|--------|-------|------|------|-----------|
| **Gemini 3 Flash** | `gemini-3-flash-preview` | Cloud VLM | Free (5 req/min) | vi, en |
| **OCR.space** | Free API | Cloud OCR | Free (500 req/day) | eng only (free tier) |

### 12.3 Installation & Compatibility

| Package | Version | Notes |
|---------|---------|-------|
| `paddleocr` | **2.10.0** (downgraded from 3.4.0) | paddleocr 3.4 incompatible with paddlepaddle-gpu 2.6.2 |
| `paddlepaddle-gpu` | 2.6.2 | Pre-installed, CUDA compatible |
| `paddlex` | 3.4.3 | For PaddleOCR-VL-1.5 support |
| `nltk` | 3.9.4 | **Newly installed** — required by PARSeq via torch.hub |
| `astor` | 0.8.1 | **Newly installed** — required by paddlepaddle JIT |
| `libgl1` | system | **Newly installed** — required by OpenCV (PaddleOCR) |
| `google-generativeai` | 0.8.6 | Pre-installed; deprecated in favor of `google.genai` |
| `transformers` | 5.4.0 | Required compatibility patches for PaddleOCR-VL-1.5 and dots.ocr |

#### Transformers 5.x Compatibility Patches

Several models required monkey-patches to work with transformers 5.4.0:

| Model | Issue | Patch |
|-------|-------|-------|
| **PaddleOCR-VL-1.5** | `ROPE_INIT_FUNCTIONS` missing `'default'` type | Register custom `_compute_default_rope` function |
| **PaddleOCR-VL-1.5** | `RotaryEmbedding.compute_default_rope_parameters` missing | Skip in `_init_weights` for custom RotaryEmbedding |
| **PaddleOCR-VL-1.5** | `cache_position` is None in `prepare_inputs_for_generation` | Provide explicit `cache_position=torch.arange(seq_len)` |
| **dots.ocr** | `mm_token_type_ids` not accepted by `model.generate()` | Delete key from inputs before generate |
| **dots.ocr** | `cache_position` is None | Same fix as PaddleOCR-VL-1.5 |
| **GOT-OCR 2.0** | `AutoModel` → no `.chat()` method | Use `GotOcr2ForConditionalGeneration` + trim input tokens |
| **PARSeq** | `model.get_transform()` removed | Build transform manually: Resize + ToTensor + Normalize(0.5, 0.5) |

### 12.4 Integration Test Results

Test image: `syn_00040.jpg` (1-word calligraphy, ground truth: `Danh`)

| Engine | Status | Output | Inference Time | VRAM | Notes |
|--------|--------|--------|---------------|------|-------|
| **Gemini 3 Flash** | **OK** | `Danh` | 22s | 0 (cloud) | Best accuracy; rate limited 5 req/min |
| **PaddleOCR** | **OK** | `an` | 4.9s | ~0.5 GB | Fast, missed initial `D` |
| **SVTRv2** | **OK** | `an` | 1.8s | ~0.5 GB | Same backend as PaddleOCR |
| **PARSeq** | **OK** | `Dank` | 5.4s | ~0.3 GB | English-only, close but wrong last char |
| **GOT-OCR 2.0** | **OK** | `D arh...` (noisy) | 10s | ~1.5 GB | Repetition issues on calligraphy |
| **PaddleOCR-VL-1.5** | **OK** | noisy/garbled | 26s | ~2 GB | Works but struggles with calligraphy style |
| **dots.ocr** | **OK** | `""` (empty) | 5s | ~6 GB | Designed for document layout, not calligraphy |
| **OCR.space** | **OK** | `""` (empty) | 0.6s | 0 (cloud) | Free tier can't handle calligraphy images |

### 12.5 Engine Quality Assessment for Calligraphy

#### Tier 1 — Production-Ready for Calligraphy Evaluation

| Engine | Strengths | Weaknesses | Recommended Use |
|--------|-----------|------------|----------------|
| **Vintern-3B** (existing) | CER 0.7% on dataset, stable, low VRAM | Autocorrect behavior on generated images | Primary OCR for VNCAL-4W metrics |
| **Gemini 3 Flash** | Best accuracy on calligraphy (`Danh` correct), Vietnamese-aware | Rate limited (5 req/min), network latency, costs at scale | Cross-validation oracle; human-GT replacement for small batches |

#### Tier 2 — Usable with Limitations

| Engine | Strengths | Weaknesses | Recommended Use |
|--------|-----------|------------|----------------|
| **PaddleOCR / SVTRv2** | Very fast (1.8–5s), lightweight | Misses characters, no Vietnamese diacritics | Quick sanity check; character detection baseline |
| **PARSeq** | Fast, good for scene text | English-only, single word/line only | Latin character verification |
| **GOT-OCR 2.0** | General-purpose end-to-end | Repetition artifacts on calligraphy style | Experimental; needs better prompt engineering |

#### Tier 3 — Not Suitable for Calligraphy

| Engine | Reason |
|--------|--------|
| **dots.ocr** | Designed for document layout parsing (PDF/tables), returns empty on calligraphy images |
| **OCR.space (free)** | Free tier has no Vietnamese support; fails on artistic text |
| **PaddleOCR-VL-1.5** | Garbled output on calligraphy; transformers 5.x compatibility fragile |

### 12.6 Multi-Engine Cross-Validation Strategy

Based on the survey results, the recommended approach for future evaluations:

```
Primary:    Vintern-3B          (fast, accurate on clean images)
Secondary:  Gemini 3 Flash      (most accurate overall, rate limited)
Tertiary:   PaddleOCR           (fast sanity check, character detection)
Fallback:   GOT-OCR 2.0         (when VLM perspective needed)
```

**Cross-validation rule:** When Vintern-3B and Gemini disagree on a character, flag it for manual review. This catches Vintern's autocorrect behavior without requiring full human-GT verification.

### 12.7 Factory Usage

All engines are available via the unified factory function:

```python
from ocr_engines import get_ocr_engine

engine = get_ocr_engine("gemini")        # Gemini 3 Flash
engine = get_ocr_engine("gemini_lite")   # Gemini 3.1 Flash Lite
engine = get_ocr_engine("gemini_2_5")    # Gemini 2.5 Flash
engine = get_ocr_engine("paddle")        # PaddleOCR (traditional)
engine = get_ocr_engine("paddle_vl")     # PaddleOCR-VL-1.5
engine = get_ocr_engine("got")           # GOT-OCR 2.0
engine = get_ocr_engine("parseq")        # PARSeq
engine = get_ocr_engine("dots")          # dots.ocr
engine = get_ocr_engine("ocr_space")     # OCR.space
engine = get_ocr_engine("svtrv2")        # SVTRv2
engine = get_ocr_engine("deepseek_ocr")  # DeepSeek-OCR (FPT Cloud)
engine = get_ocr_engine("qwen3_vl")      # Qwen3-VL-8B-Instruct (FPT Cloud)

result = engine.recognize(image)
```

### 12.8 API Keys (`.env`)

```
GEMINI_API_KEY=AIzaSy...       # Google AI Studio - gemini-3-flash-preview
GEMINI_API_KEY_2=AIzaSy...     # Key rotation (5 keys total)
OCR_SPACE_API_KEY=K818...      # OCR.space free tier
FPT_CLOUD_BASE_URL=https://mkp-api.fptcloud.com
FPT_CLOUD_API_KEY=sk-JJg...    # FPT Cloud - DeepSeek-OCR & Qwen3-VL-8B
```

### 12.9 Scripts & Reproducibility

```bash
# Test all new engines
cd experiments/scripts
python test_new_engines.py

# Test specific engines
python test_new_engines.py gemini paddle got parseq

# Self-test utility functions
python ocr_engines.py
```

**Output:** Console report with per-engine status, inference time, and recognized text.

---

## 13. Comprehensive OCR Accuracy Evaluation

### 13.1 Motivation

Section 12 evaluated engines with a single smoke-test image (`syn_00040.jpg`, GT="Danh"). To determine which engines are truly viable for production use, a comprehensive 3-phase accuracy evaluation was conducted across all available engines — including two newly integrated FPT Cloud API engines (DeepSeek-OCR, Qwen3-VL-8B-Instruct) and additional Gemini model variants (gemini-3.1-flash-lite-preview, gemini-2.5-flash).

### 13.2 New Engines Added

#### FPT Cloud API Engines

| Engine | Model ID | API | Type | Cost |
|--------|----------|-----|------|------|
| **DeepSeek-OCR** | `DeepSeek-OCR` | FPT Cloud (OpenAI-compatible) | Cloud VLM | Free |
| **Qwen3-VL-8B-Instruct** | `Qwen3-VL-8B-Instruct` | FPT Cloud (OpenAI-compatible) | Cloud VLM | Free |

Both use the OpenAI SDK with base64-encoded images and prompt `"Free OCR."`:

```python
from openai import OpenAI
client = OpenAI(api_key="sk-...", base_url="https://mkp-api.fptcloud.com")
response = client.chat.completions.create(
    model="DeepSeek-OCR",  # or "Qwen3-VL-8B-Instruct"
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "Free OCR."},
    ]}],
    max_tokens=512, temperature=0.0,
)
```

#### Gemini Model Variants

| Factory Name | Model ID | Notes |
|---|---|---|
| `gemini` | `gemini-3-flash-preview` | Already existed (Section 12) |
| `gemini_lite` | `gemini-3.1-flash-lite-preview` | **New** — lighter, faster variant |
| `gemini_2_5` | `gemini-2.5-flash` | **New** — previous-gen flash model |

All Gemini variants use the existing `GeminiOCREngine` with key rotation (5 API keys, ~25 RPM / 100 RPD total).

### 13.3 Evaluation Methodology

Three-phase evaluation using `test_all_engines_comprehensive.py`:

| Phase | Description | Images | Ground Truth Source |
|-------|-------------|--------|-------------------|
| **Phase 1** | Smoke test | 1 image (`syn_00040.jpg`) | Metadata: "Danh" |
| **Phase 2** | Generated baseline images | TC1–TC25 (23 with human-verified GT) | Human-verified (Section 11.2) |
| **Phase 3** | Dataset samples | 30 stratified samples from `dataset_8028_calligraphy_1font` | Metadata prompts |

Metrics: CER (Character Error Rate), WER (Word Error Rate), Exact Match Rate.

### 13.4 Phase 1 — Smoke Test Results

Test image: `syn_00040.jpg` (1-word calligraphy, ground truth: `Danh`)

| Engine | Output | CER | Time | Status |
|--------|--------|-----|------|--------|
| **Gemini 3 Flash** | `Danh` | **0.00%** | 24.7s | Best accuracy |
| **Gemini 3.1 Flash Lite** | `Danh` | **0.00%** | 2.6s | Best speed+accuracy |
| **Gemini 2.5 Flash** | `Dash` | 25.00% | 3.2s | Close, wrong last char |
| **Qwen3-VL-8B** (FPT) | `Dark` | 50.00% | 1.5s | Fastest API |
| **PARSeq** | `Dank` | 25.00% | 22.9s | English-only, close |
| **SVTRv2** | `an` | 50.00% | 84.5s | Missed initial character |
| **DeepSeek-OCR** (FPT) | `& & &` | 125.00% | 2.1s | Garbled output |
| **OCR.space** | `""` | 100.00% | 0.9s | Empty — can't read calligraphy |
| **dots.ocr** | `""` | 100.00% | 9.8s | Empty — document parser, not OCR |
| **GOT-OCR 2.0** | `D arh,,...` (garbled) | 8950% | 13.4s | Severe repetition artifacts |
| **PaddleOCR-VL-1.5** | — | — | — | FAILED (transformers 5.x compat) |

### 13.5 Phase 2 — Generated Images (TC1–TC25, Human-Verified GT)

| Engine | Exact Match | Mean CER | Mean WER | Tier |
|--------|-------------|----------|----------|------|
| **Qwen3-VL-8B** (FPT) | **8/23 (34.8%)** | **17.03%** | 17.03% | Tier 1 |
| **Gemini 2.5 Flash** | 7/23 (30.4%) | 18.57% | 18.57% | Tier 1 |
| **Gemini 3.1 Flash Lite** | 5/23 (21.7%) | 19.68% | 19.68% | Tier 1 |
| **Gemini 3 Flash** | 7/23 (30.4%) | 19.10% | 19.10% | Tier 1 |
| **DeepSeek-OCR** (FPT) | 0/23 (0%) | 47.69% | 47.69% | Tier 3 |
| **SVTRv2** | 0/23 (0%) | 54.26% | 54.26% | Tier 3 |
| **PARSeq** | 0/23 (0%) | 60.45% | 60.45% | Tier 3 |

#### Per-TC Detail — Qwen3-VL-8B-Instruct (Best New Engine)

| TC | Human-Verified GT | Qwen3-VL Output | CER | Match |
|----|-------------------|-----------------|-----|-------|
| TC1 | Nhấn | Nhẫn | 25.0% | |
| TC2 | An Nhiên | An Nhiên | 0.0% | EXACT |
| TC3 | Vấn Sú An | Vân Sứ An | 22.2% | |
| TC4 | Phúc dúch ctrông tón | Phúc dúch ctrông tón | 0.0% | EXACT |
| TC5 | Tâm | Tâm | 0.0% | EXACT |
| TC6 | Phụ́c | Phúc | 40.0% | autocorrect |
| TC8 | Trâi | Trải | 25.0% | |
| TC9 | Bián An | Biễn An | 14.3% | |
| TC10 | Hấnh Phụ̃c | Hạnh Phúc | 30.0% | autocorrect |
| TC11 | Tri Ân | Tri Ân | 0.0% | EXACT |
| TC12 | An Khang | An Khang | 0.0% | EXACT |
| TC13 | Tẫm Bấṭ Biân | Tấm Bật Biên | 33.3% | |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mẫn Dưỡng | 33.3% | autocorrect |
| TC15 | Thuấn Tú Nhiễn | Thuần Tử Nhiên | 21.4% | autocorrect |
| TC16 | Vấn đùs hău ốy | Vẫn bùn hãu ốy | 28.6% | |
| TC17 | An kâng thành vauglóng | An kâng thành vauglóng | 0.0% | EXACT |
| TC18 | Nhấn | Nhẫn | 25.0% | |
| TC19 | Phụ́c diĉh ctnóng lổn | Phục dịch ctrông lỗn | 28.6% | |
| TC20 | Tâi Lộc | Tài Lộc | 14.3% | |
| TC21 | Tấm Bhất Biềń | Tầm Bhất Biền | 15.4% | |
| TC22 | Tâm | Tâm | 0.0% | EXACT |
| TC24 | Thanh Tâm | Thanh Tâm | 0.0% | EXACT |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhẫn Nhũn Niún | 35.3% | |

**Notable:** Qwen3-VL exhibits moderate autocorrect behavior (TC6, TC10, TC14, TC15) similar to Vintern-3B, but retains unusual character combinations better (TC4, TC17 exact matches where unusual spellings are preserved).

#### Per-TC Detail — Gemini 3.1 Flash Lite

| TC | Human-Verified GT | Gemini Lite Output | CER | Match |
|----|-------------------|-------------------|-----|-------|
| TC1 | Nhấn | Nhẫn | 25.0% | |
| TC2 | An Nhiên | An Nhiên | 0.0% | EXACT |
| TC3 | Vấn Sú An | Vấn Sự An | 11.1% | |
| TC4 | Phúc dúch ctrông tón | Phúc dúch trông tón | 5.0% | |
| TC5 | Tâm | Tẫm | 33.3% | |
| TC6 | Phụ́c | Phúc | 40.0% | autocorrect |
| TC8 | Trâi | Trải | 25.0% | |
| TC9 | Bián An | Biăn An | 14.3% | |
| TC10 | Hấnh Phụ̃c | Hạnh Phúc | 30.0% | autocorrect |
| TC11 | Tri Ân | Tri Ân | 0.0% | EXACT |
| TC12 | An Khang | An Khang | 0.0% | EXACT |
| TC13 | Tẫm Bấṭ Biân | Tâm Bất Biến | 25.0% | autocorrect |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mãn Đường | 40.0% | autocorrect |
| TC15 | Thuấn Tú Nhiễn | Thuần Tử Nhiên | 21.4% | autocorrect |
| TC16 | Vấn đùs hău ốy | Vẫn đùs hău ỐY | 21.4% | |
| TC17 | An kâng thành vauglóng | An kâng thành vauglóng | 0.0% | EXACT |
| TC18 | Nhấn | Nhẫn | 25.0% | |
| TC19 | Phụ́c diĉh ctnóng lổn | Phúc dịch trống lổn | 33.3% | |
| TC20 | Tâi Lộc | Tài Lộc | 14.3% | |
| TC21 | Tấm Bhất Biềń | Tâm Bhất Biến | 23.1% | |
| TC22 | Tâm | Tấm | 33.3% | |
| TC24 | Thanh Tâm | Thanh Tâm | 0.0% | EXACT |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhẫn Nhuần Nín | 35.3% | |

**Notable:** Gemini Lite shows strong autocorrect on TC13, TC14, TC15 (normalizing to valid Vietnamese words). Preserves unusual spellings in TC4 and TC17 better than Vintern-3B.

#### Per-TC Detail — Gemini 2.5 Flash

| TC | Human-Verified GT | Gemini 2.5 Output | CER | Match |
|----|-------------------|-------------------|-----|-------|
| TC1 | Nhấn | Nhẫn | 25.0% | |
| TC2 | An Nhiên | An Nhiên | 0.0% | EXACT |
| TC3 | Vấn Sú An | Vân Sử An | 22.2% | |
| TC4 | Phúc dúch ctrông tón | Phúc dúch trông tốn | 10.0% | |
| TC5 | Tâm | Tâm | 0.0% | EXACT |
| TC6 | Phụ́c | Phúc | 40.0% | autocorrect |
| TC8 | Trâi | Trái | 25.0% | |
| TC9 | Bián An | Biển An | 14.3% | |
| TC10 | Hấnh Phụ̃c | Hạnh Phúc | 30.0% | autocorrect |
| TC11 | Tri Ân | Tri Ân | 0.0% | EXACT |
| TC12 | An Khang | An Khang | 0.0% | EXACT |
| TC13 | Tẫm Bấṭ Biân | Tâm Bất Biên | 25.0% | |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mãn Đường | 40.0% | autocorrect |
| TC15 | Thuấn Tú Nhiễn | Thuận Tự Nhiên | 21.4% | autocorrect |
| TC16 | Vấn đùs hău ốy | Vấn đùs hẫu ới | 21.4% | |
| TC17 | An kâng thành vauglóng | An kâng thành vui lòng | 18.2% | autocorrect |
| TC18 | Nhấn | Nhân | 25.0% | |
| TC19 | Phụ́c diĉh ctnóng lổn | Phúc dịch trông lớn | 38.1% | |
| TC20 | Tâi Lộc | Tài Lộc | 14.3% | |
| TC21 | Tấm Bhất Biềń | Tâm Bất Biến | 30.8% | autocorrect |
| TC22 | Tâm | Tâm | 0.0% | EXACT |
| TC24 | Thanh Tâm | Thanh Tâm | 0.0% | EXACT |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhẫn Nhũn Nín | 35.3% | |

**Notable:** Gemini 2.5 heavily autocorrects TC14, TC15, TC17, TC21. Loses TC4's unusual `ctrông` → `trông` (partial autocorrect). Overall CER=18.96%.

#### Per-TC Detail — Gemini 3 Flash

| TC | Human-Verified GT | Gemini 3 Flash Output | CER | Match |
|----|-------------------|-----------------------|-----|-------|
| TC1 | Nhấn | Nhẫn | 25.0% | |
| TC2 | An Nhiên | An Nhiên | 0.0% | EXACT |
| TC3 | Vấn Sú An | Vẫn Sử An | 22.2% | |
| TC4 | Phúc dúch ctrông tón | Phúc đức trường tồn | 35.0% | autocorrect |
| TC5 | Tâm | Tâm | 0.0% | EXACT |
| TC6 | Phụ́c | Phúc | 40.0% | autocorrect |
| TC8 | Trâi | Trãi | 25.0% | |
| TC9 | Bián An | Bình An | 42.9% | autocorrect |
| TC10 | Hấnh Phụ̃c | Hạnh Phúc | 30.0% | autocorrect |
| TC11 | Tri Ân | Tri Ân | 0.0% | EXACT |
| TC12 | An Khang | An Khang | 0.0% | EXACT |
| TC13 | Tẫm Bấṭ Biân | Tâm Bất Biến | 25.0% | autocorrect |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mãn Đường | 40.0% | autocorrect |
| TC15 | Thuấn Tú Nhiễn | Thuận Tự Nhiên | 21.4% | autocorrect |
| TC16 | Vấn đùs hău ốy | Vẫn đùa hău ốy | 14.3% | |
| TC17 | An kâng thành vauglóng | An kâng thành vauglóng | 0.0% | EXACT |
| TC18 | Nhấn | Nhẫn | 25.0% | |
| TC19 | Phụ́c diĉh ctnóng lổn | Phúc diêh ctnăng lổn | 19.1% | |
| TC20 | Tâi Lộc | Tài Lộc | 14.3% | |
| TC21 | Tấm Bhất Biềń | Tâm Bất Biến | 30.8% | autocorrect |
| TC22 | Tâm | Tâm | 0.0% | EXACT |
| TC24 | Thanh Tâm | Thanh Tâm | 0.0% | EXACT |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhẫn Nhửằn Niín | 29.4% | |

**Notable:** Gemini 3 Flash shows the **strongest autocorrect** of all engines — TC4 fully normalized to the original prompt `đức trường tồn`, TC9 `Bián`→`Bình`, TC13/TC14/TC15/TC21 all autocorrected to valid Vietnamese. CER=19.10% overall.

### 13.6 Phase 3 — Dataset Samples (Metadata GT)

| Engine | N | Exact Match | Mean CER | Mean WER | Tier |
|--------|---|-------------|----------|----------|------|
| **Gemini 3.1 Flash Lite** | 30 | **21/30 (70.0%)** | **3.23%** | 3.23% | Tier 1 |
| **Gemini 3 Flash** | 30 | **22/30 (73.3%)** | **3.32%** | 3.32% | Tier 1 |
| **Gemini 2.5 Flash** | 30 | 16/30 (53.3%) | 4.06% | 4.06% | Tier 1 |
| **Qwen3-VL-8B** (FPT) | 30 | **15/30 (50.0%)** | **13.75%** | 13.75% | Tier 1 |
| **DeepSeek-OCR** (FPT) | 30 | 0/30 (0%) | 62.29% | 62.29% | Tier 3 |
| **PARSeq** | 30 | 0/30 (0%) | 79.08% | 79.08% | Tier 3 |
| **SVTRv2** | 30 | 0/30 (0%) | 83.35% | 83.35% | Tier 3 |

> **Note (updated 2026-03-29):** All Gemini variants now tested on full 30 samples. Gemini 3.1 Flash Lite benefits from 500 RPD / 15 RPM quota; Gemini 2.5 Flash has 20 RPD / 5 RPM (Google Cloud fallback key used when free keys exhausted).

### 13.7 Vintern-3B Comprehensive Results (Added 2026-03-28)

Vintern-3B was missing from the Section 13 comprehensive evaluation. Full 3-phase results:

**Phase 2 — Generated Images (TC1–TC25):**

| TC | Ground Truth | Vintern-3B OCR | CER |
|----|-------------|----------------|-----|
| TC1 | Nhấn | Nhấn | **EXACT** |
| TC2 | An Nhiên | An Nhiên | **EXACT** |
| TC3 | Vấn Sú An | Vân Sư An | 22.2% |
| TC4 | Phúc dúch ctrông tón | Phúc đức trông tôn | 25.0% |
| TC5 | Tâm | Tâm | **EXACT** |
| TC6 | Phụ́c | Phúc | 40.0% |
| TC8 | Trâi | Trái | 25.0% |
| TC9 | Bián An | Biển An | 14.3% |
| TC10 | Hấnh Phụ̃c | Hạnh Phúc | 30.0% |
| TC11 | Tri Ân | Tri Ân | **EXACT** |
| TC12 | An Khang | An Khang | **EXACT** |
| TC13 | Tẫm Bấṭ Biân | Tấm Bất Biển | 25.0% |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mãn Doùng | 26.7% |
| TC15 | Thuấn Tú Nhiễn | Thuấn Tú Nhiên | 7.1% |
| TC16 | Vấn đùs hău ốy | Vẫn bùi háu ốy | 28.6% |
| TC17 | An kâng thành vauglóng | An kâng thành vulgóng | 13.6% |
| TC18 | Nhấn | Nhấn | **EXACT** |
| TC19 | Phụ́c diĉh ctnóng lổn | Phúc dịch ctrông lổn | 28.6% |
| TC20 | Tâi Lộc | Tái Lộc | 14.3% |
| TC21 | Tấm Bhất Biềń | Tâm Phật Biển | 38.5% |
| TC22 | Tâm | Tâm | **EXACT** |
| TC24 | Thanh Tâm | Thanh Tâm | **EXACT** |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhấn Nhuận Nín | 35.3% |

**Summary:** Exact=8/23 (34.8%), Mean CER=16.27%, Mean WER=16.27%

**Phase 3 — Dataset Samples (30 stratified):**

Exact=24/30 (80.0%), Mean CER=1.53%. Vintern-3B excels on standard calligraphy fonts from the dataset, struggling mainly on unusual diacritical variants (Mắn→Mẫn, gặt→gặp, Tồn→Tôn).

### 13.8 Chandra OCR Results (Added 2026-03-28)

**Chandra OCR 2** (`datalab-to/chandra-ocr-2`) is a new 9B parameter model supporting 40+ languages. Installed via `pip install chandra-ocr[hf]`.

**Phase 1 — Smoke Test:** "Danh" → "Danh" (**EXACT**, CER=0%)

**Phase 2 — Generated Images (TC1–TC25):**

| TC | Human-Verified GT | Chandra OCR Output | CER | Match |
|----|-------------------|-------------------|-----|-------|
| TC1 | Nhấn | Nhấn | 0.0% | EXACT |
| TC2 | An Nhiên | An Nhiên | 0.0% | EXACT |
| TC3 | Vấn Sú An | Văn Sử An | 22.2% | autocorrect |
| TC4 | Phúc dúch ctrông tón | Phúc dúch trông tôn | 10.0% | |
| TC5 | Tâm | Tâm | 0.0% | EXACT |
| TC6 | Phụ́c | Phúc | 40.0% | autocorrect |
| TC8 | Trâi | Trái | 25.0% | |
| TC9 | Bián An | Biển An | 14.3% | |
| TC10 | Hấnh Phụ̃c | Hành Phục | 20.0% | autocorrect |
| TC11 | Tri Ân | Trí Ân | 16.7% | |
| TC12 | An Khang | An Khang | 0.0% | EXACT |
| TC13 | Tẫm Bấṭ Biân | Tâm Bất Biên | 25.0% | |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mắn Doảng | 26.7% | |
| TC15 | Thuấn Tú Nhiễn | Thuần Tú Nhiên | 14.3% | |
| TC16 | Vấn đùs hău ốy | Văn đủ hầu ấy | 35.7% | autocorrect |
| TC17 | An kâng thành vauglóng | An khang thành vui lòng | 27.3% | autocorrect |
| TC18 | Nhấn | Nhấn | 0.0% | EXACT |
| TC19 | Phụ́c diĉh ctnóng lổn | Phức dịch ctrông lớn | 33.3% | |
| TC20 | Tâi Lộc | Tài Lộc | 14.3% | |
| TC21 | Tấm Bhất Biềń | Tâm Bất Biến | 30.8% | autocorrect |
| TC22 | Tâm | Tâm | 0.0% | EXACT |
| TC24 | Thanh Tâm | Thanh Tâm | 0.0% | EXACT |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhấn Nhũn Nún | 41.2% | |

Summary: Exact=7/23 (30.4%), Mean CER=17.25%. Chandra shows strong autocorrect behavior (TC3, TC6, TC10, TC16, TC17, TC21) — normalizing unusual characters to valid Vietnamese words more aggressively than Vintern-3B.

**Phase 3 — Dataset Samples (30 stratified):**

Exact=7/30 (23.3%), Mean CER=24.14%. Significantly worse than Vintern-3B on dataset images. Chandra struggles with short Vietnamese words (Trí→Te, Tài→下, Đạo→Jo, Mai Mỹ→Wae Wiy) and multi-word phrases (Mực Bút Trường Đông Dũng Tín→Thư Viện Đông Dương).

**Key observations:**
- Returns HTML-formatted output (`<div data-bbox="..."><p>text</p></div>`) — requires stripping HTML tags
- 9B model size (~18 GB VRAM) — heavier than Vintern-3B (7.4 GB)
- Slow inference (~15s per image vs ~7s for Vintern-3B)
- Reasonable on generated calligraphy (CER 17.25%) but poor on standard dataset (CER 24.14%)
- Verdict: **Not recommended** — heavier, slower, and less accurate than Vintern-3B

### 13.9 Cross-Engine Phase 2 Comparison Matrix (Human-Verified GT)

The following table compares all Tier 1 engine outputs side-by-side against Human-Verified ground truth for each TC. This reveals per-TC autocorrect patterns across engines.

**Prompt context:** The three Gemini columns in the **compact** table use each model’s **Default (English)** short extract from the Phase 2 comprehensive harness (same instruction family as the first column of §13.9.1–§13.9.3). **Recommended prompts** for higher aggregate scores: §14.7 (Gemini 3.1 Flash Lite, seven prompts) and §14.8 (five prompts for 3 Preview / 2.5 / 2.5 Lite). **Per-prompt OCR strings** for Gemini 3 Preview, 2.5 Flash, and 2.5 Flash Lite are in §13.9.1–§13.9.3.

| TC | Human-Verified GT | Vintern-3B | Qwen3-VL-8B | Chandra-OCR-2 | Gemini 3.1 Flash Lite — Default (English) | Gemini 2.5 Flash — Default (English) | Gemini 3 Flash — Default (English) |
|----|-------------------|------------|-------------|---------------|-------------|------------|----------------|
| TC1 | Nhấn | **Nhấn** ✓ | Nhẫn | **Nhấn** ✓ | Nhẫn | Nhẫn | Nhẫn |
| TC2 | An Nhiên | **An Nhiên** ✓ | **An Nhiên** ✓ | **An Nhiên** ✓ | **An Nhiên** ✓ | **An Nhiên** ✓ | **An Nhiên** ✓ |
| TC3 | Vấn Sú An | Vân Sư An | Vân Sứ An | Văn Sử An | Vấn Sự An | Vân Sử An | Vẫn Sử An |
| TC4 | Phúc dúch ctrông tón | Phúc đức trông tôn | **Phúc dúch ctrông tón** ✓ | Phúc dúch trông tôn | Phúc dúch trông tón | Phúc dúch trông tốn | Phúc đức trường tồn |
| TC5 | Tâm | **Tâm** ✓ | **Tâm** ✓ | **Tâm** ✓ | Tẫm | **Tâm** ✓ | **Tâm** ✓ |
| TC6 | Phụ́c | Phúc | Phúc | Phúc | Phúc | Phúc | Phúc |
| TC8 | Trâi | Trái | Trải | Trái | Trải | Trái | Trãi |
| TC9 | Bián An | Biển An | Biễn An | Biển An | Biăn An | Biển An | Bình An |
| TC10 | Hấnh Phụ̃c | Hạnh Phúc | Hạnh Phúc | Hành Phục | Hạnh Phúc | Hạnh Phúc | Hạnh Phúc |
| TC11 | Tri Ân | **Tri Ân** ✓ | **Tri Ân** ✓ | Trí Ân | **Tri Ân** ✓ | **Tri Ân** ✓ | **Tri Ân** ✓ |
| TC12 | An Khang | **An Khang** ✓ | **An Khang** ✓ | **An Khang** ✓ | **An Khang** ✓ | **An Khang** ✓ | **An Khang** ✓ |
| TC13 | Tẫm Bấṭ Biân | Tấm Bất Biển | Tấm Bật Biên | Tâm Bất Biên | Tâm Bất Biến | Tâm Bất Biên | Tâm Bất Biến |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mãn Doùng | Phúc Mẫn Dưỡng | Phúc Mắn Doảng | Phúc Mãn Đường | Phúc Mãn Đường | Phúc Mãn Đường |
| TC15 | Thuấn Tú Nhiễn | Thuấn Tú Nhiên | Thuần Tử Nhiên | Thuần Tú Nhiên | Thuần Tử Nhiên | Thuận Tự Nhiên | Thuận Tự Nhiên |
| TC16 | Vấn đùs hău ốy | Vẫn bùi háu ốy | Vẫn bùn hãu ốy | Văn đủ hầu ấy | Vẫn đùs hău ỐY | Vấn đùs hẫu ới | Vẫn đùa hău ốy |
| TC17 | An kâng thành vauglóng | An kâng thành vulgóng | **An kâng thành vauglóng** ✓ | An khang thành vui lòng | **An kâng thành vauglóng** ✓ | An kâng thành vui lòng | **An kâng thành vauglóng** ✓ |
| TC18 | Nhấn | **Nhấn** ✓ | Nhẫn | **Nhấn** ✓ | Nhẫn | Nhân | Nhẫn |
| TC19 | Phụ́c diĉh ctnóng lổn | Phúc dịch ctrông lổn | Phục dịch ctrông lỗn | Phức dịch ctrông lớn | Phúc dịch trống lổn | Phúc dịch trông lớn | Phúc diêh ctnăng lổn |
| TC20 | Tâi Lộc | Tái Lộc | Tài Lộc | Tài Lộc | Tài Lộc | Tài Lộc | Tài Lộc |
| TC21 | Tấm Bhất Biềń | Tâm Phật Biển | Tầm Bhất Biền | Tâm Bất Biến | Tâm Bhất Biến | Tâm Bất Biến | Tâm Bất Biến |
| TC22 | Tâm | **Tâm** ✓ | **Tâm** ✓ | **Tâm** ✓ | Tấm | **Tâm** ✓ | **Tâm** ✓ |
| TC24 | Thanh Tâm | **Thanh Tâm** ✓ | **Thanh Tâm** ✓ | **Thanh Tâm** ✓ | **Thanh Tâm** ✓ | **Thanh Tâm** ✓ | **Thanh Tâm** ✓ |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhấn Nhuận Nín | Nhẫn Nhũn Niún | Nhấn Nhũn Nún | Nhẫn Nhuần Nín | Nhẫn Nhũn Nín | Nhẫn Nhửằn Niín |
| | **Exact** | **8/23** | **8/23** | **7/23** | **5/23** | **6/23** | **7/23** |
| | **Mean CER** | **16.3%** | **17.0%** | **17.3%** | **19.8%** | **19.0%** | **19.1%** |

#### Gemini — Phase 2 matrix by model × prompt (full columns)

Five shared prompts: **Default (English)**, **Vietnamese Vô Tri**, **Geometric Rules**, **Expert Typography Analyst**, **Precise Scanner** (§14.8). Each subsection is **generated images only** (23 TC, human-verified GT). Row **Σ Phase 2** = mean CER and exact count on those 23 TC.

### 13.9.1 Gemini 3 Flash Preview — OCR output by prompt (Phase 2, human-verified GT)

*Source log: `experiments/results/ocr_accuracy_new_prompts_v3/log.txt`.*

| TC | Human-Verified GT | Gemini 3 Flash Preview — Default (English) | Gemini 3 Flash Preview — Vietnamese Vô Tri | Gemini 3 Flash Preview — Geometric Rules | Gemini 3 Flash Preview — Expert Typography Analyst | Gemini 3 Flash Preview — Precise Scanner |
|---|---|---|---|---|---|---|
| TC1 | Nhấn | Nhẫn | Nhấn | Nhấn | Nhấn | Nhấn |
| TC2 | An Nhiên | An Nhiên | An Nhíên | Ãn Nhỉên | An Nhỉên | An Nhíên |
| TC3 | Vấn Sú An | Vẫn Sử An | Vẫn Sú An | Vẫn Sú An | Vẫn Sủ An | Vẫn Sú An |
| TC4 | Phúc dúch ctrông tón | Phúc đức trường tồn | Phúc dúch ctrông tón | Phúc dúch ctrông tón | Phúc dúch ctrông tốn | Phúc dúch đrông tón |
| TC5 | Tâm | Tâm | Tẫ́m | Tẫ́m | Tẫm | Tẫ́m |
| TC6 | Phụ́c | Phúc | Phṹc | Phụ́c | Phṵ́c | Phúc |
| TC8 | Trâi | Trãi | Trâí | Trấi | Trâí | Trấí |
| TC9 | Bián An | Biǎn An | Biăn An | Biăn An | Biãn An | Biản An |
| TC10 | Hấnh Phụ̃c | Hãnh Phũc | Hẫnh Phụ̃c | Hấṇh Phụ̃c | Hẫnh Phụ̃c | Hấnh Phụ̃c |
| TC11 | Tri Ân | Tri Ân | Tri Ân | Tri Ân | Tri Ân | Tri Ân |
| TC12 | An Khang | An Khang | Án Kháng | An Khang | An Khang | Ãn Khang |
| TC13 | Tẫm Bấṭ Biân | Tâm Bất Biến | Tấm Bấṭ Biân | Tấm Bấṭ Biân | Tấm Bẩṭ Biân | Tấm Bất Biân |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mẫn Doŭng | Phụ́c Mấn Doŭng | Phụ́c Mấn Doŭng | Phụ́c Mấn Doŭng | Phụ́c Mấn Doŭng |
| TC15 | Thuấn Tú Nhiễn | Thuận Tự Nhiên | Thuấn Tú Nhiễn | Thuấn Tứ Nhiễn | Thuấn Tứ Nhiễn | Thuấn Tử Nhiễn |
| TC16 | Vấn đùs hău ốy | Vẫn đùs hău ốy | Vẩn đùs hău ốy | Vẩn đùs hău ốy | Vẩn đùs hău ốy | Vấn đùs hắu ốy |
| TC17 | An kâng thành vauglóng | An kâng thành vauglóng | An kâng thânh vuuglóng | An kâng thành vauglóng | An kâng thành vauglóng | An kâng thầnh vuuglóng |
| TC18 | Nhấn | Nhẫn | Nhấn | Nhấn | Nhấn | Nhấn |
| TC19 | Phụ́c diĉh ctnóng lổn | Phục diêch ctnŏng lổn | Phụ́c diếch ctnŏ́ng lổn | Phục diêh ctñŏng lổn | Phục diêh ctnăng lổn | Phụ́c dîch ctn̆ŏng lổn |
| TC20 | Tâi Lộc | Tài Lộc | Tẫi Lộc | Tẫi Lộc | Tẫi Lộc | Tấi Lộc |
| TC21 | Tấm Bhất Biềń | Tâm Bất Biến | Tẩm Bhẫt Biến | Tấm Bhẫt Biến | Tẩm Bhẫt Biến | Tấm Bhẫt Bịến |
| TC22 | Tâm | Tâm | Tẫm | Tấm | Tẫm | Tẫm |
| TC24 | Thanh Tâm | Thanh Tâm | Thanh Tâm | Thanh Tâm | Thanh Tâm | Thanh Tấm |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhẫn Nhuần Niún | Nhấn Nhŭan Nhủần Niứn | Nhấn Nhửẳn Niứn | Nhấn Nhŭản Nìín | Nhấn Nhŭản Niún |
| **Σ Phase 2** | Mean CER / exact (23 TC) | **16.93%** · 7/23 | **16.76%** · 7/23 | **13.03%** · 9/23 | **12.84%** · 7/23 | **17.18%** · 5/23 |

### 13.9.2 Gemini 2.5 Flash — OCR output by prompt (Phase 2, human-verified GT)

*Same log as §13.9.1.*

| TC | Human-Verified GT | Gemini 2.5 Flash — Default (English) | Gemini 2.5 Flash — Vietnamese Vô Tri | Gemini 2.5 Flash — Geometric Rules | Gemini 2.5 Flash — Expert Typography Analyst | Gemini 2.5 Flash — Precise Scanner |
|---|---|---|---|---|---|---|
| TC1 | Nhấn | Nhân | Nhẫn | Nhấn | Nhấn | Nhẫn |
| TC2 | An Nhiên | An Nhiên | An Nhiên | An Nhiên | An Nhiên | An Nhiên |
| TC3 | Vấn Sú An | Vân Sứ An | Vâń Súĩ An | Vẫn Sú An | Vân Sĩ An | Vân Sử An |
| TC4 | Phúc dúch ctrông tón | Phúc dúch trông tón | Phúc dúch trông tón | Phức dích trông tón | Phúc dúch trông tón | Phức dúch trông tón |
| TC5 | Tâm | Tẫm | Tẫm | Tâm | Tâm | Tẫm |
| TC6 | Phụ́c | Phúc | Phṹc | Phụ̃́c | Phúc̣̃ | Phụ́c |
| TC8 | Trâi | Trãi | Trãi | Trái | Trấi | Traí |
| TC9 | Bián An | Biển An | Biãn An | Biãn An | Biển An | Biển An |
| TC10 | Hấnh Phụ̃c | Hẵnh Phụcc | Hãnh Phụ̉c | Hãnh Phụ̃c | Hãnh Phũc | Hẫnh Phự̃c |
| TC11 | Tri Ân | Tri Ân | Trí Ân | Trí Ân | Trí Ân | Trí Ân |
| TC12 | An Khang | An Khang | An Khang | An Khang | An Khang | An Khang |
| TC13 | Tẫm Bấṭ Biân | Tâm Bất Biến | Tâm Bất Biân | Tấm Bât Biến | Tấm Bậṭ Biấn | Tâ ḿ Bâ t́, Biân |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mãn Đường | Phúc Mãn Doũng | Phụ̃c Mãn Dòũng̣ | Phụ́c Mẫn Dõũng | Phuc Mãn Dỗững |
| TC15 | Thuấn Tú Nhiễn | Thuận Tự Nhiên | Thuấn Tú̃ Nhiễ̃n | Thuâñ Tữ Nhiên | Thuần Tữ Nhiễn | Thuấn Tư Nhiên |
| TC16 | Vấn đùs hău ốy | Vân đùs hầu õy | Vấn dùs hã́u ỗy | Vẫn dùs hãu õý | Vân đùs hãu ớy | Vấn dus hẫ̀u ốy |
| TC17 | An kâng thành vauglóng | An kàng thành vuióng | An kâng thành vauglóńg | An kâng thành vauglóng | An kâng thành vaglóng | An kâng thành vuaglóng |
| TC18 | Nhấn | Nhân | Nhấn | Nhấn | Nhẫn | Nhẫn |
| TC19 | Phụ́c diĉh ctnóng lổn | Phúc dịch chồng lớn | Phục dị̂ch ctrõng lổn | Phụ̣c dîch ctõnõng lôn | Phục dịch ctñông lôn | Phục dịch ctnõng lỗn |
| TC20 | Tâi Lộc | Tài Lộc | Tâï Lôc̰ | Tái Lộc | Tài Lộc | Tài Lộc |
| TC21 | Tấm Bhất Biềń | Tâm Bất Biến | Tâm̃ Bhất́́ Biếń́ | Tẫm Bất Biễn | Tấm Bất Biến | Tấmm Bhất Biễn |
| TC22 | Tâm | Tâm | Tẫm | Tẫm | Tẫm | Tẫm |
| TC24 | Thanh Tâm | Thanh Tâm | Thanh Tâm | Thanh Tâm | Thanh Tâm | Thanh Tâm |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhẫn Nhuần Nín | Nhẫn Nhũôân Nín | Nhẫn Nhậ̃n Nín | Nhẫn Nhuẫn Nín | Nhẫn Nhuân Nín |
| **Σ Phase 2** | Mean CER / exact (23 TC) | **19.75%** · 5/23 | **21.72%** · 4/23 | **16.75%** · 7/23 | **19.04%** · 5/23 | **21.66%** · 4/23 |

### 13.9.3 Gemini 2.5 Flash Lite — OCR output by prompt (Phase 2, human-verified GT)

*Model `gemini-2.5-flash-lite`. JSON: `experiments/results/ocr_gemini_25_flash_lite_current_dataset/ocr_prompt_compare_results.json`.*

| TC | Human-Verified GT | Gemini 2.5 Flash Lite — Default (English) | Gemini 2.5 Flash Lite — Vietnamese Vô Tri | Gemini 2.5 Flash Lite — Geometric Rules | Gemini 2.5 Flash Lite — Expert Typography Analyst | Gemini 2.5 Flash Lite — Precise Scanner |
|---|---|---|---|---|---|---|
| TC1 | Nhấn | Tết Nhàn | y Nhẫn | Nhân | e Nhãn | Nhân |
| TC2 | An Nhiên | An Nhiên | An Nhiên | An Nhiên | An Nhiên | An Nhiên |
| TC3 | Vấn Sú An | ân Sū An | ân Sũ An | Vân Sũ An | ân sū An | Vân Sü An |
| TC4 | Phúc dúch ctrông tón | ch h g Phúc đúc trôn tôn | Phúc đức trôn tôn | phúc đức tròn vẹn | c h ng Phú dức trôn ton | Phúc đúcrốn lốn |
| TC5 | Tâm | Tâm Tình | Tâmn | Tâm Thư | Tâm Tri | Tâm T |
| TC6 | Phụ́c | Chúc Phụ | Chúc Phuc | Lưu ý Phổ | Đuíc Phò | TƯC Ph |
| TC8 | Trâi | Trãi | Trãi | Trãi | Trãi | Trãi |
| TC9 | Bián An | Biǎn An | Biǎn An | Bián An | Biản An | Biển An |
| TC10 | Hấnh Phụ̃c | Hạnh Phúc | Hãn Phủ̃c | Hạnh Phúc | Hạnh Phúc | Hạnh Phúc |
| TC11 | Tri Ân | Trí Ân | Trì Ấn | Trì Ân | Trí Ân | Tri ân |
| TC12 | An Khang | An Khang | An Khang | Tin Khang | Tuy Khang | An Khang |
| TC13 | Tẫm Bấṭ Biân | Tâm Bất Biến | Tâm Bất Biến | Tâm Bất Biến | tâm Bất Biến | Tâm Bất Biến |
| TC14 | Phụ́c Mấn Doŭng | Phúc Mãn Đông yume | Chúc Mãn Dưỡng | Phúc Mẫn Đại ngã | Phúc Mãn Đuinga | Phúc Mãn Đaing |
| TC15 | Thuấn Tú Nhiễn | Tiên Như | uân uĩên | iên nữ iên | quan ũ iên Thư | uân ữ iên |
| TC16 | Vấn đùs hău ốy | vân bùs hàu ờy | Văn bùs hãu õy | Văn bùs hầu tỹ | Văn bùs hâu ởÿ | Văn bủs hâu ớỹ |
| TC17 | An kâng thành vauglóng | Tượng ánh long kà thà vua | hững qu kú thứ vũ | nh à nh l ong k á th u ng | này anh lòng ká thu vua | ing ảnh long ka thu vu |
| TC18 | Nhấn | iết Nhân | vơ Nhân | Nhiề | Nhan | an Nhà |
| TC19 | Phụ́c diĉh ctnóng lổn | c dich Phú ng lơn ctnó | íc dịch ng lôn Phụ ctno | íc dịch Phụ ng lớn ctno | íc dịch ng lớn Phụ ctnò | iich lơn Phi ctnò |
| TC20 | Tâi Lộc | Tài Lộc | Tài Lộc | Tài Lộc | Tai Lô | Tài Lộc |
| TC21 | Tấm Bhất Biềń | Tâm Bất Biến | Tâm Phát Biện | Tâm Phật Biên | Tâm Phát Biên | Tâm Bất Biến |
| TC22 | Tâm | Tâm Tình | Tâm tình | tâm | Tâm | Tâm |
| TC24 | Thanh Tâm | Thank You | Thanh Tú | Cảm ơn | Thank you | Thanh Tôm |
| TC25 | Nhầ́n Nhŭå̀n Niín | Nhân Nhüân | Nhân Nhüần | Nhân Nhüàn | Nhân Nhũn | Nhân Nhũn |
| **Σ Phase 2** | Mean CER / exact (23 TC) | **61.57%** · 2/23 | **49.88%** · 2/23 | **48.80%** · 2/23 | **51.49%** · 2/23 | **40.35%** · 3/23 |

**Key observations from the matrix:**

1. **TC6 (Phụ́c)** — Universal autocorrect: ALL 6 engines normalize `Phụ́c` (combining acute on ụ) to `Phúc`. No engine preserves the unusual Unicode composition.

2. **TC10 (Hấnh Phụ̃c)** — Universal autocorrect: ALL engines output some form of `Hạnh Phúc` or `Hành Phục`. None preserve the actual `Hấnh Phụ̃c`.

3. **TC4 (Phúc dúch ctrông tón)** — Only Qwen3-VL preserves this exact unusual spelling. **Gemini 3 Flash** *(default in matrix; best prompts: Expert Typography Analyst · Geometric Rules per §14.8)* shows the **strongest autocorrect** in this default run, fully normalizing to the original prompt text `đức trường tồn`.

4. **TC17 (An kâng thành vauglóng)** — Qwen3-VL, **Gemini 3.1 Flash Lite** *(best: Strict Literal · Expert Typography Analyst · Vietnamese Vô Tri)*, and **Gemini 3 Flash** *(best: Expert Typography Analyst · Geometric Rules)* preserve the unusual `vauglóng` in these cells, while Chandra and **Gemini 2.5 Flash** *(best: Default English · Geometric Rules gen-only)* autocorrect to `vui lòng`.

5. **TC14 (Phụ́c Mấn Doŭng)** — Every engine autocorrects differently: `Doùng`, `Dưỡng`, `Doảng`, `Đường`. None preserve the original `Doŭng`.

6. **Simple TCs (TC2, TC11, TC12, TC22, TC24)** — All 6 Tier 1 engines agree on exact matches for these clean, simple calligraphy images.

7. **Compact Gemini columns (§13.9 matrix)** — Use **Default (English)** only; see §13.9.1–§13.9.3 for every **model × prompt** string. For aggregate lift vs default, see §14.8 (e.g. **Gemini 3 Flash Preview** + **Expert Typography Analyst** ~8.9% combined CER vs ~12.8% Default on the same 23 TC + 15 DS benchmark).

### 13.10 Comprehensive Engine Ranking (Updated)

Metrics use **default** harness prompts (§13.9). For Gemini rows, **best OCR prompt names** are appended after the model (§14.7–§14.8).

**Prompt-aware ranking (same 38 items as §13.10.1):** The table below is **default-prompt only**. On the **§14.8** harness (**23** human-verified generated TC + **15** dataset crops), **Gemini 3 Flash Preview** with **Expert Typography Analyst** reaches **8.86% combined CER** and **19/38** exact — the **documented runner-up to Vintern-3B** for **blended** gen+DS accuracy among all configurations we report. Rank **#2 Qwen3-VL-8B** reflects **default** API prompts: it **ties** Vintern on generated exact (8/23) but stays **weaker on dataset CER** than that tuned-Gemini row on the overlapping DS slice (see §13.10.1 vs §13.10 DS column).

| Rank | Engine | Gen CER | Gen Exact | DS CER | DS Exact | VRAM | Cost | Verdict |
|------|--------|---------|-----------|--------|----------|------|------|---------|
| 1 | **Vintern-3B** | 16.3% | 34.8% | **1.5%** | **80%** | ~7 GB | Free | **Primary** — best overall accuracy |
| 2 | **Qwen3-VL-8B** (FPT Cloud) | **17.0%** | **34.8%** | 9.9% | 60% | 0 | Free | **Strong secondary (default prompt)** — no GPU; see note above for prompt-tuned G3 Preview |
| 3 | **Chandra-OCR-2** | 17.3% | 30.4% | 24.1% | 23% | ~18 GB | Free | Not recommended — heavy, slow, poor on dataset |
| 4 | **Gemini 3.1 Flash Lite** — *best prompts:* **Strict Literal (English)** (gen), **Expert Typography Analyst** / **Vietnamese Vô Tri** (DS) — §14.7 | 19.8% | 21.7% | **3.2%** | **70%** | 0 | Free | **Strong dataset accuracy** — 500 RPD |
| 5 | **Gemini 2.5 Flash** — *best prompts:* **Default (English)** (balanced), **Geometric Rules** (gen-only; poor on DS) — §14.8 | 19.0% | 26.1% | 4.1% | 53% | 0 | Free | Good all-round — 20 RPD |
| 6 | **Gemini 3 Flash** — *best prompts:* **Expert Typography Analyst** (combined/DS), **Geometric Rules** (lowest gen CER) — §14.8 | 19.1% | 30.4% | **3.3%** | **73%** | 0 | Free | **Strong dataset accuracy** — rate limited |
| 7 | **DeepSeek-OCR** (FPT Cloud) | 100% | 0% | 100% | 0% | 0 | Free | Not suitable — returns empty strings |
| 8 | **SVTRv2** | 54.3% | 0% | 83.4% | 0% | ~0.5 GB | Free | Not suitable — no Vietnamese diacritics |
| 9 | **PARSeq** | 60.5% | 0% | 79.1% | 0% | ~0.3 GB | Free | Not suitable — English only |
| 10 | **GOT-OCR 2.0** | — | — | — | — | ~1.5 GB | Free | Not suitable — severe repetition artifacts |
| 11 | **dots.ocr** | 100% | 0% | 100% | 0% | ~6 GB | Free | Not suitable — returns layout JSON, not text |
| 12 | **OCR.space** | 95.5% | 0% | 99.2% | 0% | 0 | Free | Not suitable — no Vietnamese support |
| 13 | **PaddleOCR-VL-1.5** | — | — | — | — | ~2 GB | Free | FAILED — transformers 5.x incompatible |

**Gemini 2.5 Flash Lite** — *best prompts:* **Precise Scanner** (gen/combined on §14.8 sweep), **Geometric Rules** (dataset exact on `data/current_dataset` subset). Not in the Phase 2 matrix; weak on AI-generated baseline TCs vs heavier Gemini models (§14.8).

### 13.10.1 Gemini — full metrics by model and prompt (Phase 2 + dataset)

Numeric summary for the **five shared prompts** in §14.8. **Gemini 3 Flash Preview** and **Gemini 2.5 Flash**: 23 generated TC (human-verified GT) + **15** dataset crops — **38** items total (`experiments/results/ocr_accuracy_new_prompts_v3/log.txt`). **Gemini 2.5 Flash Lite**: same 23 TC + **13** dataset images from `data/current_dataset` (two files skipped) — **36** items (`experiments/results/ocr_gemini_25_flash_lite_current_dataset/ocr_prompt_compare_results.json`). CER = mean character error rate; exact = string-identical to GT.

**Strongest row on the 38-item harness:** **Gemini 3 Flash Preview** + **Expert Typography Analyst** — **8.86%** combined CER, **19/38** exact — i.e. the **true second-place** configuration after **Vintern-3B** when both generated and dataset splits count (§13.10 note).

| Model | Prompt | Gen CER | Gen exact | DS CER | DS exact | Combined CER | Combined exact |
|-------|--------|---------|-----------|--------|--------|--------------|----------------|
| Gemini 3 Flash Preview | Default (English) | 16.93% | 7/23 | 6.39% | 8/15 | 12.77% | 15/38 |
| Gemini 3 Flash Preview | Vietnamese Vô Tri | 16.76% | 7/23 | 9.53% | 5/15 | 13.91% | 12/38 |
| Gemini 3 Flash Preview | Geometric Rules | 13.03% | 9/23 | 8.32% | 9/15 | 11.17% | 18/38 |
| Gemini 3 Flash Preview | Expert Typography Analyst | 12.84% | 7/23 | 2.76% | 12/15 | 8.86% | 19/38 |
| Gemini 3 Flash Preview | Precise Scanner | 17.18% | 5/23 | 6.65% | 10/15 | 13.02% | 15/38 |
| Gemini 2.5 Flash | Default (English) | 19.75% | 5/23 | 6.47% | 8/15 | 14.51% | 13/38 |
| Gemini 2.5 Flash | Vietnamese Vô Tri | 21.72% | 4/23 | 11.58% | 7/15 | 17.72% | 11/38 |
| Gemini 2.5 Flash | Geometric Rules | 16.75% | 7/23 | 15.42% | 2/15 | 16.23% | 9/38 |
| Gemini 2.5 Flash | Expert Typography Analyst | 19.04% | 5/23 | 8.81% | 6/15 | 15.00% | 11/38 |
| Gemini 2.5 Flash | Precise Scanner | 21.66% | 4/23 | 12.86% | 3/15 | 18.19% | 7/38 |
| Gemini 2.5 Flash Lite | Default (English) | 61.57% | 2/23 | 8.44% | 5/13 | 42.39% | 7/36 |
| Gemini 2.5 Flash Lite | Vietnamese Vô Tri | 49.88% | 2/23 | 8.12% | 5/13 | 34.80% | 7/36 |
| Gemini 2.5 Flash Lite | Geometric Rules | 48.80% | 2/23 | 7.03% | 7/13 | 33.72% | 9/36 |
| Gemini 2.5 Flash Lite | Expert Typography Analyst | 51.49% | 2/23 | 10.14% | 6/13 | 36.56% | 8/36 |
| Gemini 2.5 Flash Lite | Precise Scanner | 40.35% | 3/23 | 7.75% | 6/13 | 28.58% | 9/36 |

### 13.11 Key Findings (Updated)

1. **Vintern-3B confirms its position as the best overall engine.** Now tested in the full comprehensive framework: 34.8% exact / 16.3% CER on generated images and 80.0% exact / 1.5% CER on dataset — the best dataset accuracy among all engines tested.

2. **Chandra OCR 2 is not competitive.** Despite being a 9B model (vs Vintern-3B's 3B), it is worse on both generated (30.4% vs 34.8% exact) and dataset images (23.3% vs 80.0% exact). It requires ~18 GB VRAM and is 2x slower per inference.

3. **Qwen3-VL-8B-Instruct is the best *default-prompt* API engine** in the main ranking table: it **matches** Vintern-3B on generated-image exact (8/23) without a GPU. For **mixed** gen+DS work where the pipeline may **fix an OCR prompt**, **Gemini 3 Flash Preview + Expert Typography Analyst** is the **documented runner-up overall** (**8.86%** combined CER, **19/38** exact on the §14.8 38-item harness, §13.10.1) — ahead of default-prompt Qwen on that blended metric.

4. **Gemini 3 Flash Preview** — *best prompts:* **Expert Typography Analyst** (combined/DS), **Geometric Rules** (lowest gen CER in the five-prompt sweep). **Default** run in §13.9: ~19.1% gen CER / 7/23 exact; **full** Phase 3 table elsewhere uses 30 DS images (22/30, 3.3% CER). Strongest default-run autocorrect on TC4. **Expert Typography Analyst** (§14.8) yields **12.84%** gen CER, **2.76%** DS CER (12/15), **8.86%** combined, **19/38** exact — the **effective second-best** configuration after Vintern-3B for this benchmark. **Geometric Rules** reaches **13.03%** gen CER with **11.17%** combined.

5. **Gemini 2.5 Flash** — *best prompts:* **Default (English)** (mixed/balanced), **Geometric Rules** (gen-only). For production on mixed sources (§14.8), **Default (English)** ~6.5% DS CER, ~14.5% combined. **Geometric Rules** ~16.8% gen CER but ~15.4% DS CER — use only when workload is almost entirely distorted AI-generated calligraphy.

6. **Gemini 3.1 Flash Lite** — *best prompts:* **Strict Literal (English)** (generated CER), **Vietnamese Vô Tri** / **Expert Typography Analyst** (dataset — tie, §14.7). No single prompt wins both; pick by pipeline.

7. **Gemini 2.5 Flash Lite** — *best prompts:* **Precise Scanner** (gen/combined), **Geometric Rules** (dataset exact on `current_dataset` subset, §14.8). Unreliable on human-verified baseline generations; prefer heavier Gemini for Phase 2–style images.

8. **DeepSeek-OCR returns empty strings** with the default "Free OCR." prompt on calligraphy images. Not a viable engine.

9. **Local open-source models (SVTRv2, PARSeq, GOT-OCR, dots.ocr) remain unsuitable** for Vietnamese calligraphy.

### 13.12 Updated Multi-Engine Cross-Validation Strategy

```
Primary:    Vintern-3B
Secondary:  Qwen3-VL-8B (default API prompt)
            Gemini 3 Flash Preview + Expert Typography Analyst (prompt-tuned; §14.8 38-item harness)
Oracle:     Gemini 3.1 Flash Lite (Expert Typography Analyst / Vietnamese Vô Tri on dataset crops)
Fallback:   Gemini 2.5 Flash (Default (English); Geometric Rules if gen-only)
```

On the **23+15** §14.8 harness, prefer **Gemini 3 Flash Preview — Expert Typography Analyst** over default-prompt Qwen when disputing Vintern on **mixed** sources. Optional: **Geometric Rules** for gen-heavy batches (§14.8).

**Gemini 2.5 Flash Lite** — *Precise Scanner* / *Geometric Rules* only on **`data/current_dataset`**-style images (§14.8).

**Cross-validation rule:** When Vintern-3B and Qwen3-VL disagree, flag for Gemini Lite spot-check. This three-engine consensus catches autocorrect behavior without requiring human-GT verification for every image.

### 13.13 Installation Requirements

The comprehensive test requires the following packages (in addition to base environment):

```bash
pip install openai python-dotenv rapidfuzz google-genai
pip install transformers accelerate timm pytorch_lightning nltk
pip install paddlepaddle paddleocr==2.10.0 qwen-vl-utils
pip install "chandra-ocr[hf]"
apt-get install -y libgl1 libxcb1
```

### 13.14 Scripts & Reproducibility

```bash
# Comprehensive evaluation (all engines, all phases)
cd experiments/scripts
python test_all_engines_comprehensive.py --phase all

# Test specific engines (including new Vintern-3B and Chandra)
python test_all_engines_comprehensive.py --engines vintern_3b,chandra --phase all

# Prompt comparison mode (default vs strict anti-autocorrect prompt)
python test_all_engines_comprehensive.py --engines vintern_3b --phase all --prompt_compare

# All prompt-capable engines with comparison
python test_all_engines_comprehensive.py --phase all --prompt_compare

# Gemini variants only
python test_all_engines_comprehensive.py --engines gemini,gemini_lite,gemini_2_5 --phase all --prompt_compare
```

**Output files:**
- `experiments/results/ocr_accuracy/all_engines_comprehensive.json`
- `experiments/results/ocr_accuracy/vintern_3b_comprehensive.json`
- `experiments/results/ocr_accuracy/chandra_comprehensive.json`
- `experiments/results/ocr_accuracy/gemini3flash_prompt_compare.json`
- `experiments/results/ocr_accuracy/gemini_prompt_compare.json`
- `experiments/results/ocr_accuracy/fpt_cloud_prompt_compare.json`
- `experiments/results/ocr_accuracy/got_prompt_compare.json`
- `experiments/results/ocr_accuracy/dots_prompt_compare.json`
- `experiments/results/ocr_accuracy/remaining_engines.json`
- `experiments/results/ocr_accuracy/gemini3flash_full.json`
- `experiments/results/ocr_accuracy/gemini_phase3_complete.json`

---

## 14. Prompt Comparison Study: Default vs Strict Anti-Autocorrect (2026-03-28)

### 14.1 Motivation

Vietnamese calligraphy OCR engines tend to "autocorrect" unusual glyph shapes into known Vietnamese words, reducing literal accuracy. This section tests whether a strict anti-autocorrect prompt improves OCR accuracy across all engines.

Two prompts were compared:
- **Default (Vietnamese):** "Trích xuất chính xác các chữ thư pháp trong ảnh. Chỉ trả về văn bản, không mô tả."
- **Strict Literal (English):** A detailed prompt instructing the model to never autocorrect, never hallucinate underdots, use Unicode combining characters for conflicting marks, and output only the extracted text.

### 14.2 Engines Tested

Only engines that support custom prompts via `recognize_with_prompt()` were tested:

| Engine | Supports Custom Prompt | Mechanism |
|--------|----------------------|-----------|
| Vintern-3B | Yes | `model.chat()` with custom question |
| Gemini 3 Flash | Yes | API `contents=[prompt, image]` |
| Gemini 3.1 Flash Lite | Yes | API `contents=[prompt, image]` |
| Gemini 2.5 Flash | Yes | API `contents=[prompt, image]` |
| Qwen3-VL-8B | Yes | OpenAI-compatible API with custom text |
| DeepSeek-OCR | Yes | OpenAI-compatible API with custom text |
| GOT-OCR 2.0 | Yes (via processor) | Processor `prompt` parameter |
| dots.ocr | Yes | Qwen-VL chat template |
| Chandra OCR | No | Uses `prompt_type` enum only |
| PARSeq | No | Pure vision model |
| SVTRv2 | No | PaddleOCR pipeline |
| OCR.space | No | API with language param only |

### 14.3 Results — Phase 2 (Generated Images, TC1–TC25)

| Engine | Default CER | Strict CER | Delta | Default Exact | Strict Exact |
|--------|-------------|------------|-------|---------------|--------------|
| **Vintern-3B** | 16.27% | **14.98%** | **-1.29%** | 8/23 | 8/23 |
| **Gemini 3.1 Flash Lite** | 19.82% | **18.94%** | **-0.88%** | 5/23 | 5/23 |
| **Gemini 2.5 Flash** | 18.96% | **18.82%** | **-0.14%** | 6/23 | 5/23 |
| **Gemini 3 Flash** | 19.10% | **15.63%** | **-3.47%** | 7/23 | 6/23 |
| **Qwen3-VL-8B** | **17.03%** | 17.92% | +0.89% | 8/23 | 7/23 |
| **DeepSeek-OCR** | 100% | 100%+ | N/A | 0/23 | 0/23 |
| **GOT-OCR 2.0** | 100%+ | 5964%+ | N/A | 0/23 | 0/23 |
| **dots.ocr** | 100% | 100% | 0% | 0/23 | 0/23 |

### 14.4 Results — Phase 3 (Dataset Samples)

| Engine | Default CER | Strict CER | Delta | Default Exact | Strict Exact |
|--------|-------------|------------|-------|---------------|--------------|
| **Vintern-3B** | **1.53%** | 2.08% | +0.55% | **24/30** | 23/30 |
| **Gemini 3.1 Flash Lite** | **3.23%** | 3.64% | +0.41% | 21/30 | 22/30 |
| **Gemini 3 Flash** | **3.32%** | 6.73% | +3.41% | **22/30** | 15/30 |
| **Gemini 2.5 Flash** | **4.06%** | 7.96% | +3.90% | 16/30 | 15/30 |
| **Qwen3-VL-8B** | **9.92%** | 11.09% | +1.17% | 18/30 | 15/30 |

### 14.5 Analysis

1. **Strict prompt provides CER improvement on generated images** for Gemini 3 Flash (-3.47%), Vintern-3B (-1.29%), and Gemini Lite (-0.88%), but same or fewer exact matches. The improvement comes from the model producing slightly more literal diacritical marks. Notably, Gemini 3 Flash strict achieved EXACT match on TC14 (`Phụ́c Mấn Doŭng`) — the only engine/prompt combination to do so.

2. **Strict prompt significantly hurts dataset accuracy for Gemini 3 Flash and 2.5 Flash** — Gemini 3 Flash (+3.41% CER, 22→15 exact), Gemini 2.5 Flash (+3.90% CER, 16→15 exact). On standard calligraphy where the default prompt already works well, the strict English prompt introduces noise, adding spurious combining characters and misreading clean diacritics.

3. **Gemini 3.1 Flash Lite is the exception** — strict prompt barely affects it: +0.41% CER but actually gains 1 exact match (21→22). This is the only engine where strict prompt doesn't clearly hurt dataset accuracy.

4. **GOT-OCR 2.0 catastrophically fails with strict prompt** — the model interprets the long English prompt as a text generation seed and produces essays, random character sequences, and URLs instead of OCR output. CER exceeds 5000%.

5. **DeepSeek-OCR and dots.ocr are unaffected** — both engines produce empty/garbled output regardless of prompt.

6. **Gemini 3 Flash shows the largest strict-vs-default trade-off**: -3.47% CER improvement on generated images but +3.41% CER degradation on dataset, with 7 fewer exact matches on dataset (22→15). Gemini 2.5 Flash follows a similar pattern (-0.14% gen, +3.90% dataset).

### 14.6 Conclusion

**The strict anti-autocorrect prompt provides no practical benefit.** While it offers minor CER improvement on generated calligraphy images (~1% for Vintern-3B), it hurts dataset accuracy and provides no improvement in exact match rate. The default Vietnamese prompt should remain the standard for all production use.

The strict prompt is only useful for **diagnostic purposes** — examining exactly what the model "sees" in unusual glyph shapes — not for improving accuracy.

---

### 14.7 Expanded Prompt Comparison for Gemini 3.1 Flash Lite (2026-03-30)

Following the initial 2-prompt comparison, a more comprehensive 7-prompt evaluation was conducted on Gemini 3.1 Flash Lite to identify the optimal prompt for Vietnamese calligraphy OCR.

#### Prompts Tested

| ID | Prompt Type | Description |
|----|-------------|-------------|
| 1 | Default (Vietnamese) | "Trích xuất chính xác các chữ thư pháp trong ảnh. Chỉ trả về văn bản, không mô tả." |
| 2 | Default (English) | "Extract exactly the calligraphy text in the image. Only return the text, no description." |
| 3 | Strict Literal (English) | Detailed rules: NO AUTOCORRECT, Unicode combining, no hallucinated underdots |
| 4 | Vietnamese Vô Tri | Vietnamese: "không tự động sửa lỗi", strict diacritic rules |
| 5 | Geometric Rules | English: Axis analysis for Sắc (/) vs Ngã (~), underdot detection |
| 6 | Expert Typography Analyst | Morphological analysis: Sắc = upward thrust, Ngã = S-curve |
| 7 | Precise Scanner | Step-by-step: scan char-by-char, identify base letter, check top/bottom diacritics |

#### Results — Generated Images (23 TC)

| Prompt | Gen CER | Gen Exact |
|--------|---------|-----------|
| Default (Vietnamese) | 19.7% | 22% |
| Default (English) | 19.2% | 26% |
| **Strict Literal (English)** | **16.2%** | **26%** |
| Vietnamese Vô Tri | 23.5% | 26% |
| Geometric Rules | 20.3% | 22% |
| Expert Typography Analyst | 19.7% | 26% |
| Precise Scanner | 19.2% | 22% |

#### Results — Dataset Images (15 samples)

| Prompt | DS CER | DS Exact |
|--------|--------|----------|
| Default (Vietnamese) | 5.5% | 67% |
| Default (English) | 4.9% | 67% |
| Strict Literal (English) | 4.2% | 73% |
| **Vietnamese Vô Tri** | **3.4%** | **80%** |
| Geometric Rules | 4.6% | 67% |
| **Expert Typography Analyst** | **3.4%** | **80%** |
| Precise Scanner | 3.7% | 73% |

#### Key Findings

1. **Best for Generated Images (AI calligraphy):** `Strict Literal (English)` — 16.2% CER, 26% exact match. This prompt provides the largest CER improvement (-3.5% vs default) on AI-generated images.

2. **Best for Dataset Images (real calligraphy):** `Vietnamese Vô Tri` or `Expert Typography Analyst` — 3.4% CER, 80% exact match. Both achieve highest accuracy on clean calligraphy.

3. **Default English vs Default Vietnamese:** English version (19.2%) slightly outperforms Vietnamese (19.7%) on generated images, suggesting the model responds better to English instructions even for Vietnamese text.

4. **Trade-off Pattern:** Prompts that perform best on generated images (Strict Literal) are not optimal for dataset images, and vice versa. This confirms the earlier finding that no single prompt serves all use cases.

5. **Recommended for Production:**
   - Use `Strict Literal (English)` when evaluating AI-generated calligraphy (maximizes detection of rendering errors)
   - Use `Expert Typography Analyst` for clean dataset images (maximizes accuracy)

---

### 14.8 Five-Prompt OCR Sweep — Gemini 3 Preview, 2.5 Flash, 2.5 Flash Lite (2026-03-30)

This extends the prompt comparison to **three Gemini API models** using the **same five prompts** (no Default Vietnamese, no Strict Literal): **Default (English)**, **Vietnamese Vô Tri**, **Geometric Rules**, **Expert Typography Analyst**, **Precise Scanner**.

| Setting | Value |
|---------|--------|
| **Script** | `experiments/scripts/eval_ocr_prompt_compare.py` |
| **Generated images** | 23 TCs with **human-verified** ground truth (`GENERATED_IMAGE_GROUND_TRUTH`), images under `experiments/results/baseline_4w/` |
| **Gemini 3 Flash Preview & Gemini 2.5 Flash — dataset** | 15 stratified samples from `data/raw/dataset_8028_calligraphy_1font` (metadata as in §13.6) |
| **Gemini 2.5 Flash Lite — dataset** | 15 samples drawn from `data/current_dataset` using `metadata.jsonl` → `metadata.content`; two files are unreadable by PIL (`medium/7_words/syn_07126.jpg`, `medium/7_words/syn_06465.jpg`) and were **skipped**, so **13** dataset images count per prompt (**36** total items vs **38** for the other two models) |
| **Raw logs** | `experiments/results/ocr_accuracy_new_prompts_v3/log.txt` (3 Preview + 2.5 Flash; original run omitted 2.5 Flash Lite and hit a script error after the summary table), `experiments/results/ocr_gemini_25_flash_lite_current_dataset/log.txt` (2.5 Flash Lite) |

#### Results — Gemini 3 Flash Preview (`gemini-3-flash-preview`)

| Prompt | Gen CER | Gen exact | DS CER | DS exact | Combined CER | Combined exact |
|--------|---------|-----------|--------|----------|--------------|----------------|
| Default (English) | 16.9% | 7/23 | 6.4% | 8/15 | 12.8% | 15/38 (39.5%) |
| Vietnamese Vô Tri | 16.8% | 7/23 | 9.5% | 5/15 | 13.9% | 12/38 (31.6%) |
| Geometric Rules | 13.0% | 9/23 | 8.3% | 9/15 | 11.2% | 18/38 (47.4%) |
| Expert Typography Analyst | 12.8% | 7/23 | **2.8%** | **12/15** | **8.9%** | **19/38 (50.0%)** |
| Precise Scanner | 17.2% | 5/23 | 6.7% | 10/15 | 13.0% | 15/38 (39.5%) |

#### Results — Gemini 2.5 Flash (`gemini-2.5-flash`)

| Prompt | Gen CER | Gen exact | DS CER | DS exact | Combined CER | Combined exact |
|--------|---------|-----------|--------|----------|--------------|----------------|
| Default (English) | 19.8% | 5/23 | 6.5% | 8/15 | 14.5% | 13/38 (34.2%) |
| Vietnamese Vô Tri | 21.7% | 4/23 | 11.6% | 7/15 | 17.7% | 11/38 (28.9%) |
| Geometric Rules | 16.8% | 7/23 | 15.4% | 2/15 | 16.2% | 9/38 (23.7%) |
| Expert Typography Analyst | 19.0% | 5/23 | 8.8% | 6/15 | 15.0% | 11/38 (28.9%) |
| Precise Scanner | 21.7% | 4/23 | 12.9% | 3/15 | 18.2% | 7/38 (18.4%) |

#### Results — Gemini 2.5 Flash Lite (`gemini-2.5-flash-lite`, dataset `data/current_dataset`)

| Prompt | Gen CER | Gen exact | DS CER | DS exact | Combined CER | Combined exact |
|--------|---------|-----------|--------|----------|--------------|----------------|
| Default (English) | 61.6% | 2/23 | 8.4% | 5/13 | 42.4% | 7/36 (19.4%) |
| Vietnamese Vô Tri | 49.9% | 2/23 | 8.1% | 5/13 | 34.8% | 7/36 (19.4%) |
| Geometric Rules | 48.8% | 2/23 | **7.0%** | **7/13** | 33.7% | 9/36 (25.0%) |
| Expert Typography Analyst | 51.5% | 2/23 | 10.1% | 6/13 | 36.6% | 8/36 (22.2%) |
| Precise Scanner | **40.4%** | **3/23** | 7.8% | 6/13 | **28.6%** | **9/36 (25.0%)** |

#### Key findings

1. **Gemini 3 Flash Preview** — Strong overall: **Expert Typography Analyst** achieves the best combined CER (~8.9%) and dataset CER (~2.8%). **Geometric Rules** helps most on generated CER (~13.0%) among the five prompts.
2. **Gemini 2.5 Flash** — Roughly tracks 3 Preview on generated images but **dataset accuracy collapses on Geometric Rules** (15.4% DS CER, 2/15 exact), so that prompt is a poor fit for this model on clean calligraphy.
3. **Gemini 2.5 Flash Lite** — Much weaker on **AI-generated** baseline images (hallucinated extra phrases, English debris, ~40–62% gen CER) while remaining **usable on real dataset crops** (~7–10% DS CER on 13 valid images). **Precise Scanner** yields the best generated CER and combined CER in this run; **Geometric Rules** is best for dataset exact rate (7/13).
4. **Comparability** — Lite combined metrics use **36** items (23 gen + 13 DS); 3 Preview and 2.5 Flash use **38** (23 + 15). Replacing or fixing the two corrupt JPEGs would restore 15 dataset samples for Lite.

**Reproduce (2.5 Flash Lite + current dataset):**

```bash
python experiments/scripts/eval_ocr_prompt_compare.py \
  --engines gemini_2_5_lite \
  --prompts default_en,vi_vo_tri,geometric,expert_typography,precise_scanner \
  --dataset_root data/current_dataset \
  --n_dataset_samples 15 \
  --generated_dir experiments/results/baseline_4w \
  --output_dir experiments/results/ocr_gemini_25_flash_lite_current_dataset
```

---

## 15. Corrected VNCAL-4W Scores (Human-Verified Ground Truth)

### 15.1 Motivation

Sections 1-10 evaluated generation quality by running **Vintern-3B OCR** on each generated image, then comparing OCR output to the prompt ground truth (GT). Section 11.4 revealed that Vintern-3B **silently autocorrects** malformed characters into valid Vietnamese words, inflating CA and DIS scores. For example:

| TC | Image Actually Shows | OCR Read | Prompt GT | Old CA | Old DIS |
|----|---------------------|----------|-----------|--------|---------|
| TC6 | `Phụ́c` (combining acute on ụ) | `Phúc` ✗ | `Phúc` | 1.0 | 1.0 |
| TC10 | `Hấnh Phụ̃c` (combining tilde on ụ) | `Hạnh Phúc` ✗ | `Hạnh Phúc` | 1.0 | 1.0 |
| TC14 | `Phụ́c Mấn Doŭng` | `Phúc Mãn Doùng` ✗ | `Phúc Mãn Đường` | 0.79 | 0.50 |

In each case, OCR autocorrected the image text to something closer to the prompt GT, giving artificially high scores. The **true** generation quality is worse than reported.

### 15.2 Methodology

- **CA and DIS**: Recomputed using **Human-Verified GT** (Section 11.2, `GENERATED_IMAGE_GROUND_TRUTH` dict) instead of OCR output
- **LAS and SCS**: Unchanged — these metrics use image spatial/visual features, not OCR text
- **Composite**: Recomputed from corrected CA, DIS, and unchanged LAS, SCS
- **Fallback**: TC7 (rendered Chinese `上善`, already CA=0/DIS=0) and TC23 (numerals `8028`, OCR was correct) use original scores — human-verified GT not needed
- **Script**: `experiments/scripts/eval_corrected_scores_4w.py`

### 15.3 Corrected Overall Scores

| Metric | OCR-Based (Sec 2) | Corrected (Human-Verified) | Delta | Relative Change |
|--------|-------------------|---------------------------|-------|-----------------|
| **CA** | 0.7535 | **0.7174** | -0.0361 | -4.8% |
| **DIS** | 0.4640 | **0.3347** | -0.1293 | **-27.9%** |
| **LAS** | 0.4822 | 0.4822 | 0.0000 | unchanged |
| **SCS** | 0.7381 | 0.7381 | 0.0000 | unchanged |
| **Composite** | 0.6238 | **0.5788** | -0.0450 | -7.2% |

**DIS drops by 28%** — the largest correction. OCR autocorrect was primarily masking diacritic rendering errors. The composite drops from 0.6238 to 0.5788 but remains **Acceptable** (≥0.50).

**Corrected Rating Distribution:**

| Rating | CA | DIS | Composite |
|--------|-----|-----|-----------|
| Excellent | 7 | 7 | 1 |
| Good | 1 | 0 | 6 |
| Acceptable | 6 | 1 | 8 |
| Fail | 11 | **17** | 10 |

DIS now has **17/25 Fail** ratings (up from the OCR-based count), confirming diacritic rendering is the model's weakest area.

### 15.4 Per-TC Corrected Results

| TC | Words | Prompt GT | Source | Old CA | New CA | ΔCA | Old DIS | New DIS | ΔDIS | Old Comp | New Comp | ΔComp |
|----|-------|-----------|--------|--------|--------|-----|---------|---------|------|----------|----------|-------|
| TC1 | 1 | Nhẫn | HV | 0.75 | 0.75 | — | 0.00 | 0.00 | — | 0.502 | 0.502 | — |
| TC2 | 2 | An Nhiên | HV | 1.00 | 1.00 | — | 1.00 | 1.00 | — | 0.846 | 0.846 | — |
| TC3 | 3 | Vạn Sự An | HV | 0.78 | 0.78 | — | 0.00 | 0.00 | — | 0.522 | 0.522 | — |
| **TC4** | 4 | Phúc đức trường tồn | HV | 0.84 | **0.65** | **-0.19** | 0.40 | **0.20** | **-0.20** | 0.631 | **0.514** | -0.12 |
| TC5 | 1 | Tâm | HV | 1.00 | 1.00 | — | 1.00 | 1.00 | — | 0.830 | 0.830 | — |
| **TC6** | 1 | Phúc | HV | 1.00 | **0.60** | **-0.40** | 1.00 | **0.00** | **-1.00** | 0.829 | **0.439** | **-0.39** |
| TC7 | 1 | Lộc | FB | 0.00 | 0.00 | — | 0.00 | 0.00 | — | 0.235 | 0.235 | — |
| TC8 | 1 | Trí | HV | 0.50 | 0.50 | — | 0.00 | 0.00 | — | 0.414 | 0.414 | — |
| TC9 | 2 | Bình An | HV | 0.57 | 0.57 | — | 0.00 | 0.00 | — | 0.436 | 0.436 | — |
| **TC10** | 2 | Hạnh Phúc | HV | 1.00 | **0.70** | **-0.30** | 1.00 | **0.00** | **-1.00** | 0.845 | **0.490** | **-0.36** |
| TC11 | 2 | Tri Ân | HV | 1.00 | 1.00 | — | 1.00 | 1.00 | — | 0.840 | 0.840 | — |
| TC12 | 2 | An Khang | HV | 1.00 | 1.00 | — | 1.00 | 1.00 | — | 0.839 | 0.839 | — |
| **TC13** | 3 | Tâm Bất Biến | HV | 0.67 | **0.75** | **+0.08** | 0.33 | 0.33 | — | 0.550 | **0.580** | +0.03 |
| **TC14** | 3 | Phúc Mãn Đường | HV | 0.79 | **0.60** | **-0.19** | 0.50 | **0.00** | **-0.50** | 0.641 | **0.451** | **-0.19** |
| **TC15** | 3 | Thuận Tự Nhiên | HV | 0.71 | **0.79** | **+0.07** | 0.33 | **0.00** | **-0.33** | 0.576 | **0.518** | -0.06 |
| TC16 | 4 | Vạn sự như ý | HV | 0.36 | 0.36 | — | 0.00 | 0.00 | — | 0.440 | 0.440 | — |
| **TC17** | 4 | An khang thịnh vượng | HV | 0.52 | **0.64** | **+0.11** | 0.00 | 0.00 | — | 0.423 | **0.462** | +0.04 |
| TC18 | 1 | Nhẫn | HV | 0.75 | 0.75 | — | 0.00 | 0.00 | — | 0.592 | 0.592 | — |
| **TC19** | 4 | Phúc đức trường tồn | HV | 0.55 | **0.43** | **-0.12** | 0.20 | **0.00** | **-0.20** | 0.474 | **0.382** | -0.09 |
| TC20 | 2 | Tài Lộc | HV | 0.86 | 0.86 | — | 0.50 | 0.50 | — | 0.662 | 0.662 | — |
| TC21 | 3 | Tâm Bất Biến | HV | 0.69 | 0.69 | — | 0.33 | 0.33 | — | 0.516 | 0.516 | — |
| TC22 | 1 | Tâm | HV | 1.00 | 1.00 | — | 1.00 | 1.00 | — | 0.834 | 0.834 | — |
| TC23 | 2 | 8028 | FB | 1.00 | 1.00 | — | 1.00 | 1.00 | — | 0.895 | 0.895 | — |
| TC24 | 2 | Thanh Tâm | HV | 1.00 | 1.00 | — | 1.00 | 1.00 | — | 0.838 | 0.838 | — |
| **TC25** | 3 | Nhẫn Nhường Nhịn | HV | 0.50 | **0.53** | **+0.03** | 0.00 | 0.00 | — | 0.382 | **0.393** | +0.01 |

**Source:** HV = Human-Verified, FB = Fallback (OCR). Bold = changed values. 9 of 25 TCs affected.

### 15.5 Most Affected Test Cases

**TC6 (Phúc)** — Largest single drop: Composite 0.829 → 0.439
- Image rendered `Phụ́c` (combining acute on ụ instead of standard ú)
- OCR autocorrected `Phụ́c` → `Phúc`, perfectly matching the prompt → CA=1.0, DIS=1.0
- Corrected: CA=0.60 (2 of 5 chars differ in NFC form), DIS=0.00 (the single diacritic char is wrong)

**TC10 (Hạnh Phúc)** — Second largest drop: Composite 0.845 → 0.490
- Image rendered `Hấnh Phụ̃c` (ạ→ấ wrong tone, ú→ụ̃ combining tilde)
- OCR autocorrected to `Hạnh Phúc` (exact prompt match) → CA=1.0, DIS=1.0
- Corrected: CA=0.70, DIS=0.00 (both diacritic chars have wrong tone marks)

**TC14 (Phúc Mãn Đường)** — Third largest drop: Composite 0.641 → 0.451
- Image rendered `Phụ́c Mấn Doŭng` (multiple diacritic errors + Đ→D)
- OCR read `Phúc Mãn Doùng` (partially autocorrected: Phụ́c→Phúc, Mấn→Mãn)
- Corrected: CA=0.60, DIS=0.00 (all 4 diacritic chars have wrong marks)

**Notable upward corrections:**
- **TC13**: CA 0.67→0.75 (+0.08) — OCR misread worse than the actual image text differs from prompt
- **TC17**: CA 0.52→0.64 (+0.11) — OCR introduced extra errors not present in the image

### 15.6 Corrected Averages by Word Count

| Word Count | n | CA | DIS | LAS | SCS | Composite | Rating |
|------------|---|------|------|------|------|-----------|--------|
| 1-Word | 7 | 0.6571 | 0.2857 | 0.4812 | 0.7592 | 0.5495 | Acceptable |
| 2-Word | 8 | 0.8911 | 0.6875 | 0.4808 | 0.7538 | 0.7307 | Good |
| 3-Word | 6 | 0.6892 | 0.1111 | 0.4440 | 0.6940 | 0.4966 | **Fail** |
| 4-Word | 4 | 0.5180 | 0.0500 | 0.5438 | 0.7360 | 0.4497 | **Fail** |

Compared to OCR-based averages:
- **1-Word**: CA 0.84→0.66 (-0.18), DIS 0.50→0.29 (-0.21) — TC6 autocorrect had outsized effect
- **2-Word**: CA 0.93→0.89 (-0.04), DIS 0.75→0.69 (-0.06) — TC10 autocorrect partially offset by accurate TCs
- **3-Word**: CA 0.67→0.69 (+0.02), DIS 0.20→0.11 (-0.09) — OCR misreadings partially cancelled out autocorrect
- **4-Word**: CA 0.57→0.52 (-0.05), DIS 0.15→0.05 (-0.10) — long phrases have almost zero diacritic fidelity

### 15.7 Key Findings

1. **DIS is 28% worse than OCR-based measurement** (0.464 → 0.335). Diacritic rendering quality was the most significantly overstated metric due to OCR autocorrect. With 17/25 DIS ratings at Fail, diacritic fidelity is the model's most critical weakness.

2. **CA drops modestly** (0.754 → 0.717, -4.8%). Some TCs actually improved (TC13, TC17, TC25) because Vintern-3B sometimes misreads characters in ways that *increase* edit distance to the prompt, so human-verified GT can be closer to the prompt than OCR output.

3. **Composite drops from 0.624 to 0.579** (-7.2%), remaining Acceptable but now closer to the Fail boundary (0.50). The true baseline is weaker than previously reported.

4. **3-Word and 4-Word phrases now rate Fail** in composite (0.497 and 0.450 respectively), dropping below the 0.50 threshold. This was masked by OCR autocorrect in the original evaluation.

5. **The case for LoRA fine-tuning is even stronger.** The corrected DIS of 0.335 is far below the target of 0.80, and the corrected composite of 0.579 is well below the target of 0.85. The gap that LoRA must close is larger than originally measured.

6. **OCR-based evaluation is unreliable for generation quality assessment.** Any OCR engine that performs autocorrection will inflate CA and DIS scores. Future evaluations should use either human verification or strict-literal OCR engines for generated image assessment.

---

*Document updated: 2026-03-30 | Reference: EVAL-VNCAL-4W-v1.0 | Model: Qwen-Image-2512 (base)*
*Scripts: `experiments/scripts/eval_baseline_4w.py`, `metrics_4w.py`, `enc_tests.py`, `vae_tests.py`, `eval_ocr_accuracy.py`, `eval_ocr_prompt_compare.py`, `ocr_engines.py`, `test_new_engines.py`, `test_all_engines_comprehensive.py`, `test_gemini3flash_full.py`, `eval_corrected_scores_4w.py`*
