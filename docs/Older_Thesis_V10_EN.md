# MASTER'S THESIS

---

**Thesis Title:**
# Fine-tuning the Ideogram4 Diffusion Transformer for Generating Vietnamese Calligraphy Images

---

**Field:** Computer Science - Specialization: Artificial Intelligence  
**Year of Defense:** 2026

> **Version Note (V10):** This thesis draft updates the new research direction (transitioning from Qwen-Image/ERNIE-Image models (V8–V9.2) to **Ideogram4**). This update focuses on **Section 3.5 (Region-Grouped DPO (R-GDPO) for diacritic generation)**, which improves accuracy on a difficult dataset from 57% to 85% over 5 refinement rounds, along with critical configuration adjustments—such as normalizing bbox coordinates to `[y, x, y, x]` and experimenting with bbox removal during inference.

---

## TABLE OF CONTENTS

- [Acknowledgments](#acknowledgments)
- [List of Tables](#list-of-tables)
- [List of Figures](#list-of-figures)
- [List of Appendices](#list-of-appendices)
- [1. Introduction](#1-introduction)
  - [1.1. Problem Statement](#11-problem-statement)
  - [1.2. Research Objectives and Scope](#12-research-objectives-and-scope)
  - [1.3. Process Overview and Domain-Specific Challenges](#13-process-overview-and-domain-specific-challenges)
    - [1.3.1. Overall Pipeline](#131-overall-pipeline)
    - [1.3.2. Domain-Specific Challenges](#132-domain-specific-challenges)
  - [1.4. Literature Review](#14-literature-review)
    - [1.4.1. GAN-Based Approaches](#141-gan-based-approaches)
    - [1.4.2. Diffusion Models](#142-diffusion-models)
    - [1.4.3. Diffusion Transformer and Large Language Models](#143-diffusion-transformer-and-large-language-models)
    - [1.4.4. Efficient Fine-tuning and Quantization](#144-efficient-fine-tuning-and-quantization)
    - [1.4.5. Commercial AI Models and Research Gaps](#145-commercial-ai-models-and-research-gaps)
  - [1.5. Proposed Method and Contributions](#15-proposed-method-and-contributions)
  - [1.6. Thesis Structure](#16-thesis-structure)
- [2. Theoretical Background](#2-theoretical-background)
  - [2.1. AI Image Generation - From GAN to Diffusion Transformer](#21-ai-image-generation---from-gan-to-diffusion-transformer)
    - [2.1.1. Generative Adversarial Network (GAN)](#211-generative-adversarial-network-gan)
    - [2.1.2. Denoising Diffusion Probabilistic Models (DDPM)](#212-denoising-diffusion-probabilistic-models-ddpm)
    - [2.1.3. Latent Diffusion and Flow Matching](#213-latent-diffusion-and-flow-matching)
  - [2.2. Ideogram4 Architecture](#22-ideogram4-architecture)
    - [2.2.1. Overview of Three-Component Architecture](#221-overview-of-three-component-architecture)
    - [2.2.2. Single-Stream MMDiT and Masked Self-Attention](#222-single-stream-mmdit-and-masked-self-attention)
    - [2.2.3. Qwen3-VL Text Encoder and 13-Layer Concatenation](#223-qwen3-vl-text-encoder-and-13-layer-concatenation)
    - [2.2.4. Structured JSON Prompt and Bounding Box](#224-structured-json-prompt-and-bounding-box)
    - [2.2.5. Default Inference Parameters and Custom CFG](#225-default-inference-parameters-and-custom-cfg)
  - [2.3. Efficient Fine-tuning on FP8 Weights](#23-efficient-fine-tuning-on-fp8-weights)
    - [2.3.1. Low-Rank Adaptation (LoRA)](#231-low-rank-adaptation-lora)
    - [2.3.2. The Problem of LoRA on FP8 Linear Layers](#232-the-problem-of-lora-on-fp8-linear-layers)
    - [2.3.3. Mixed Precision and Memory Optimization](#233-mixed-precision-and-memory-optimization)
  - [2.4. Vietnamese Calligraphy - Visual and Linguistic Characteristics](#24-vietnamese-calligraphy---visual-and-linguistic-characteristics)
    - [2.4.1. Cultural Context](#241-cultural-context)
    - [2.4.2. Geometric Characteristics of Brush Strokes](#242-geometric-characteristics-of-brush-strokes)
    - [2.4.3. Tonal System and Unicode Standard](#243-tonal-system-and-unicode-standard)
  - [2.5. Overall System Architecture](#25-overall-system-architecture)
- [3. Implementation and Evaluation](#3-implementation-and-evaluation)
  - [3.1. System Setup](#31-system-setup)
    - [3.1.1. Hardware Configuration](#311-hardware-configuration)
    - [3.1.2. Software Environment](#312-software-environment)
    - [3.1.3. Anti-Outbid Workflow](#313-anti-outbid-workflow)
  - [3.2. Data and Preprocessing](#32-data-and-preprocessing)
    - [3.2.1. Data Source and Schema Conversion (Synthetic Data)](#321-data-source-and-schema-conversion-synthetic-data)
    - [3.2.2. Per-Sample Bounding Box](#322-per-sample-bounding-box)
    - [3.2.3. Compound Word Expansion (Phase 4)](#323-compound-word-expansion-phase-4)
    - [3.2.4. Preprocessing](#324-preprocessing)
  - [3.3. FP8-LoRA Training Infrastructure and Configuration](#33-fp8-lora-training-infrastructure-and-configuration)
    - [3.3.1. `Fp8LoRALinear` Wrapper and LoRA Injector](#331-fp8loralinear-wrapper-and-lora-injector)
    - [3.3.2. Target Strategy - DiT Self-Attention Only](#332-target-strategy---dit-self-attention-only)
    - [3.3.3. Flow-Matching Loss with Diacritic Mask](#333-flow-matching-loss-with-diacritic-mask)
    - [3.3.4. Hyperparameter Table](#334-hyperparameter-table)
    - [3.3.5. Phased Implementation Strategy](#335-phased-implementation-strategy)
  - [3.4. Results and Evaluation](#34-results-and-evaluation)
    - [3.4.1. Phase 0: Raw Inference Validation (Pivot Justification)](#341-phase-0-raw-inference-validation-pivot-justification)
    - [3.4.2. Phase 1: 11 Architectural Probes](#342-phase-1-11-architectural-probes)
    - [3.4.3. Phase 2: FP8-LoRA Infrastructure Smoke Test](#343-phase-2-fp8-lora-infrastructure-smoke-test)
    - [3.4.4. Baseline 14k Training Results](#344-baseline-14k-training-results)
    - [3.4.5. Comparison with Predecessor Baseline (V9.2)](#345-comparison-with-predecessor-baseline-v92)
  - [3.5. Resolving the Diacritic Bottleneck with Region-Grouped DPO (R-GDPO)](#35-resolving-the-diacritic-bottleneck-with-region-grouped-dpo-r-gdpo)
    - [3.5.1. Micro Diacritic Classifier (Oracle) - Iterative Reinforcement](#351-micro-diacritic-classifier-oracle---iterative-reinforcement)
    - [3.5.2. Seed-Variance Diagnosis: Sampling Error vs. Weight Error](#352-seed-variance-diagnosis-sampling-error-vs-weight-error)
    - [3.5.3. Mask-SFT (Tone-Zone) Hits Ceiling - Theoretical Confirmation](#353-mask-sft-tone-zone-hits-ceiling---theoretical-confirmation)
    - [3.5.4. R-GDPO Method and 5-Round Results](#354-r-gdpo-method-and-5-round-results)
    - [3.5.5. Generalization to Complex Vowels - Vowel-Specific Error Axes](#355-generalization-to-complex-vowels---vowel-specific-error-axes)
  - [3.6. Ongoing Mitigation Directions (Phase 3 Expansion & Phase 4)](#36-ongoing-mitigation-directions-phase-3-expansion--phase-4)
- [4. Conclusion](#4-conclusion)
  - [4.1. Summary of Theoretical and Practical Value](#41-summary-of-theoretical-and-practical-value)
    - [4.1.1. Core Contributions](#411-core-contributions)
    - [4.1.2. Theoretical Contributions](#412-theoretical-contributions)
    - [4.1.3. Practical Contributions](#413-practical-contributions)
  - [4.2. Summary of Key Results](#42-summary-of-key-results)
  - [4.3. Current Limitations](#43-current-limitations)
  - [4.4. Future Development Directions](#44-future-development-directions)
- [References](#references)
- [Appendices](#appendices)

---

## ABSTRACT

Quốc ngữ calligraphy is a unique and increasingly popular art form in Vietnam. However, applying artificial intelligence to automate the creation of high-quality, style-controllable calligraphy works remains limited. This research presents the development of an automated Vietnamese calligraphy image generation system through Low-Rank Adaptation (LoRA) fine-tuning on the **Ideogram4** model (a multimodal Diffusion Transformer (DiT) image generation model). This model utilizes the **Qwen3-VL-8B-Instruct** text encoder and accepts input in the form of a **structured JSON prompt, including text elements accompanied by bounding boxes**.

Initial research based on the predecessor model (Qwen-Image/ERNIE-Image, version V9.2) demonstrated feasibility but revealed two main limitations: (i) difficulty in accurately aligning Vietnamese syllables with text tokens, and (ii) limitations arising from using only a single layer of the text encoder for signal extraction. The Ideogram4 architecture was chosen to overcome these limitations: the bounding box mechanism allows for more natural character positioning, and the encoder's concatenation of features from 13 layers (totaling 53,248 dimensions) facilitates better transmission of tonal signals into the DiT.

The primary technical contribution of this research is the **development of an FP8-weighted LoRA training pipeline for Ideogram4**. Since the current DiffSynth-Studio tool does not support LoRA training for the `Fp8Linear` layers of this model, the research proposes: (1) using an `Fp8LoRALinear` wrapper that allows gradient computation through the bf16 LoRA branch while keeping the original FP8 weights frozen; (2) a strategy of fine-tuning only the self-attention blocks of the DiT (approximately 60.2 million parameters), based on the empirical observation that the text encoder's ability to discriminate diacritics degrades from the 9th layer onward; (3) preparing a dataset of 7,866 single-word images in JSON format, with detailed bounding box coordinates calculated for each sample (achieving an IoU of 0.973 with the actual ink region).

Experimental results show that: (1) The pre-fine-tuned Ideogram4 model has a **relatively low character misrecognition rate**, but only **2 out of 6 words were completely correct in terms of diacritics**. This indicates good character formation quality, but correct diacritic generation remains a separate problem to be solved; (2) Structural probes helped identify mandatory requirements (such as bbox coordinates and font names to avoid generating Chinese characters by mistake); (3) After 14,177 training steps, the model significantly reduced diacritic errors and could generate good images even without providing bbox coordinates during inference.

To address the bottleneck in distinguishing complex diacritics, the research applied the **Region-Grouped Direct Preference Optimization (R-GDPO)** method. By building a diacritic classifier to evaluate errors and combining it with region-based preference optimization, the whole-word recognition accuracy on a difficult dataset increased from **57% to 85%** over 5 refinement rounds. This result demonstrates that preference optimization is more effective than standard regression methods in correcting diacritic generation errors.

This research provides a **complete LoRA fine-tuning pipeline for DiT models with FP8 weights**, contributing to the problem of generating artistic images requiring high linguistic accuracy, with potential applications in graphic design and the preservation of digital calligraphy culture.

**Keywords:** Vietnamese Calligraphy, Fine-tuning, LoRA, Diffusion Transformer, Ideogram4, Qwen3-VL, Text-to-Image Generation, Bounding-box conditioning, FP8 quantization, MMDiT.

---

## ACKNOWLEDGMENTS

*(To be completed by the author)*

---

## LIST OF TABLES

- **TABLE 2.1:** Overview of Ideogram4's Three-Component Architecture
- **TABLE 3.1:** Hardware Configuration for Ideogram4 Fine-tuning
- **TABLE 3.2:** Software Environment for Ideogram4 Fine-tuning
- **TABLE 3.3:** LoRA Target Modules in DiT
- **TABLE 3.4:** Complete Hyperparameters Table for FP8-LoRA Training
- **TABLE 3.5:** Phase 0 Raw Inference Validation Results
- **TABLE 3.6:** Comparison with Predecessor Baseline (V9.2 vs V10)
- **TABLE 3.7:** R-GDPO 5-Round Whole-Word Accuracy Results
- **TABLE 3.8:** Generalization to Complex Vowels (Stress Panel)
- **TABLE 4.1:** Summary of Key Experimental Results

---

## LIST OF FIGURES

- **FIGURE 1.1:** Overall Pipeline of Vietnamese Calligraphy Generation
- **FIGURE 2.1:** Ideogram4 Architecture Overview
- **FIGURE 2.2:** Single-Stream MMDiT and Masked Self-Attention Mechanism
- **FIGURE 2.3:** Geometric Characteristics of Vietnamese Calligraphy Brush Strokes
- **FIGURE 3.1:** Per-Sample Bounding Box Calculation and IoU Validation
- **FIGURE 3.2:** Fp8LoRALinear Wrapper and Gradient Flow
- **FIGURE 3.3:** Flow-Matching Loss with Diacritic Mask
- **FIGURE 3.4:** Phase 1 Architectural Probes: Qwen3-VL Hidden State Diacritic Discrimination
- **FIGURE 3.5:** R-GDPO Training Pipeline and Preference Pair Mining

---

## LIST OF APPENDICES

- **APPENDIX A:** Full Hyperparameter Configuration Script
- **APPENDIX B:** Source Code of the Fp8LoRALinear Wrapper and Injector
- **APPENDIX C:** Detailed Evaluation Samples from R-GDPO Checkpoints
- **APPENDIX D:** Micro Diacritic Classifier (Oracle) Architecture and Training Details

---

## 1. INTRODUCTION

### 1.1. Problem Statement

In the contemporary Vietnamese cultural and artistic landscape, Quốc ngữ calligraphy (also known as "modern calligraphy" or "Latin calligraphy") is undergoing a strong renaissance. Unlike traditional Sino-Vietnamese calligraphy, which uses Chinese characters, Quốc ngữ calligraphy is based on the Latin alphabet augmented with tonal diacritics, combined with brush and ink techniques to create works imbued with traditional calligraphic style. These works frequently appear in exhibition spaces, as gifts for Tet (Lunar New Year), for interior decoration, or as backgrounds for brands carrying Vietnamese cultural identity.

However, creating a high-quality calligraphy piece requires years of practice by artisans, and the demand for personalization (e.g., writing specific names or custom wishes) makes it difficult to scale production. This is a problem that Generative AI is well-equipped to solve, especially given that Diffusion Models have achieved superior image generation quality.

Practical applications of an automated Vietnamese calligraphy image generation system include: (i) **Graphic Design** (rapidly generating custom posters, banners, and greeting cards); (ii) **E-commerce** (customizing printed products like framed art, apparel, and ceramics with customer names); (iii) **Art Education** (providing diverse learning materials on calligraphy); and (iv) **Digital Cultural Preservation** (digitizing and recreating fading calligraphic styles).

The key differentiator of this research from existing tools is not merely "generating calligraphy images," but rather **controlling the style according to a specific font**, ensuring reproducibility and scalability. As analyzed in Section 1.4, current commercial models only output a fixed default style, while rendering from machine fonts completely lacks the natural vitality of brush strokes.

### 1.2. Research Objectives and Scope

The central objective of this thesis is to **build a high-quality text-to-image Vietnamese calligraphy generation system**, where the model is required to simultaneously satisfy two criteria: (1) *Orthographic Accuracy* (all characters, tonal diacritics, and diacritical marks must be rendered correctly and legibly); and (2) *Calligraphic Aesthetics* (brush strokes must exhibit clear thick-thin differentiation, evoking the wrist movement of a human artisan).

The research scope is limited as follows:
- **Content Length:** 1 to 4 syllables (single-word and compound), prioritizing common compound phrases (names, idioms, greetings).
- **Calligraphy Font:** "Thu Phap Thanh Cong" (a single font during the research phase).
- **Output Resolution:** 1024×1024 pixels (the native resolution of Ideogram4, a multiple of 16).
- **Base Model:** **Ideogram4** (`ideogram-ai/ideogram-4-fp8`), inferred via DiffSynth-Studio.
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation) on FP8 weights—no full-parameter fine-tuning, no modification of the text encoder.

A crucial scoping decision: this research **inherits and transitions the model** from the predecessor branch (Qwen-Image/ERNIE-Image, V9.2). The predecessor branch serves as a *baseline for comparison* and a *source of technical lessons* (especially regarding "Text Collapse", LoRA rank challenges, and diacritic issues), but its *architecture-specific findings* (single-layer injection, 3-axis RoPE, `self_attention.*` module paths) **do not transfer** to Ideogram4 and are entirely replaced.

### 1.3. Process Overview and Domain-Specific Challenges

#### 1.3.1. Overall Pipeline

Previous research on the ERNIE-Image model (V9.2), while achieving some basic results, exposed two architectural limitations:
1. **Difficulty in syllable positioning:** Requiring the model to draw each Vietnamese syllable in a specific location on the image from plain text input demands complex auxiliary techniques.
2. **Limitations in text feature extraction:** Tonal signals were extracted from a single hidden layer of the encoder, making it difficult to comprehensively capture linguistic information.

These issues often led to generating images with incorrect strokes despite correct diacritic recognition. The **Ideogram4** model is designed to overcome these limitations. The current system (V10) is built in five phases:

- **Phase 0 (Base Model Validation):** Testing the raw Ideogram4 (unfine-tuned) on difficult calligraphy cases to assess feasibility before building the training infrastructure.
- **Phase 1 (Data Preparation and Model Analysis):** Building a data converter to the JSON format required by Ideogram4, and running probes to analyze the behavior of the tokenizer and text encoder.
- **Phase 2 (FP8-LoRA Training Infrastructure):** Proposing and implementing the `Fp8LoRALinear` wrapper and a custom loss function to enable LoRA training on FP8-weighted models.
- **Phase 3 (Baseline Training):** Fine-tuning the model on a dataset of 7,866 single-word images and evaluating quality.
- **Phase 4 (Compound Word Expansion):** Inheriting the best results from Phase 3, proceeding with continuous training on data containing multi-syllable phrases.

#### 1.3.2. Domain-Specific Challenges

- **Challenge 1: Complexity of the Vietnamese Tonal System.** Vietnamese has 6 tones and 12 basic vowels, creating approximately 134 character variants. In calligraphy, diacritics are not mere symbols; they must be drawn with soft, balanced brush strokes harmonizing with the overall character.
- **Challenge 2: Processing on FP8 Models.** Ideogram4 is distributed with FP8 weights, and current supporting libraries (DiffSynth-Studio) do not provide standard training mechanisms for this format. Designing a stable gradient flow for LoRA on frozen FP8 base weights is a major technical challenge.
- **Challenge 3: Precise Diacritic Discrimination.** Models often struggle to distinguish visually similar diacritics (e.g., tilde vs. dot below). Analysis in Chapter 3 shows that the text encoder's ability to discriminate diacritics degrades in deeper layers, necessitating a specialized optimization approach.

### 1.4. Literature Review

#### 1.4.1. GAN-Based Approaches
Generative Adversarial Networks (GAN) [1] were among the earliest methods applied to text image generation. Works like **zi2zi** [2] (2017) and **CalliGAN** [3] (2020) demonstrated the ability to learn Chinese calligraphy styles via Image-to-Image translation. **CycleGAN** [4] (Zhu et al., 2017) extended this to translate between styles without parallel data. However, GAN approaches have two fundamental limitations for Vietnamese calligraphy: (i) *mode collapse* (failing to cover all 134 character variants); (ii) *lack of conditional text generation capability* (most are Image-to-Image, not Text-to-Image).

#### 1.4.2. Diffusion Models
**DDPM** [5] (Ho et al., 2020) and **DDIM** [6] (Song et al., 2020) marked a turning point in image generation. **Latent Diffusion Models (LDM)** [7] (Rombach et al., 2022), the basis of Stable Diffusion, introduced diffusion in latent space. Works applying this to in-image text, such as **TextDiffuser** [8] (Chen et al., 2023) and **GlyphControl** [9] (Yang et al., 2023), improved Latin character accuracy but still rely on CLIP text encoders optimized for Latin, not for languages with complex diacritic systems like Vietnamese.

#### 1.4.3. Diffusion Transformer and Large Language Models
The advent of Transformer architectures and Vision-Language models like **CLIP** [10], **BLIP-2** [11], and **Multimodal Large Language Models (MLLMs)** opened the door to integrating deep language understanding into image generation. **FLUX** [12] (Black Forest Labs, 2024) and **Wan** [13] (Alibaba, 2024) are prime representatives of the Diffusion Transformer (DiT) generation.

**Ideogram4** [14] advances this in two directions crucial for Vietnamese. First, its text encoder is **Qwen3-VL-8B-Instruct** [29], and the model **concatenates hidden states from 13 layers** (0, 3, 6, …, 33, 35) into a 53,248-dimensional feature, avoiding the "choose a single layer" dilemma. Second, the model accepts a **structured JSON prompt**, where each text element is accompanied by a bounding box (bbox), making span alignment a native property of the model.

#### 1.4.4. Efficient Fine-tuning and Quantization
**DreamBooth** [15] (Ruiz et al., 2022) allows personalization of Stable Diffusion but requires full-parameter fine-tuning. **LoRA** [16] (Hu et al., 2021) decomposes the update matrix into the product of two low-rank matrices, reducing trainable parameters from billions to tens of millions. In the context of **FP8** [21] base weights, a new challenge arises: how to train a bf16 adapter while the base weights are frozen 8-bit floating-point numbers. This thesis contributes a specific solution to this problem (Sections 2.3, 3.3).

#### 1.4.5. Commercial AI Models and Research Gaps
As of early 2026, several commercial tools can generate Vietnamese calligraphy-style images:
- **Nano Banana 2 (Google, 2025–2026):** Shows significant diacritic accuracy but outputs only a **fixed default style**; users **cannot specify font or brush style**.
- **GPT Image 1.5 (High mode) (OpenAI, 2025–2026):** Similarly, high visual quality but fixed default style and **black-box API**—cannot be fine-tuned on private data or extended to new fonts.
- **Machine Font Rendering (.ttf/.otf):** Convenient and consistent, but produces images that are merely **mechanical arrangements of static glyphs**—lacking brush pressure variance, ink flow effects, random "dry brush" (phi bạch), or aesthetic resonance between characters.

**Research Gap:** There is no open-source work publishing a Vietnamese calligraphy generation model that **controls style by specific font, is reproducible, and scalable**, especially on an FP8-weighted DiT model. This is the gap this thesis fills.

### 1.5. Proposed Method and Contributions

The core contribution of this research is the **successful construction of an automated Vietnamese calligraphy image generation system capable of font-specific style control with high linguistic accuracy, based on the Diffusion Transformer (Ideogram4)**.

To achieve this, the research implemented several supporting technical solutions:
- **Developing an FP8-LoRA fine-tuning pipeline:** Proposing and implementing the `Fp8LoRALinear` wrapper and a custom loss function that accounts for specific weights in diacritic regions.
- **Solving the diacritic generation problem via Preference Alignment:** Applying Region-Grouped DPO (R-GDPO) combined with a diacritic classifier to overcome errors that standard regression loss cannot resolve, improving accuracy from 57% to 85%.
- **Model specification survey and data processing:** Conducting a comprehensive empirical evaluation of Ideogram4's behavior with Vietnamese and improving the dataset by calculating precise character coordinates for each image sample.

### 1.6. Thesis Structure

- **Chapter 2** presents the theoretical foundation: generative models from GANs to Diffusion Transformers, the detailed architecture of Ideogram4, the mathematics of LoRA and the problem of LoRA on FP8 weights, and the geometric-linguistic characteristics of Vietnamese calligraphy.
- **Chapter 3** details the system construction: environment setup, data conversion and preprocessing, FP8-LoRA training infrastructure design, experimental results from Phase 0 to the 14k baseline, and the **R-GDPO program resolving the diacritic bottleneck** (§3.5), elevating whole-word accuracy from 57% to 85%.
- **Chapter 4** summarizes contributions, identifies remaining limitations, and proposes future development directions.

---

## 2. THEORETICAL BACKGROUND

### 2.1. AI Image Generation - From GAN to Diffusion Transformer

#### 2.1.1. Generative Adversarial Network (GAN)
GANs [1] consist of two competing networks: a **Generator (G)** that generates fake images from noise vectors, and a **Discriminator (D)** that distinguishes real from fake. The minimax objective function is:
$$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$
Two fundamental limitations: (i) **mode collapse** (converging to a narrow set of images, failing to cover all 134 Vietnamese character variants); (ii) **training instability** (difficulty in reaching Nash equilibrium).

#### 2.1.2. Denoising Diffusion Probabilistic Models (DDPM)
**DDPM** [5] introduces a forward process adding Gaussian noise over T steps:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$$
and learns the reverse process by predicting noise:
$$\mathcal{L}_{DDPM} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$
DDPM surpasses GANs in diversity and stability but requires up to T=1000 inference steps.

#### 2.1.3. Latent Diffusion and Flow Matching
**LDM** [7] performs diffusion in **latent space** rather than pixel space: a VAE compresses the image to a latent, diffusion occurs on the latent, and a decoder decompresses it. **Flow Matching** [17] (and **Rectified Flow** [18]) learns the **velocity field** $v_t$ directly:
$$\frac{dx_t}{dt} = v_\theta(x_t, t, c), \qquad \mathcal{L}_{FM} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]$$
with $x_t = (1-t)x_0 + t\epsilon$. Straighter trajectories allow image generation with fewer steps and **better preservation of sharp edges** (crucial for calligraphy). Ideogram4 uses flow-matching with its own scheduler (`FlowMatchScheduler("Ideogram4")`) featuring log-SNR shifting based on image resolution.

### 2.2. Ideogram4 Architecture

#### 2.2.1. Overview of Three-Component Architecture
**Ideogram4** (`ideogram-ai/ideogram-4-fp8`) is a multimodal Diffusion Transformer, supported for inference by DiffSynth-Studio since commit `6d103c0` (2026-06-05). Its three components are:

| Component | Parameters | Role |
|---|---|---|
| **Condition Encoder** | Qwen3-VL-8B-Instruct, 13 hidden layers concatenated | Encodes JSON prompt, generates 53,248-dim features |
| **VAE** | 128-channel latent, ×8 compression factor | Compresses 1024×1024 image → 64×64 latent grid (4,096 tokens × 128) |
| **DiT Backbone** | Single-stream MMDiT, 34 layers, 4608 emb, 18 heads | Generates image via flow-matching ODE |

Compared to the predecessor ERNIE-Image (V9.2): emb_dim 4608 (vs 4096), 34 layers (vs 36), text encoder features 53,248 dims (vs 3,072 dims from a single layer), `rope_theta` = 5,000,000 (vs 256), MRoPE 3-axis `[24, 20, 20]`.

#### 2.2.2. Single-Stream MMDiT and Masked Self-Attention
Ideogram4 uses a **single-stream MMDiT** architecture: text and image tokens are concatenated into a unified sequence and processed through shared self-attention blocks. Unlike pure cross-attention mechanisms, Ideogram4 applies **masked self-attention**, where `segment_ids` and `indicator` delineate the role of each token group (conditioning text, input image, output image). Each block is modulated by an **AdaLN-zero gate (tanh)** based on the timestep.

A crucial detail for training: the `model_fn_ideogram4` function returns `-out[:, max_text_tokens:]` (i.e., the output requires *negation* and *slicing to only image tokens*). The training step must precisely match this convention (detailed in Section 3.3).

#### 2.2.3. Qwen3-VL Text Encoder and 13-Layer Concatenation
The `Ideogram4TextEncoder` wraps **Qwen3-VL-8B-Instruct** and outputs the **concatenation of hidden states from 13 layers** (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35), totaling **13 × 4096 = 53,248 dimensions**, which is then projected via `llm_cond_proj` (53248 → 4608) to feed into the DiT. Using multiple layers simultaneously is the design reason the "single-layer injection" dilemma of V9.2 disappears.

However, a key empirical finding (Probe 4, Section 3.4.2) reveals that *Qwen3-VL itself "loses diacritic discrimination ability" from layer 9 onwards*: pairs differing only in diacritics (e.g., Tâm/Tấm, Vượng/Vương) have cosine distances near 0 in upper layers. The direct consequence: **do not inject LoRA into the text encoder**—diacritic discrimination must come from the DiT learning from pixels.

#### 2.2.4. Structured JSON Prompt and Bounding Box
Ideogram4 accepts prompts in a structured JSON format, containing `high_level_description`, `style_description` (4 sub-fields: aesthetics/lighting/photo/medium), and `compositional_deconstruction` with an `elements` array. Each text element takes the form:
```json
{"type": "text", "bbox": [y1, x1, y2, x2], "text": "Vượng", "desc": "..."}
```
The bbox coordinate system is verified to be **normalized [0, 1000]** (mapped to a 1024×1024 image) following the **image-axis/NumPy `[y, x, y, x]`** convention (row first, column second) (*not* the PIL/CSS `[x, y, x, y]` convention). This is a critical correction (Section 3.4.2): evidence comes from geometric analysis of official DiffSynth examples (e.g., logo "F1" bbox `[657, 0, 755, 142]` describing "bottom left" only matches when reading y first), confirmed when a 2×2 grid layout initially rendered as a vertical column due to swapped coordinates. The bbox mechanism allows anchoring each Vietnamese syllable to a defined spatial region—solving the span alignment problem natively.

#### 2.2.5. Default Inference Parameters and Custom CFG
Verified default parameters: `cfg_scale = 7.0`, `num_inference_steps = 32` (optimal threshold; official examples use 48), `height = width = 1024`, `seed = 42`. Ideogram4 uses a **non-standard CFG formula**:
$$\text{noise\_pred} = \text{cfg} \cdot \text{posi} + (1 - \text{cfg}) \cdot \text{nega}$$
different from the standard CFG `nega + cfg·(posi − nega)`. During training, standard CFG dropout (dropping text) is applied, while the custom formula is used only at inference.

### 2.3. Efficient Fine-tuning on FP8 Weights

#### 2.3.1. Low-Rank Adaptation (LoRA)
LoRA [16] decomposes the update $\Delta W$ into the product of two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + BA, \qquad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)$$
$W_0$ is frozen; the forward pass uses the rsLoRA (root-stabilized) scaling factor $\alpha/\sqrt{r}$:
$$h = W_0 x + \frac{\alpha}{\sqrt{r}} \cdot BAx$$
For Ideogram4, the target is 68 attention modules (34 blocks × {qkv, o}); rank r=64, α=64 yields ~60.2 million parameters (≈0.75% of the 8B base model)—a standard LoRA budget.

#### 2.3.2. The Problem of LoRA on FP8 Linear Layers
The central obstacle of V10: Ideogram4's `Fp8Linear` layer stores FP8 weights as a **buffer** (not an `nn.Parameter`), and dequantizes *on-the-fly* during the forward pass:
```python
def forward(self, x):
    w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
    return F.linear(x, w, bias)
```
Because `weight` is a buffer, gradients **do not** flow into it (which is actually *advantageous* for the LoRA paradigm, as we want the base frozen). The proposed solution is the `Fp8LoRALinear` wrapper, which keeps `Fp8Linear` as a submodule (frozen FP8 buffer) and adds two trainable bf16 LoRA parameters:
$$y = \text{base}_{FP8}(x) + \frac{\alpha}{\sqrt{r}} \cdot B\,(A\,x)$$
Since `x` is typically in bf16, the dequant branch creates a bf16 matrix `w` (numerically stable). Kaiming initialization for $A$, zero for $B$ → $\Delta W = 0$ at initialization (LoRA is "invisible" at startup). Smoke tests (Section 3.4.3) confirm gradients flow correctly into $B$ and the FP8 buffer does not leak gradients.

#### 2.3.3. Mixed Precision and Memory Optimization
- **FP8 (e4m3):** Base weights of DiT and text encoder → significant VRAM reduction.
- **BF16:** LoRA adapters and gradient computation → ensures numerical precision.
- **Gradient checkpointing:** `Ideogram4DiT` lacks built-in support, so it must be wrapped via `torch.utils.checkpoint` (enabled via `IDEOGRAM4_GRAD_CKPT=1` env var)—mandatory for A100 40GB.
- **8-bit AdamW** (bitsandbytes) for optimizer states.

### 2.4. Vietnamese Calligraphy - Visual and Linguistic Characteristics

#### 2.4.1. Cultural Context
Quốc ngữ calligraphy emerged in the late 20th century, combining the brush-holding techniques of Sino-Japanese calligraphy with the Latin alphabet and tonal diacritics—a unique art form with no precedent. Three core visual elements: stroke elegance, compositional balance, and rich ink texture.

#### 2.4.2. Geometric Characteristics of Brush Strokes
- **Thin strokes (thanh):** Brush parallel to the body, thin (1/5–1/10 of thick strokes).
- **Thick strokes (đậm):** Main vertical strokes, emphasizing brush pressure.
- **Dry brush (phi bạch):** Depleted ink creating white spaces within strokes—the highest frequency texture, most difficult to reproduce.

#### 2.4.3. Tonal System and Unicode Standard
Vietnamese has 6 tones × 12 vowel variants ≈ **134 character variants**. A crucial confirmation from Probe 1 (Section 3.4.2): the Qwen3-VL tokenizer handles Vietnamese diacritics stably (standard **NFC**, each diacritic is a distinct token, no byte-fallback). However, tokenizer stability *does not* mean the encoder retains diacritic discrimination in upper layers (see Section 2.2.3)—these are two distinct layers of the problem.

### 2.5. Overall System Architecture

```text
[Vietnamese JSON Prompt (NFC) + per-syllable bbox]
    → [Qwen3-VL-8B Encoder (FP8, FROZEN) → concat 13 layers = 53248 dim]
    → [llm_cond_proj 53248 → 4608]
                                      │ (text features)
[Gaussian Noise] → [DiT 34 layers Single-Stream MMDiT]
                     ↑ LoRA r=64 (Fp8LoRALinear) only at attention.{qkv,o}, BF16
    → [Sequence Latent (B, 4096, 128)]
    → [VAE Decoder]
    → [1024×1024 Calligraphy Image]
```
Framework: **DiffSynth-Studio** (manages model + `Ideogram4Pipeline`) + **HuggingFace Accelerate** (BF16 single-GPU). The LoRA training infrastructure (`Fp8LoRALinear` wrapper, injector, loss function, training loop) is *custom-built* for this thesis, as DiffSynth only provides the inference path.

---

## 3. IMPLEMENTATION AND EVALUATION

### 3.1. System Setup

#### 3.1.1. Hardware Configuration
| Component | Specifications |
|---|---|
| **GPU (Inference / Probe)** | NVIDIA A100 40GB (FP8, ~26GB VRAM during inference) |
| **GPU (Smoke / Training)** | NVIDIA RTX PRO 6000 Blackwell 96GB; A100 40/80GB |
| **VRAM Verification Test (grad ckpt OFF)** | ~64 GB |
| **Estimated Training VRAM (grad ckpt ON)** | ~25 GB → comfortable on A100 40GB |
| **Smoke Speed** | ~1.4 s/step (after warmup) → 14k steps ≈ 5.4 hours |
| **Inference Cost** | ~48 s/image @ 32 steps, cfg=7.0 |

*Crucial VRAM note:* The `Ideogram4Pipeline` defaults to loading *both* `transformer` and `unconditional_transformer` (~27.5 GB just for the two DiTs). To train LoRA-only on an A100 40GB, we **skip loading `unconditional_transformer`** and use CFG dropout during training (Section 3.3).

#### 3.1.2. Software Environment
| Library | Role |
|---|---|
| PyTorch | Core deep learning framework |
| DiffSynth-Studio (commit `6d103c0+`) | `Ideogram4Pipeline`, scheduler, VAE/encoder |
| HuggingFace Transformers | Qwen3-VL Tokenizer, model loading |
| HuggingFace Accelerate | Mixed precision, training loop |
| bitsandbytes | 8-bit AdamW |
| safetensors | Save/load LoRA-only checkpoints |
| rclone | Google Drive backup, recovery from interruptions |

#### 3.1.3. Anti-Outbid Workflow
Since training runs on cloud GPUs via a bidding mechanism (which can be "outbid" and lose the machine), a workflow saving optimizer states every 500 steps and periodically backing up via rclone (inherited from V9.2) is applied to enable warm-restarts.

### 3.2. Data and Preprocessing

#### 3.2.1. Data Source and Schema Conversion (Synthetic Data)
> **Data Scope (Crucial):** All training data is **synthetic** (machine-rendered from the "Thu Phap Thanh Cong" font, **not real handwritten calligraphy images**. This is a deliberate decision: the primary goal is to *succeed on font-generated data* (which is consistent, has accurate diacritic labels, and can be generated at scale). Using *real handwritten calligraphy* is left for future development (§4.4), partly because collecting a large enough, correctly-labeled, stylistically consistent real calligraphy dataset is extremely difficult. Thus, the "font style" the model learns is that of this *digitized font*, and the contribution lies in reproducing and controlling this style via diffusion generation—with stroke variance, ink effects, and layout—rather than mechanically copying font glyphs.

Specifically, the source data consists of **7,866 single-word images** from the V8.7 dataset (`dataset_v8_7_capital_phase_a`), rendered from the "Thu Phap Thanh Cong" font at 1024×1024, accompanied by pre-existing **diacritic masks**. The converter script `convert_v8_7_to_ideogram4_json.py` (280 LOC) generates `data/dataset_v10_phase_a/metadata.jsonl` (7,866/7,866 samples) following the **OFFICIAL v3.1 schema**.

Each record includes: `high_level_description`, `style_description` (4 sub-fields), `compositional_deconstruction.background`, and an `elements` array where each text element carries `bbox`, `text`, and `desc`. Two decisive factors in each element's `desc`:
1. **Anti-Chinese Character Clause** ("Vietnamese Latin characters, DO NOT use Chinese characters.") - Probe 6 confirms this is *causal*, not decorative.
2. **Font Name** ("Font: Thu Phap Thanh Cong.") - Probe §20 confirms this is the *key distinguishing factor*: without the font name, the model reverts to generating Chinese characters in a 2-line layout.

#### 3.2.2. Per-Sample Bounding Box
It was discovered that a fixed bbox `[200, 350, 824, 674]` (≈62% × 32% canvas) *does not match* the actual ink position of short words (font sizes 510–720 → characters occupy 60–70% of height). The converter therefore **replays the V8.7 generator's font math**: `_v87_optimal_font(text)` + `replay_v87_bbox(text)` to calculate a specific bbox for each sample, then normalizes it to [0, 1000].

Validation on 8 random samples: average IoU of **0.973** (range 0.912–0.992) between the replayed bbox and the actual ink region bbox. Across all 7,866 samples: **1,802 unique bbox values** (compared to 1 in the fixed version); width is ~69% of canvas (nearly constant due to auto-fit), while height varies 24–70% depending on word complexity. The model thus learns the correlation "shorter word → taller bbox".

#### 3.2.3. Compound Word Expansion (Phase 4)
Since V8.7 only contains single-word data, Phase 4 designs supplementary multi-syllable data: 2 syllables (names, short idioms), 3 syllables, 4 syllables (traditional idioms like "An Khang Thịnh Vượng"). Strict principles:
- **Per-syllable elements** (each syllable = one element + one bbox). Probe 3 proves: a 4-syllable phrase packed into *one* element is blocked by the safety filter (3/3), whereas splitting it into *four* elements renders successfully (3/3).
- Bboxes per syllable are calculated via font math based on layout (horizontal/vertical/grid 2×2).
- Diacritic mask = union of masks per syllable.
- Repeat the anti-Chinese clause + font name in *every* element (lesson from Probe §20).

#### 3.2.4. Preprocessing
- **Unicode NFC** for all strings (confirms Qwen3-VL tokenizer is consistent with NFC).
- **No image augmentation** - V8.7 is already deterministic single-font; augmentation would misalign with the v3.1 replayed bbox.
- Light *prompt* augmentation is possible (10% wording change for anti-Chinese, 10% change in `style_description`) to increase robustness—optional, deferred for v0.

### 3.3. FP8-LoRA Training Infrastructure and Configuration

#### 3.3.1. `Fp8LoRALinear` Wrapper and LoRA Injector
The technical core of V10 is the `Fp8LoRALinear` wrapper (Section 2.3.2). The `inject_lora_into_dit` injector traverses the DiT, finds `Fp8Linear` modules matching the `attention.qkv` / `attention.o` patterns, and replaces them with the wrapper. Two utility functions, `freeze_non_lora` and `collect_lora_params`, ensure only LoRA parameters are trainable. Checkpoints save **LoRA-only** (~115 MB safetensors).

#### 3.3.2. Target Strategy - DiT Self-Attention Only
Module listing (211 `Fp8Linear`, 13 patterns) leads to the targeting decision:

| Module Pattern | Count | LoRA Target? |
|---|---|---|
| `layers.*.attention.qkv` | 34 | **Yes** |
| `layers.*.attention.o` | 34 | **Yes** |
| `layers.*.feed_forward.{w1,w2,w3}` | 102 | No (FFN) |
| `layers.*.adaln_modulation` | 34 | No (timestep modulation) |
| `input_proj`, `llm_cond_proj`, `adaln_proj`, … | N/A | No (head/cond) |
| **Total Target** | **68** | - |

Decision basis: (a) **Probe 4** (text encoder loses diacritic discrimination at layer 9+ → injecting into text encoder is futile); (b) **render §20.4** (confusion of ơ/ọ/ộ appears in *pixel output* → diacritic discrimination must come from the DiT learning from the image side); (c) **skip FFN** (V9.2 also used this pattern; FFN-LoRA consumes many parameters for marginal gain); (d) **skip AdaLN** - relates only to timestep, not diacritic layers.

#### 3.3.3. Flow-Matching Loss with Diacritic Mask
Ideogram4's latent is a *sequence* `(B, 4096, 128)`, not an image grid—so the mask must also be flattened:
```python
mask_64  = F.interpolate(mask, size=(64, 64), mode='max')   # (B,1,64,64)
mask_flat = mask_64.flatten(2).transpose(1, 2)              # (B,4096,1)
weight    = mask_flat * factor + (1 - mask_flat) * bg_weight  # factor=10

pred   = -raw_out[:, max_text_tokens:]            # match negation + slice image tokens
target = scheduler.training_target(z, noise, t)   # = noise - z
loss   = (scheduler.training_weight(t) * (pred - target).pow(2) * weight).mean()
```
Three subtle points corrected via peer review: (i) **negating** the DiT output and **slicing correctly** to image tokens (`max_text_tokens:`); (ii) using the **scheduler's API** (`add_noise`, `training_target`, `training_weight`) to automatically apply Ideogram4-specific log-SNR shifting instead of manual calculation; (iii) diacritic mask weight ×10 (the diacritic/background ratio of ~10:1 was confirmed in smoke tests).

#### 3.3.4. Hyperparameter Table
| Hyperparameter | Value | Rationale |
|---|---|---|
| LoRA Rank (r) | **64** | Sweet spot inherited from V9.2 |
| LoRA Alpha (α) | **64** | rsLoRA scaling $\alpha/\sqrt{r}$ |
| LoRA Dropout | **0.0** | DiT input is already noisy |
| Target | **attention.{qkv, o}** | 68 modules, ~60.2M params (≈0.75%) |
| Base LR | **1×10⁻⁵** | Cosine decay, stable (V9.2) |
| Warmup Steps | **100** | Safe |
| Cosine min ratio | **0.10** | - |
| Batch Size | **1** | Pipeline fixes `text_z_padding` to batch=1 |
| Gradient Accumulation | **4** | Effective batch = 4 |
| Optimizer | **8-bit AdamW** | bitsandbytes |
| Diacritic factor | **10** | Diacritic mask weight |
| Precision | **BF16** (LoRA) / **FP8** (base) | - |
| Resolution | **1024×1024** | Native Ideogram4 |
| Max Steps | **14,000** | Comparable to V9.2 baseline for contrast |
| Save Checkpoint | **Every 500 steps** | Catch optimal point |
| Seed | **42** | - |
| CFG dropout (train) | **0** at v0 | Nega path = image-only string, not equivalent to zero llm_features → deferred |

#### 3.3.5. Phased Implementation Strategy
The infrastructure was built in 5 phased steps to isolate risk: (1) `Fp8LoRALinear` + inject/freeze/collect/save/load utilities; (2) full-DiT gradient smoke test with 1 real prompt; (3) dataset + masked loss + 5-step trainer; (4) 50-step trainer + scheduler API + grad ckpt; (5) full launcher + interruption handling. Implementation files (~750 LOC): `hybrid_peft_ideogram4.py`, `dataset_v10.py`, `diacritic_loss_v10.py`, `train_ideogram4_v10.py`, `infer_v10_lora.py`, and smoke scripts.

### 3.4. Results and Evaluation

#### 3.4.1. Phase 0: Raw Inference Validation (Pivot Justification)
Initial evaluation: running raw, unfine-tuned Ideogram4 on 6 difficult Vietnamese calligraphy cases with JSON prompts. **Notable result (prompt language dictates quality:** same model, same schema, same seed, only changing description fields from English to Vietnamese (with emphasis on "not Chinese characters") → *correct Latin Vietnamese character* render rate (character-level, no line splitting/Chinese characters) jumped from **~25% to ~83%**).

We must distinguish two levels of correctness: (a) *character-level correctness* (readable Latin Vietnamese, correct letters) and (b) *tone-exact correctness* (correct diacritics). On the 6 difficult cases, the raw model achieved a **very low character error rate (~1/6)** but only **2/6 were tone-exact**:

| # | Word | Diacritic Class | Render (V10 raw) | Character | Diacritic |
|---|---|---|---|---|---|
| 1 | **Vượng** | ư + horn + dot below | "Vượng" | Pass | Pass |
| 2 | **Hạnh** | a + dot below | "Hạnh" | Pass | Pass |
| 3 | Phúc | u + acute | "Phức" | Error u→ư (extra horn) | Pass acute |
| 4 | Tâm | â + level | "Tầm" | Pass | Error extra grave |
| 5 | Nhẫn | â + tilde | "Nhận" | Pass | Error tilde→dot below |
| 6 | Tã | a + tilde | "Tả" | Pass | Error tilde→hook above |

**Total: only 1 character error (Phúc→Phức), but tone-exact correctness is only 2/6 (≈33%, namely Vượng + Hạnh)** on raw images. Compared to V8.7 (which required 14,000 steps and completely failed on ư+horn/dot below cases), this is still a **significant character-level improvement** and justifies the model transition (*but diacritics remain the main unresolved bottleneck at the raw level*. The three diacritic errors share the same "fingerprint" as the diacritic bottleneck measured in the predecessor model, confirming the problem is *cross-architecture*. **Conclusion: model transition is justified at the character level; diacritics require a dedicated R-GDPO program - §3.5).**

#### 3.4.2. Phase 1: 11 Architectural Probes
A suite of 11 probes (176 image generations, ~3 hours GPU, peak 35 GB VRAM) extracted hard constraints:

- **Probe 1 - Tokenizer:** Qwen3-VL is NFC-compliant, each diacritic is a distinct token, no byte-fallback; full prompt ≈ 97–105 tokens, p95 ≈ 264, **well below** the 2048 budget.
- **Probe 2 - Schema:** Plain text prompts (non-JSON) are **100% blocked by safety filters - 8/8**; JSON wrapper is mandatory. Empty `style_description` is OK.
- **Probe 3 - Compound + bbox:** A 4-syllable phrase in *one* element → blocked 3/3; split *per syllable* - 4 elements + 4 bboxes → renders 3/3. **Conclusion: compounds must be per-syllable.**
- **Probe 4 - Hidden states (most critical finding):** Qwen3-VL **loses diacritic discrimination ability from layer 9+** (pairs differing only in diacritics have 1−cos < 0.001 in upper layers - layer 0: 0.42–0.55). VI vs EN prompt cos @layer 35 = 0.9455. → **LoRA target = DiT only.**
- **Probe 4-expand - Bbox matrix:** Bbox normalized [0, 1000]; optimal threshold ~62% canvas; **omitting bbox → 11/11 safety blocks** (for the *raw* model); bbox too narrow (22%) → 1/5 renders. **Conclusion: bbox is mandatory for the raw model.** Two additional corrections discovered later: (i) **`[y, x, y, x]` image-axis convention** (Section 2.2.4) (x/y swap errors don't show in single-word images (glyphs are nearly symmetric, so IoU 0.973 remains self-consistent) but only expose themselves in compound grid layouts); (ii) **after LoRA fine-tuning, the "mandatory bbox" constraint disappears** (the trained model no longer triggers the safety filter when bbox is missing, and a no-bbox prompt (one text element + newline `\n`) becomes the *default* inference mode as it is the only option without character errors. Training data retains bboxes for spatial supervision); only inference drops the bbox.
- **Probe 5 - Hyperparameters:** steps∈{24,32,48}, cfg∈{5,7,9}, seed∈{42,43,44} all OK; finalized at steps=32, cfg=7.0, seed=42.
- **Probe 6 - Anti-Chinese:** 4 wording variants all OK - 28/32 render; use the shortest wording.
- **Probe §19–§20 - OFFICIAL Schema + font name:** OFFICIAL schema vastly outperforms simple schema (1-element phrase: simple blocked, official renders; 2-line: 39KB vs 183KB content). **Font name is the distinguishing factor** Latin/Chinese: missing font → reverts to 4 Chinese characters - 煥熒青倉); with font → Vietnamese Latin characters. Note: raw render style is *English italic*, not yet Vietnamese brush → requires style learning during fine-tuning.

Synthesizing the 11 probes into **9 hard rules** for the converter (JSON mandatory, always emit bbox, compound per-syllable, per-sample bbox, finalized hyperparams, anti-Chinese clause + font name, accept ơ/ọ/ộ errors at raw level).

#### 3.4.3. Phase 2: FP8-LoRA Infrastructure Smoke Test
Three smoke test steps (on RTX PRO 6000 96GB) all PASS:

- **Step 1 - Isolated wrapper (6/6 PASS):** correctly injects **68 modules**, **60.2M LoRA parameters**; *identity at initialization* (max |Δ| = 0.00 due to $B$=0); after backward, $B$ has gradient 68/68 (while $A$ = 0/68 because chain rule is cut when $B$=0 - as expected); FP8 buffer does not leak gradients; LoRA-only checkpoint 114.8 MB reloads with max |Δ| = 0.
- **Step 2 - Full-DiT gradient on real input (6/7 PASS):** v3.1 prompt "Và" → 237 tokens; identity at initialization on real input; image token prediction has correct shape (1, 4096, 128); $B$ gradient 68/68. Discovered a **VAE encode bug** in DiffSynth (encoder outputs (1,64,128,128) but patchify is off by 256-dim) → workaround `vae_encode_mu_only` - take μ = first 32 channels, patchify 2×2 → correct 128 dim per `[μ | logσ²]` convention.
- **Step 3 - Minimal 5-step trainer (3/3 PASS):** end-to-end pipeline loop → inject → freeze → scheduler(training=True) → dataset → loss → optimizer; loss trajectory [0.089, 0.169, 0.158, 0.316, 0.156], no NaN/Inf; LoRA Δ ≈ 5×10⁻⁴ after 5 steps of AdamW@lr=1e-4 - correct order of magnitude); throughput 1.4 s/step → 14k steps ≈ 5.4 hours; diacritic/background loss ratio ~10:1 confirms factor=10 works.

**Phase 2 Conclusion: custom-built FP8-LoRA infrastructure runs correctly** - gradients flow into LoRA, no leakage into FP8 buffer, loss decreases, no divergence.

#### 3.4.4. Baseline 14k Training Results
LoRA training for 14,177 steps on 7,866 single-word images (RTX PRO 6000 96GB, 8-bit AdamW, LR=5×10⁻⁵, ~1.3 hours with corrected `[y, x, y, x]` bbox metadata). Key observations:

1. **Significant reduction in diacritic errors** compared to the raw model, but *tone-swapping errors persist* (no longer concentrated in one pattern but dispersed into "diacritic channel noise": adding extra diacritics to toneless words - Tâm → Tàm), confusing grave ↔ hook above, and collapsing tilde/hook above → dot below.
2. **Font style is acquired.** The checkpoint shifts from the raw model's English-italic style (§20.5) toward the "Thu Phap Thanh Cong" brush strokes - confirming LoRA learns style features.
3. **No-bbox becomes the default mode** (Section 3.4.2): the fine-tuned model no longer requires bboxes during inference.

The diacritic channel noise exposes a **flaw in the loss function**: the render-diff-based diacritic mask only covers regions *with* diacritics; for toneless words, the mask is empty → the region "that should be empty" above the vowel only receives background weight, so *hallucinating an extra diacritic* is barely penalized. This observation is the starting point for the R-GDPO program (§3.5).

#### 3.4.5. Comparison with Predecessor Baseline (V9.2)
| Criterion | V9.2 (ERNIE-Image) | V10 (Ideogram4) |
|---|---|---|
| Raw Vietnamese Latin render (no fine-tune) | Requires ~14k steps | **83% immediately when raw** |
| "Vượng" case (ư+horn+dot below) | Systemic failure | Near perfect (raw) |
| Syllable span alignment | `add_span` workaround | Native via per-syllable bbox |
| Text injection point | Single-layer dilemma (-2) | 13-layer concat, no choice needed |
| Diacritic bottleneck | Unresolved | **Resolved via R-GDPO: 57%→85%** (§3.5) |
| Training infrastructure | Pre-existing (LoRA script) | **Custom-built (FP8 LoRA + R-GDPO)** |

### 3.5. Resolving the Diacritic Bottleneck with Region-Grouped DPO (R-GDPO)

The core experimental focus of this version is shifting the diacritic discrimination bottleneck (previously identified as *cross-architecture and unresolved*) into a measurable and directly optimizable preference alignment problem.

#### 3.5.1. Micro Diacritic Classifier (Oracle) - Iterative Reinforcement
Every subsequent step (diagnosis, evaluation, DPO pair mining) requires an *oracle* to objectively evaluate diacritic correctness (since Vietnamese OCR like VietOCR, EasyOCR-vn has <80% accuracy and auto-corrects errors). The classifier is built on the **ANTIQA** [33] blueprint (~3M parameter CNN, 2-channel grayscale + Sobel input, strip-convolution for long strokes, multi-scale pooling, SE-gating), trained on synthetic font renders + brush style augmentation (ink-bleed, blur, paper texture - per DeepFont) to bridge the synthetic → brush domain gap.

Methodological note: the oracle was **iteratively reinforced over 6 versions**, each time human manual evaluation (the ultimate ground truth) caught an error type the oracle missed:

| Version | Added Capability | Author-discovered Error |
|---|---|---|
| v1–v2 | Classify 6 tones, 384px resolution | Short words (mark ~10px) lacked resolution |
| v3 | + *modifier* output (circumflex/breve/horn) | Vowel diacritic errors (ơ→ô, missing ư horn) missed by tone head |
| v4 | + multi-label (5-bit tone) | Stacked diacritics: Nhẫn drawn with both tilde *and* dot below |
| v5 | + manual stroke augmentation (crossed-marks) | Mistook font 'B' flourish for a grave accent |
| v6 | + decoy swash (long decorative stroke, empty label) | Distinguish "decorative stroke" from "diacritic" |

Oracle v6 achieves tone-acc/mod-acc ≈ 0.99 on synthetic sets and matches manual counting ±1 on real images. Lesson: **refusal to predict + visual inspection cannot be skipped in the DPO pipeline** - each model round finds a new error type the oracle hasn't covered.

An important evaluation improvement proposed by the author is **graded scoring**: 1.0 = perfect, **0.5 = correct tone *present* but with extra diacritics** (stacked diacritics containing the correct tone), 0 = wrong. The 0.5 score not only measures more accurately but also creates a *smoother gradient* for DPO (Section 3.5.4).

#### 3.5.2. Seed-Variance Diagnosis: Sampling Error vs. Weight Error
Before investing in training, a diagnostic probe distinguishes the error source: for each word in a 20-word difficult panel, generate 16 images at fixed seeds, score with the oracle, calculate *flip-rate*:
- flip-rate 10–90% (sometimes right, sometimes wrong) → **sampling error** (underdetermination) → fix at inference;
- wrong ≈100% across all seeds → **weight error** (learned deficiency) → fix during training.

Results on baseline 14k: 9 words stable-correct, 6 VARIABLE, 5 STABLE-WRONG; **~64% of error mass lies in STABLE-WRONG** → confirms training-side intervention is needed. Empirical confusion matrix: level→grave (30 times, largest hallucination error), X→dot below (47 times), hook above is the weakest class (41 errors). This is the system's *first objective confusion matrix* - replacing small-sample guesswork.

#### 3.5.3. Mask-SFT (Tone-Zone) Hits Ceiling - Theoretical Confirmation
The first intervention was a **tone-zone mask**: expand the mask to cover the *entire potential diacritic region* on all words (including toneless words, to penalize hallucination), reducing the weight from 10 down to 2–4 (per DreamVideo-2 evidence that small region weights >~4 cause instability). Results over two 4,000-step forks: **net accuracy unchanged (62% → 62%)** - hallucination on toneless words was fixed (Tâm 6%→94%) but errors *shifted to dot below* (X→dot below 47→81) and one fork broke the ư/ơ horn.

This result perfectly aligns with empirical Mask-SFT evaluations from **GlyphPrinter** [32]: *"due to lack of penalty for wrong glyphs, the model still tends to generate extra or missing strokes"*. Conclusion: MSE regression loss can only *reconstruct*, it cannot *teach discrimination* (preference gradient is needed). - A notable bug: initial masks stuck to the character body due to PIL advancing when inserting combining-marks; fixed by segment-by-segment rendering.

#### 3.5.4. R-GDPO Method and 5-Round Results
**R-GDPO** [32] (Region-Grouped DPO, CVPR 2026) optimizes DPO in the flow-matching velocity space, inserting the region mask *inside* the L2 norm of the velocity residual:

$\mathcal{L} = -\log\sigma\!\Big(\!-\beta\big[(\|M(v^w\!-\!v_\theta^w)\|^2\!-\!\|M(v^w\!-\!v_{ref}^w)\|^2) - (\|M(v^l\!-\!v_\theta^l)\|^2\!-\!\|M(v^l\!-\!v_{ref}^l)\|^2)\big]\Big) + \lambda_{anchor}\,\text{MSE}(v_\theta^w, v^w)$

Adapted implementation for FP8-LoRA: (i) *policy* = trained LoRA (init from baseline); (ii) *reference* = frozen snapshot of the LoRA init itself, realized via **parameter swapping (adapter-toggle)** so it costs no extra VRAM for a second model; (iii) same noise ε and timestep t for the winner/loser pair → common body gradient cancels out, forcing updates into the masked region; (iv) *winner-anchor* MSE term prevents winner degradation. Preference pairs are mined **on-policy** (winner/loser generated from the current checkpoint itself, per GlyphPrinter formula), with oracle v6 assigning labels with refusal-to-predict handling. Configuration: β=2, λ_anchor=1, 800 steps/round, selective gradient checkpointing K=13 - ~89GB/96GB.

Whole-word results on the 20-word difficult panel, 16 seeds/word (oracle v6, consistent measurement across all rounds):

| Milestone | Baseline | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|
| Whole-word (%) | 57 | 68 | 71 | 72 | 79 | **85** |
| Δ | - | +35 | +12 | +3 | +22 | +20 |

Round-by-round interpretation (all mechanisms verified via confusion matrix):
- **R1** (broad on-policy pairs): +35 points, Tâm 1→10, Bả 2→13.
- **R2** (*ground-truth winner* font render for 0-winner words like Nhẫn): **Nhẫn 0→11 (major breakthrough**; side effect Nhấn 10→2 due to *neighbor bleed* - pushing "draw tilde/hook above on â" also pulls acute in the same â cluster).
- **R3** (counterweight pairs for Nhấn): Nhấn recovers 2→4, Mở →7.
- **R4** (*two-tier mining* based on 0.5 score: full>loser + full>half "teach dropping extra diacritics" + half>loser "teach drawing correct tone"): **second major breakthrough +7 points**, Nhần escapes 0 for the first time, Nhấn 4→8, Tâm/Nhân →15.
- **R5** (first time *no ground-truth winner needed* (all words bootstrap winner on-policy): Nhần 1→9, Nhấn 8→15 - fully recovered), Tường fixes neighbor-bleed 10→12.

The trajectory shows R-GDPO is *not* monotonically diminishing: two-tier (R4) and on-policy bootstrap (R5) re-accelerate after the R3 plateau - evidence the method has remaining capacity.

#### 3.5.5. Generalization to Complex Vowels - Vowel-Specific Error Axes
To test if tone improvement generalizes, a stress panel of 6 complex vowel words (Xoắn, Nguyễn, Muỗng, Tiếng, Chuẩn, Hiểu) was generated on the R5 checkpoint. Clean finding: **the problem is not syllable complexity but *specific vowel context*:**

| Vowel | Word (tone) | Correct/16 |
|---|---|---|
| **ê** (circumflex) | Tiếng (acute), Nguyễn (tilde), Hiểu (hook) | 16, 15, 14 |
| **ô** | Muỗng (tilde) | 6 |
| **â** | Chuẩn (hook) | 5 |
| **ă** (breve) | Xoắn (acute) | 1–2 |

The **ê cluster is handled quite well** (the model generates correct accompanying diacritics in most cases, showing R-GDPO can generalize to common vowels). However, remaining vowels (â, ô, and ă) still show systematic errors - stable-wrong). Notably, the "â" vowel often errors when combined with complex diacritics as in Nhẫn/Nhần, a general model difficulty rather than an isolated word issue.

The research process shows that instead of focusing on helping the model "understand" how to write Vietnamese (which it already achieves basic quality on quite easily), the focus must be on **building efficient training infrastructure and improving correct diacritic generation through preference alignment**. Although accuracy has been raised from 57% to 85%, the system still has points requiring further mitigation.

### 3.6. Ongoing Mitigation Directions (Phase 3 Expansion & Phase 4)

Based on the analysis in Section 3.5.5, although accuracy has reached a respectable level (85%), the system is continuously being optimized within the thesis framework to thoroughly resolve remaining issues:

1. **Optimizing R-GDPO on weak vowel axes (Phase 3 expansion):** The R-GDPO method is being continuously applied to fix errors on "â", "ô", and "ă" vowels, especially cases with hook/grave combined with "â". This process includes expanding the evaluation panel focused on rare vowel contexts.
2. **Upgrading the Oracle for complex vowels:** The current classifier is being upgraded to not only verify diacritics but also validate base characters. This makes the DPO data mining process for complex vowel words more accurate and safe.
3. **Expanded compound word training (Phase 4):** Starting from the current best R-GDPO checkpoint, the system is undergoing continuous training on data containing multi-syllable phrases. The deployed strategy combines single-word and compound-word data in an increasing ratio (curriculum learning), while experimenting with placing easily confused compounds in the same image to create natural adversarial signals.

---

## 4. CONCLUSION

### 4.1. Summary of Theoretical and Practical Value

#### 4.1.1. Core Contributions
The core and most significant contribution of this research is the **successful construction of an automated Vietnamese calligraphy image generation system capable of font-specific style control, based on the Ideogram4 model**. This achievement was realized by solving two major technical problems: (1) building an FP8-LoRA fine-tuning infrastructure for DiT models, opening a path for tuning quantized models; and (2) largely resolving the diacritic generation bottleneck using Region-Grouped DPO (R-GDPO), elevating whole-word accuracy to 85%.

#### 4.1.2. Theoretical Contributions
The research provides a comprehensive empirical evaluation of the behavior of multimodal Diffusion Transformers (specifically Ideogram4 and Qwen3-VL) when processing Vietnamese. Theoretical contributions include specifying the degradation of diacritic discrimination in the text encoder's deeper layers, the decisive role of prompt language, the importance of structural constraints (like coordinates and font names), and demonstrating the necessity of preference optimization to overcome the limitations of traditional regression losses in text-containing image generation.

#### 4.1.3. Practical Contributions
1. A dataset of 7,866 samples standardized to OFFICIAL v3.1 + per-sample `[y,x,y,x]` bboxes (IoU 0.973).
2. ~750 LOC of custom FP8-LoRA training infrastructure + ~600 LOC of R-GDPO infrastructure (oracle, pair miner, DPO trainer), fully reproducible.
3. A suite of 11 probes + robustness report + objective diacritic confusion matrix, reusable for future research.
4. A custom micro diacritic classifier (oracle)—a non-OCR evaluation tool for generative Vietnamese calligraphy.
5. Compound word expansion design (Phase 4) with a continuous learning strategy.

### 4.2. Summary of Key Results

| Result | Value |
|---|
| Raw Ideogram4 (character-level accuracy) | ~5/6 (~83% readable) |
| Raw Ideogram4 (tone-exact accuracy) | 2/6 (~33%) - diacritics are the bottleneck |
| Quality jump due to prompt language | ~25% → ~83% (character-level) |
| Qwen3-VL tone-collapse | From layer 9+ (1−cos < 0.001) |
| LoRA target | 68 DiT-attention modules, ~60.2M (0.75%) |
| Bbox coordinate convention correction | `[y, x, y, x]` image-axis (not PIL) |
| Per-sample bbox IoU | 0.973 (1,802 unique values) |
| **Whole-word diacritic accuracy (R-GDPO 5 rounds)** | **57% → 85%** on 20-word difficult panel |
| Mask-SFT (regression) hits ceiling | 62% (errors shift to dot below) - preference gradient required to break ceiling |
| Diacritic Oracle | ~3M CNN, reinforced over 6 versions, matches manual count ±1 |
| Vowel generalization | ê resolved (16/15/14); â/ô/ă remain weak |

### 4.3. Current Limitations

In the current version and checkpoint, the research still has several limitations:
1. **Limited Evaluation Dataset:** The 85% accuracy is measured on a designed panel of 20 difficult words (crafted to test the most error-prone cases); there is no comprehensive Character Error Rate (CER) evaluation on a large, random test set. Furthermore, the current oracle focuses primarily on diacritic validation, so evaluation results for words with complex vowels may not reflect the absolute accuracy of the base characters.
2. **Time-Consuming Manual Evaluation:** The quality of the R-GDPO method heavily depends on the accuracy of the custom classifier. Whenever a new generation error type emerges, the classifier must be manually updated and re-evaluated, making the process labor-intensive.
3. **Temporary Technical Workarounds:** To handle the VAE encoding bug in DiffSynth, the research currently relies on a temporary workaround (`vae_encode_mu_only`).
4. **High Inference Latency:** Current generation speed is approximately 48 seconds per image (at 32 steps). Applying a multi-image generation and selection mechanism (choosing the best out of N samples) to increase the rate of perfect images will further increase computational costs.

### 4.4. Future Development Directions

Future research directions (post-thesis completion) may include:
1. **Building a Production Pipeline:** Optimizing the image generation and classification process (best-of-N sampling combined with early stopping) to ensure high-quality output, while experimenting with quantization or model distillation techniques to reduce inference time and integrate into a Web API.
2. **Diversifying Calligraphy Styles:** Integrating style loss functions to make brush stroke characteristics more natural, and expanding the model to support generation across multiple calligraphy fonts, rather than just the single font used currently.
3. **Utilizing Real Handwritten Calligraphy Data:** Once the model achieves high accuracy in character structuring from synthetic (machine font) data, a promising direction is to collect and fine-tune the model on a dataset of real handwritten calligraphy images. The biggest challenge here is building a dataset large enough, with detailed diacritic labeling and consistent stylistic quality.

In summary, this research has successfully executed the transition to the Ideogram4 model, **proposed an FP8-weighted DiT LoRA fine-tuning pipeline**, and most importantly, **provided an effective solution for Vietnamese diacritic generation** (improving accuracy from 57% to 85% using Region-Grouped DPO combined with a custom diacritic classifier). This result not only proves the feasibility of using preference optimization to resolve localized image generation errors (like diacritics), but also lays the groundwork for applying AI to the preservation and development of Vietnamese calligraphy art.

---

## REFERENCES

[1] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 27, 2014.

[2] X. Tian, "zi2zi: Master Chinese Calligraphy with Conditional Adversarial Networks," GitHub Repository, 2017. [Online]. Available: https://github.com/kaonashi-tyc/zi2zi

[3] Z. Lyu, X. Bai, B. Shi, and C. Yao, "CalliGAN: Style and Structure-aware Chinese Calligraphy Character Generator," in *Proc. IEEE/CVF CVPR Workshops*, 2020.

[4] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," in *Proc. IEEE ICCV*, 2017, pp. 2223–2232.

[5] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," in *NeurIPS*, vol. 33, 2020, pp. 6840–6851.

[6] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," in *ICLR*, 2021.

[7] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," in *Proc. IEEE/CVF CVPR*, 2022, pp. 10684–10695.

[8] J. Chen, Y. Huang, D. Lv, L. Cui, Q. Chen, and F. Wei, "TextDiffuser: Diffusion Models as Text Painters," in *NeurIPS*, 2023.

[9] Y. Yang, J. Wang, Z. Liu, Y. Guo, and L. Wei, "GlyphControl: Glyph Conditional Control for Visual Text Generation," in *NeurIPS*, 2023.

[10] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. ICML*, 2021, pp. 8748–8763.

[11] J. Li, D. Li, S. Savarese, and S. Hoi, "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," in *Proc. ICML*, 2023.

[12] Black Forest Labs, "FLUX.1," Technical Report, 2024. [Online]. Available: https://blackforestlabs.ai

[13] Alibaba Research, "Wan: Open and Advanced Large-Scale Video Generative Models," arXiv preprint, 2024.

[14] Ideogram AI, "Ideogram 4," Model card & weights (`ideogram-ai/ideogram-4-fp8`), 2026. [Online]. Available: https://www.modelscope.cn/models/ideogram-ai/ideogram-4-fp8

[15] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation," in *Proc. IEEE/CVF CVPR*, 2023, pp. 22500–22510.

[16] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *ICLR*, 2022.

[17] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, "Flow Matching for Generative Modeling," in *ICLR*, 2023.

[18] X. Liu, C. Gong, and Q. Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," in *ICLR*, 2023.

[19] ModelScope/Alibaba, "DiffSynth-Studio: A Diffusion Engine," GitHub Repository, 2024–2026. [Online]. Available: https://github.com/modelscope/DiffSynth-Studio

[20] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *NeurIPS*, vol. 35, 2022.

[21] P. Micikevicius et al., "FP8 Formats for Deep Learning," arXiv preprint arXiv:2209.05433, 2022.

[22] kohya-ss et al., "sd-scripts / rsLoRA: Rank-Stabilized LoRA scaling," GitHub Repository, 2024.

[23] M. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3)," in *Proc. ICML*, 2024.

[24] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," in *NeurIPS*, 2022.

[25] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization (AdamW)," in *ICLR*, 2019.

[26] T. Chen, B. Xu, C. Zhang, and C. Guestrin, "Training Deep Nets with Sublinear Memory Cost (Gradient Checkpointing)," arXiv preprint arXiv:1604.06174, 2016.

[27] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)," *IEEE Trans. Image Process.*, vol. 13, no. 4, pp. 600–612, 2004.

[28] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)," in *Proc. IEEE CVPR*, 2018.

[29] Qwen Team, "Qwen3-VL Technical Report," arXiv preprint, 2025–2026.

[30] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)," in *NeurIPS*, vol. 30, 2017.

[31] B. Wallace, M. Dang, R. Rafailov, et al., "Diffusion Model Alignment Using Direct Preference Optimization (Diffusion-DPO)," in *Proc. IEEE/CVF CVPR*, 2024.

[32] Shuai, Li, Ding, and Tao (Fudan Univ. & NTU), "GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering," in *Proc. IEEE/CVF CVPR* (Highlight), 2026. [Online]. Available: https://github.com/FudanCVL/GlyphPrinter

[33] (ANTIQA) "A Compact Multi-Scale Quality Assessor for AI-Generated Text Rendering," arXiv preprint, 2026.

[34] Fan, Zheng, Yeh, and Liu, "CFG-Zero*: Improved Classifier-Free Guidance via Zero-Init and Optimized Scale," arXiv preprint arXiv:2503.18886, 2025.

[35] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks (Temperature Scaling)," in *Proc. ICML*, 2017.

---

## APPENDICES

### APPENDIX A: Full Hyperparameter Configuration Script

Below is the exact argument parser configuration used for the core contribution of this thesis: the **R-GDPO (Region-Grouped Direct Preference Optimization)** training pipeline (`experiments/scripts/v10/train_rgdpo_v10.py`). This script fine-tunes the baseline LoRA checkpoint to resolve the diacritic bottleneck.

```python
# Core arguments from parse_args() in train_rgdpo_v10.py
# Usage example: python train_rgdpo_v10.py \
#     --init_lora_from experiments/checkpoints/v10_baseline_yxyx/step-14177.safetensors \
#     --max_steps 800 --save_steps 200 --lr 1e-5 --beta 2.0 --lambda_anchor 1.0 \
#     --output_dir experiments/checkpoints/v10_rgdpo_roundN

ap.add_argument("--pairs", type=Path, default=PROJECT_ROOT / "data" / "dpo_pairs_v10" / "pairs.jsonl")
ap.add_argument("--init_lora_from", type=Path, required=True, help="Baseline or previous R-GDPO checkpoint")
ap.add_argument("--output_dir", type=Path, required=True)
ap.add_argument("--max_steps", type=int, default=600)  # Typically 800 per round in practice
ap.add_argument("--save_steps", type=int, default=200)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--warmup_steps", type=int, default=30)
ap.add_argument("--beta", type=float, default=2.0, help="DPO temperature (per GlyphPrinter)")
ap.add_argument("--lambda_anchor", type=float, default=1.0, help="Winner-MSE anchor coefficient to prevent degradation")
ap.add_argument("--grad_accum", type=int, default=4)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--log_every", type=int, default=10)

# Note: Base LoRA config (rank=64, alpha=64, targets=["attention.qkv", "attention.o"]) 
# is hardcoded in the script's inject_lora_into_dit() call.
```

*(For reference, the exact verified hyperparameters used for the preceding Baseline FP8-LoRA training (`v10_baseline_yxyx/step-14177.safetensors`) were: `--max_steps 14177`, `--lr 1e-4`, `--diacritic_factor 10.0`, `--bg_weight 1.0`, and standard bf16 AdamW (`--use_8bit_adam` was **false**).)*

### APPENDIX B: Source Code of the Fp8LoRALinear Wrapper and Injector

Exact implementation of the `Fp8LoRALinear` wrapper enabling bf16 LoRA training on frozen FP8 base weights (`experiments/scripts/v10/hybrid_peft_ideogram4.py`):

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Fp8LoRALinear(nn.Module):
    """LoRA wrapper around an Fp8Linear base.
    Forward: y = base(x) + (alpha / sqrt(rank)) * (lora_B @ lora_A @ x)
    The base is kept as a submodule. Its FP8 weight + scale buffers stay
    frozen because they are not nn.Parameter. LoRA A/B are bf16 nn.Parameter.
    """
    def __init__(
        self,
        base: nn.Module,
        rank: int = 64,
        alpha: float = 64.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError(f"base must expose in_features/out_features, got {type(base)}")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        # rsLoRA scaling
        self.scaling = self.alpha / math.sqrt(self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Trainable bf16 factors
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=dtype))
        # Standard LoRA init: Kaiming A, zero B → identity at construction
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen FP8 dequant path (gradient cannot flow to buffer weight/scale)
        base_out = self.base(x)
        # LoRA delta. Cast factors to x.dtype to avoid mixed-precision linear errors.
        a_t = self.lora_A.to(x.dtype)
        b_t = self.lora_B.to(x.dtype)
        delta = F.linear(F.linear(self.dropout(x), a_t), b_t)
        return base_out + delta * self.scaling


def inject_lora_into_dit(
    dit: nn.Module,
    target_patterns: Iterable[str] = ("attention.qkv", "attention.o"),
    rank: int = 64,
    alpha: float = 64.0,
    dropout: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
) -> list[tuple[str, Fp8LoRALinear]]:
    """Walk DiT, wrap matched Fp8Linear modules in Fp8LoRALinear in place."""
    targets = tuple(target_patterns)
    to_replace = []
    for name, module in dit.named_modules():
        if not (type(module).__name__ == "Fp8Linear" and hasattr(module, "weight") and hasattr(module, "weight_scale")):
            continue
        if not any(p in name for p in targets):
            continue
        to_replace.append((name, name.rsplit(".", 1)[-1] if "." in name else name, module))

    replaced = []
    for full_name, _attr, base in to_replace:
        parent_path, _, attr = full_name.rpartition(".")
        parent = dit.get_submodule(parent_path) if parent_path else dit
        wrapper = Fp8LoRALinear(base=base, rank=rank, alpha=alpha, dropout=dropout, dtype=dtype)
        
        # Move wrapper to base's device (buffers, not parameters)
        device = next((buf.device for buf in base.buffers()), None) or next((p.device for p in base.parameters()), None)
        if device is not None:
            wrapper = wrapper.to(device)
            
        setattr(parent, attr, wrapper)
        replaced.append((full_name, wrapper))
    return replaced
```

### APPENDIX C: Detailed Evaluation Samples from R-GDPO Checkpoints

Whole-word accuracy progression on the 20-word difficult panel (16 seeds/word, Oracle v6):

| Word | Baseline (14k) | R-GDPO R1 | R-GDPO R2 | R-GDPO R3 | R-GDPO R4 | R-GDPO R5 |
|---|---|---|---|---|---|---|
| **Tâm** (â + level) | 1/16 (6%) | 10/16 (62%) | 11/16 (68%) | 12/16 (75%) | 15/16 (93%) | 15/16 (93%) |
| **Nhẫn** (â + tilde) | 0/16 (0%) | 2/16 (12%) | 11/16 (68%) | 10/16 (62%) | 14/16 (87%) | 15/16 (93%) |
| **Nhấn** (â + dot below)| 10/16 (62%)| 8/16 (50%) | 2/16 (12%) | 4/16 (25%) | 8/16 (50%) | 15/16 (93%) |
| **Bả** (a + hook) | 2/16 (12%) | 13/16 (81%) | 14/16 (87%) | 14/16 (87%) | 15/16 (93%) | 16/16 (100%)|
| **Tường** (ư + level) | 8/16 (50%) | 10/16 (62%) | 11/16 (68%) | 12/16 (75%) | 13/16 (81%) | 15/16 (93%) |
| **Panel Average** | **57%** | **68%** | **71%** | **72%** | **79%** | **85%** |

*Note: The dip in R2/R3 for "Nhấn" was an expected side effect of neighbor-bleed during aggressive tilde/hook optimization, which was successfully corrected by the counterweight pairs in R3 and two-tier mining in R4.*

### APPENDIX D: Micro Diacritic Classifier (Oracle) Architecture and Training Details

The Oracle is a custom ~3.16M parameter CNN designed to evaluate diacritic correctness independently of standard OCR tools (which suffer from <80% accuracy and auto-correction on stylized text).

**Architecture (ANTIQA-inspired):**
- **Input:** 2-channel image tensor (Grayscale + Sobel edge map), resized to 384×384px.
- **Backbone:** 4-stage CNN with strip-convolution kernels `(1×7)` and `(7×1)` optimized for long, thin brush strokes.
- **Attention:** Squeeze-and-Excitation (SE) gating after each convolutional block to emphasize diacritic regions over the main character body.
- **Pooling:** Multi-scale global pooling from stage 3 and stage 4 to capture diacritics of varying sizes (e.g., small dot below vs. large tilde).
- **Head:** Multi-label classification head outputting:
  - 5-bit tone vector (acute, grave, hook, tilde, dot below; level tone = all-zero).
  - 4-bit modifier vector (circumflex, breve, horn1, horn2).

**Training Regimen:**
- **Dataset:** On-the-fly synthetic syllable renders from `vietnamesesyllable_7184_capital_clean.txt`, class-balanced across the six Vietnamese tones. The default training run uses `STEPS=8000` and `BATCH=64`, producing 512,000 synthetic training renders, with a fixed-seed validation set of 1,200 renders.
- **Fonts:** Samples are rendered using a random font selected from seven fonts in `experiments/gen_dataset/fonts`, including `Thu_Phap_Thanh_Cong_Unicode.ttf`, `HL-Latre_uni.ttf`, `HL-NetbutlongUni.ttf`, `HL-Netco1Uni.ttf`, `HL-OngDo-Unicode.ttf`, `HL-Thanhcao-unicode.ttf`, and `HL-Thufap2-unicode.ttf`.
- **Augmentation:** The pipeline applies DeepFont-style domain augmentation, including slight rotation, ink-bleed/thinning via morphological filters, Gaussian blur, paper texture noise, Gaussian noise, gamma/contrast jitter, and JPEG artifacts. It also includes stacked-tone marks, procedural crossed-marks, and decoy-swash augmentations to harden the oracle against stacked diacritics, font flourishes, and false tone marks.
- **Loss Function:** Binary Cross-Entropy with Logits (BCEWithLogitsLoss), summed over the 5-bit tone head and 4-bit modifier head.
- **Optimizer:** AdamW, LR=3×10⁻⁴, weight decay=1×10⁻⁴, Cosine Annealing with `eta_min = LR × 0.05`, 8,000 default steps.
- **Performance:** The default run records `tone_acc ≈ 85.83%` and `modifier_acc ≈ 95.17%` on the fixed synthetic validation set; the hardened v6 oracle was additionally spot-checked against manual labels on generated calligraphy images, with the session log reporting manual-count agreement within ±1.

---

*Version: V10 (Ideogram4 model transition) (updated Chapter 3.5 - R-GDPO resolves diacritic bottleneck, 57%→85%).*
*Draft Date: 2026-06-09. Updated: 2026-06-13.*