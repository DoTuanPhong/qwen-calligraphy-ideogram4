MINISTRY OF EDUCATION AND TRAINING

FPT UNIVERSITY

Fine-tuning Qwen Image for Generating Vietnamese Calligraphy Images with Accurate Diacritics

Author

Đỗ Tuấn Phong

A thesis submitted in partial fulfillment of the requirements for the degree of Master of Software Engineering

Supervisor:

Dr. Nguyễn Bích Thủy

© Copyright by Đỗ Tuấn Phong 2026

---

# Fine-tuning Qwen Image for Generating Vietnamese Calligraphy Images with Accurate Diacritics

Đỗ Tuấn Phong

Program: Master of Software Engineering

Specialization: Artificial Intelligence

FPT University

2026

---

## Abstract

Quốc ngữ calligraphy is a distinctive Vietnamese art form in which the Latin-based Vietnamese script is rendered using traditional calligraphic brushwork. While modern image generation models can produce aesthetically rich images, they still struggle significantly with accurately rendering Vietnamese, because Vietnamese relies heavily on tone marks and vowel diacritics. A generated image may look visually appealing yet fail semantically if it renders `Cữu` as `Cưu`, `Chưởng` as `Chưỡng`, or `Gẫy` as `Gậy`. In Vietnamese calligraphy, these marks are not decorative details; they are linguistically mandatory components that must simultaneously integrate with brush-stroke geometry.

This thesis was officially registered under the topic of fine-tuning Qwen Image, and initially followed a Qwen-Image-based implementation path. During the research, the final implementation shifted to Ideogram4, a Diffusion Transformer image generation model with stronger text rendering capabilities. This decision was driven by two empirical reasons. First, Qwen Image required excessive VRAM for the iterative local experimentation cycle. Second, the zero-shot Vietnamese rendering capability of Qwen Image was too weak, often making errors at the character level before the more subtle tone-mark problem could even be isolated. ERNIE Image was considered as an intermediate direction and was generally better than Qwen Image: it typically preserved approximately the correct number of characters and occasionally generated some Vietnamese phrases correctly. However, ERNIE Image still had many character-level errors, and notes from the earlier ERNIE Image phase showed that the Mistral3/Ministral3-based text encoding pipeline with byte-level BPE made it very difficult to align Vietnamese syllables with tokens. Therefore, Ideogram4 was selected as the final experimental backbone.

The central contribution of this thesis is an Ideogram4 DiT-LoRA implementation for the registered goal of fine-tuning Vietnamese calligraphy image generation, organized as a glyph-binding-oriented pipeline to improve Vietnamese diacritic accuracy. This pipeline diagnoses where diacritic signal persists or degrades, fine-tunes selected DiT modules via LoRA, stabilizes high-variance checkpoints, and ultimately trains directly on multi-word layouts to move closer to the goal of long-sentence generation.

Earlier attention-only LoRA experiments plateaued at approximately 32–39 correct words out of 60 on the difficult single-word panel. Hidden-state and projection probes revealed that Qwen3-VL is not globally blind to Vietnamese diacritics; the diacritic signal persists in the conditioning. The primary bottleneck lies in the DiT not consistently binding that weak diacritic signal to the correct glyph geometry during image generation. Based on this diagnosis, the thesis expanded the LoRA target from attention-only to a broader set of DiT modules: `attention.qkv`, `attention.o`, `feed_forward.w1`, `feed_forward.w2`, `feed_forward.w3`, and `adaln_modulation`.

This wide-target configuration broke through the single-word plateau, achieving 48/60 at the best single checkpoint and 52/60 after averaging compatible LoRA checkpoints. However, single-word results did not automatically transfer well to multi-word images. The best single-word checkpoint still produced many errors when writing two-line calligraphy layouts. Therefore, the thesis built a compound dataset consisting of 4/5/7/8-word images, split into two center-aligned lines, rendered from the target calligraphy font and paired with bounding-box-free Ideogram4 prompts. This dataset covers all 406 Vietnamese diacritical token IDs listed at the tokenizer level across both uppercase and lowercase.

On the compound Eval28 panel comprising 28 images and a total of 168 words, the single-word gold checkpoint `soup567` produced 56 errors. After compound bridge fine-tuning and compatible checkpoint averaging, the `soup_e4r2r3r4` milestone reduced errors to 6/168. A follow-up branch from this milestone with learning rate `3e-5`, then lightly averaged at a 90% new branch + 10% old milestone ratio, further reduced errors to 4/168, corresponding to approximately 97.6% word-level accuracy on this panel. The results demonstrate that the problem of generating accurate Vietnamese calligraphy requires not only a strong text-to-image base model, but also correct diagnosis of where the linguistic signal resides, proper adjustment of DiT modules, checkpoint stabilization, and direct training on the multi-word layout distribution.

**Keywords:** Vietnamese calligraphy, Qwen Image, Ideogram4, fine-tuning, LoRA, Diffusion Transformer, text-to-image generation, Vietnamese diacritics, glyph binding.

---

## Acknowledgements

I would like to express my deepest gratitude to Dr. Nguyễn Bích Thủy, my supervisor, for her guidance, feedback, and patience throughout the research process. The problem in this thesis changed many times: from initial Qwen-Image experiments, to ERNIE Image, then to Ideogram4, from checking individual Vietnamese diacritic errors to diagnosing conditioning signal and finally training on multi-word layouts. Her feedback helped the research maintain a clear direction when experimental results were noisy and when several appealing hypotheses had to be discarded based on evidence.

I also thank FPT University and the MSE-AI program for enabling me to pursue a topic that combines artificial intelligence techniques with Vietnamese cultural heritage. This is a problem with both technical significance and practical applications in design, education, and the preservation of Quốc ngữ calligraphy.

I am grateful to the open-source communities and engineering teams that developed the tools used in this research, including Ideogram AI, the Qwen team, DiffSynth-Studio, PyTorch, HuggingFace Transformers, safetensors, and the Python machine learning ecosystem. The experiments in this thesis relied heavily on reproducible scripts, checkpoint management, LoRA conversion, and GPU-based evaluation.

Finally, I thank my family and friends for their encouragement and patience. Many important results of this thesis came from manually evaluating each calligraphy image — a time-consuming but necessary process to ensure the reliability of the conclusions.

---

# Table of Contents

Acknowledgements

List of Tables

List of Figures

List of Appendices

1. Introduction

1.1. Problem Statement

1.2. Research Objectives and Scope

1.2.1. Scope Change from Qwen-Image to Ideogram4

1.2.2. Research Objectives and Boundaries

1.3. Process Overview and Domain Challenges

1.3.1. Overall Research Pipeline

1.3.2. Domain-Specific Challenges

1.4. Literature Review

1.4.1. GAN-Based Methods

1.4.2. Diffusion Models

1.4.3. Diffusion Transformers and Multimodal Text Encoders

1.4.4. Parameter-Efficient Fine-Tuning

1.4.5. Commercial Models and Research Gap

1.5. Proposed Method and Contributions

1.6. Thesis Structure

2. Theoretical Foundations

2.1. AI Image Generation: From GANs to Diffusion Transformers

2.2. Ideogram4 Architecture

2.3. Parameter-Efficient Fine-Tuning Techniques

2.4. Vietnamese Calligraphy: Visual and Linguistic Characteristics

3. Implementation and Evaluation

3.1. System Setup

3.2. Data and Preprocessing

3.3. Diagnostic Probes

3.4. Fine-Tuning Configuration

3.5. Evaluation Protocol

3.6. Results

4. Conclusion

4.1. Theoretical and Practical Value

4.2. Summary of Key Results

4.3. Current Limitations

4.4. Future Directions

References

Appendices

---

# List of Tables

**Table 1.1:** Comparison of competitors and baselines for Vietnamese calligraphy image generation

**Table 2.1:** Overview of the three main components in the Ideogram4 pipeline

**Table 3.1:** Hardware configuration

**Table 3.2:** Software environment

**Table 3.3:** Qwen3-VL signal probe results

**Table 3.4:** LoRA module groups in the DiT

**Table 3.5:** Single-word panel results

**Table 3.6:** Compound Eval28 result progression

**Table H.1:** Comparison image list and figure readiness status

**Table J.1:** Reference and evidence checklist

---

# List of Figures

**Figure 1.1:** Overall research pipeline for Vietnamese calligraphy image generation
Expected filename: `docs/thesis/figures/fig_1_1_research_pipeline.png`

**Figure 1.2:** Visual comparison of competitor methods and the proposed checkpoint
Expected filename: `docs/thesis/figures/fig_1_2_competitor_baseline_comparison.png`

**Figure 1.3:** Comparison of base capabilities of Qwen Image, ERNIE Image, and Ideogram4 before fine-tuning
Expected filename: `docs/thesis/figures/fig_1_3_base_model_capability_comparison.png`

**Figure 2.1:** Ideogram4 architecture used in this thesis
Expected filename: `docs/thesis/figures/fig_2_1_ideogram4_architecture.png`

**Figure 2.2:** Multi-layer Qwen3-VL conditioning fed into the DiT
Expected filename: `docs/thesis/figures/fig_2_2_qwen3vl_multilayer_conditioning.png`

**Figure 3.1:** Wide-target LoRA insertion points in the Ideogram4 DiT
Expected filename: `docs/thesis/figures/fig_3_1_widetarget_lora_injection_points.png`

**Figure 3.2:** No-bounding-box and bounding-box layout comparison for compound prompts
Expected filename: `docs/thesis/figures/fig_3_2_no_bbox_vs_bbox_layout_comparison.png`

**Figure 3.3:** Font-rendered reference versus model-generated compound layout
Expected filename: `docs/thesis/figures/fig_3_3_font_reference_vs_model_generation.jpg`

**Figure 3.4:** Result progression from the single-word plateau to the final compound checkpoint
Expected filename: `docs/thesis/figures/fig_3_4_result_progression.png`

**Figure 3.5:** Before/after examples on difficult Vietnamese diacritical words
Expected filename: `docs/thesis/figures/fig_3_5_single_word_before_after_examples.png`

**Figure 3.6:** Compound Eval28 before/after comparison between `soup567` and the final checkpoint `soup_lr3e5_gold4_9to1`
Expected filename: `docs/thesis/figures/fig_3_6_compound_eval28_before_after.png`

**Figure 3.7:** Remaining errors of the current compound gold checkpoint
Expected filename: `docs/thesis/figures/fig_3_7_remaining_error_cases.png`

**Figure 3.8:** Testing the fine-tuned checkpoint for combining calligraphy with Ideogram4's base image generation capability
Expected filename: `docs/thesis/figures/fig_3_8_calligraphy_with_base_model_capability.png`

---

# List of Appendices

**Appendix A:** Main wide-target training command

**Appendix B:** Compound data generation command

**Appendix C:** Checkpoint post-processing and averaging script

**Appendix D:** Checkpoint registry

**Appendix E:** Evaluation directories

**Appendix F:** Remaining errors of the current compound gold checkpoint

**Appendix G:** Internal probe reports

**Appendix H:** Planned comparison image list

**Appendix I:** Prompts used for comparison figures

**Appendix J:** Reference and evidence checklist

---

# 1. Introduction

## 1.1. Problem Statement

Quốc ngữ calligraphy is the art of rendering the Latin-based Vietnamese script using calligraphic brushwork. Unlike standard Latin text, Vietnamese has a dense system of tone marks and diacritics. A very small change in a mark can transform one word into an entirely different word. For example, `Cưu`, `Cừu`, `Cứu`, `Cửu`, `Cữu`, and `Cựu` have visually similar forms but carry distinct meanings.

In calligraphy, the problem is further complicated because marks must not only be orthographically correct. They must integrate into the overall brushwork, maintaining rhythm, stroke thickness variation, slant, and the handcrafted feel. A nặng dot must not resemble a random ink blot; a ngã tilde must not degenerate into a hỏi hook; circumflex and tone marks must be correctly positioned yet remain natural within the stroke layout.

Modern text-to-image models can produce visually appealing calligraphy images, and recent systems increasingly use large multimodal encoders and Diffusion Transformer-style architectures [15], [17], [21–22]. However, commercial image generation models typically do not allow fine-tuning on a specific font, do not guarantee reproducibility, and do not provide fine-grained control over individual Vietnamese diacritics. Conversely, direct rendering from a digital font yields orthographically correct results but lacks brushstroke liveliness: there is no ink variation, dryness, pressure modulation, or natural stroke interaction. The research gap addressed by this thesis lies between these two extremes: learning a specific calligraphic style while preserving accurate Vietnamese diacritics.

The topic was initially registered under the Qwen Image direction. However, during the research, Qwen Image revealed two major limitations: excessively high VRAM requirements for the local iteration cycle and weak initial Vietnamese rendering, often making errors at the character level. ERNIE Image was considered as an alternative and was better than Qwen Image in several respects: it typically preserved approximately the correct number of characters and occasionally generated some Vietnamese phrases correctly. Nevertheless, ERNIE Image still produced many character-level errors; moreover, notes from the earlier ERNIE Image phase showed that the Mistral3/Ministral3 tokenizer/encoding pipeline with byte-level BPE made syllable-level diacritic alignment difficult. These observations are treated as internal experimental evidence and are summarized in Appendix J, while the corresponding model families are documented in [17, 21].

Therefore, the final implementation of this thesis shifted to Ideogram4, whose open technical description reports a strong text-rendering foundation and a Qwen3-VL-based conditioning interface [15, 16]. This shift did not change the research objective: it remained fine-tuning a modern image generation model to produce Vietnamese calligraphy with accurate diacritics. What changed was that the experimental backbone was selected based on evidence rather than insisting on a less suitable architecture.

## 1.2. Research Objectives and Scope

### 1.2.1. Scope Change from Qwen-Image to Ideogram4

The official thesis title retains the term "Qwen Image" because this was the registered name when the initial research direction focused on Qwen-Image. In this thesis, the title is understood as the administrative title and the title of the research objective: fine-tuning a text-to-image model to generate Vietnamese calligraphy images with accurate diacritics.

During experimentation, the implementation scope changed. Qwen Image was not favorable for the research cycle due to VRAM costs and a weak Vietnamese baseline. ERNIE Image showed some more promising signals than Qwen Image but was still limited by character-level errors and byte-level BPE tokenizer issues. Ideogram4 was selected because it has a stronger text rendering foundation and is more practical for local LoRA training [15]. The Ideogram4 pipeline also uses Qwen3-VL-based text conditioning [16], so it remains directly related to the initial research question about Qwen-family conditioning signal.

Therefore, the experimental chapters of this thesis report the final validated implementation: the Ideogram4 DiT-LoRA pipeline for generating Vietnamese calligraphy images with accurate diacritics.

### 1.2.2. Research Objectives and Boundaries

The primary objective of this thesis is to build a fine-tuning pipeline for generating Vietnamese calligraphy images with accurate diacritics. The final implementation uses Ideogram4 and LoRA within the scope of the registered Qwen Image topic [8, 15].

At a more specific level, the research analyzes the error patterns of Ideogram4 when generating Vietnamese calligraphy images with diacritics, then determines whether errors primarily originate from the Qwen3-VL text encoder, the prompt, insufficient LoRA capacity, or the DiT's glyph-binding behavior. From this diagnosis, the thesis designs a LoRA configuration to improve Vietnamese diacritic rendering without full fine-tuning, builds data and evaluation panels for both single-word and multi-word layouts, and examines whether single-word improvements transfer to compound layouts and realistic sentences. The final output of the research is not merely a checkpoint, but a reproducible pipeline for training, checkpoint conversion, evaluation rendering, and manual scoring.

The experimental scope is deliberately limited for interpretability. The final implementation uses Ideogram4 as the base model and keeps the Qwen3-VL text encoder frozen. Fine-tuning uses LoRA adapters on the DiT backbone, with Thu Phap Thanh Cong Unicode as the primary target style font and 1024×1024 resolution. Evaluation focuses on word-level accuracy verified by human inspection, because OCR is not yet reliable enough for stylized Vietnamese calligraphy. Content includes single Vietnamese words and compound 4/5/7/8-word images split into two lines.

This thesis does not claim to solve all calligraphic styles or all forms of Vietnamese text rendering. The focus is an experimental pipeline for one specific calligraphy font, with the primary goal being Vietnamese diacritic accuracy.

## 1.3. Process Overview and Domain Challenges

### 1.3.1. Overall Research Pipeline

The final research pipeline begins by observing the Ideogram4 baseline on Vietnamese calligraphy prompts to identify common error patterns. The research then moves to signal diagnosis using hidden-state probes, token-span probes, and projection probes to check whether Vietnamese diacritic signal persists in Qwen3-VL and reaches the DiT input [15, 16]. Once the bottleneck is clearer, LoRA is trained on difficult Vietnamese words [8]. The attention-only setup is tried first, but after reaching a plateau, the LoRA target is expanded to a broader set of DiT modules.

After the single-word phase, the pipeline addresses checkpoint instability. Effective LoRA checkpoints are averaged if they are compatible and appear to lie in the same optimization basin, following the general idea of weight averaging and model soups [13, 14]. This stabilizes results without increasing inference cost. The final phase transitions from single-word to compound layouts: multi-word two-line data is generated from the target calligraphy font, the best single-word checkpoint is used as the starting point, and the model is evaluated on a fixed compound panel.

<!-- Figure placeholder: copy the final pipeline diagram to docs/thesis/figures/fig_1_1_research_pipeline.png -->
![Figure 1.1. Overall research pipeline for Vietnamese calligraphy image generation.](figures/fig_1_1_research_pipeline.png)

**Figure 1.1.** Overall research pipeline for Vietnamese calligraphy image generation. The pipeline begins with baseline observation, proceeds through Qwen3-VL/DiT signal diagnosis and wide-target LoRA training, then stabilizes compatible checkpoints through averaging before moving to compound-layout training and fixed-seed evaluation.

### 1.3.2. Domain-Specific Challenges

The first challenge comes from the Vietnamese writing system itself. Vietnamese has six tones and numerous vowel variants such as `â`, `ă`, `ê`, `ô`, `ơ`, `ư`; a very small mark error can completely corrupt a word. In calligraphy, marks must also be integrated into the brushwork, not merely positioned correctly as in a printed font. If marks are too detached, the image loses its calligraphic quality; if marks blend too heavily into strokes, readers may misinterpret them.

The second challenge lies in fine-tuning and layout. Attention-only LoRA improves performance to some extent but quickly plateaus at 32–39/60, indicating that adjusting attention alone is insufficient to correct glyph geometry. Moreover, the same training recipe can produce either a good checkpoint or a severe collapse; for example, one warm-continue round suffered from many extra nặng dot errors. When transitioning to multi-word images, difficulties increase further because visual text rendering systems must simultaneously handle layout, spacing, character identity, and local glyph geometry [10], [24–26].

## 1.4. Literature Review

### 1.4.1. GAN-Based Methods

GANs train a generator and discriminator in an adversarial manner [1]. In calligraphy tasks, GANs have been used to convert printed glyphs into calligraphic style [2, 3]. However, GANs are prone to mode collapse and have difficulty ensuring full coverage of Vietnamese diacritic combinations.

### 1.4.2. Diffusion Models

Diffusion models generate images through iterative denoising [4, 5]. Compared to GANs, diffusion models are more stable and produce more diverse images. Latent Diffusion further reduces computational cost by denoising in latent space rather than pixel space, making high-resolution image generation more feasible [6].

### 1.4.3. Diffusion Transformers and Multimodal Text Encoders

Diffusion Transformers replace U-Net with transformer blocks in the denoising network [7]. Transformers facilitate the integration of text tokens and image tokens, but accurate text rendering remains challenging because the model must convert symbolic signals into spatial geometry [10], [24–26]. For Vietnamese, the important signals are often very small yet linguistically significant.

Models such as Qwen-Image, ERNIE Image, FLUX, and Ideogram4 represent the trend of using DiT and powerful text encoders [15], [17], [21–22]. However, each model has different bottlenecks. Qwen Image is VRAM-heavy and has a weak Vietnamese baseline; ERNIE Image is better than Qwen Image but faces alignment risks due to byte-level BPE; Ideogram4 provides the most practical foundation for this pipeline.

Common architectural observations across the three open-source approaches surveyed. All three models in the comparison table belong to the Diffusion Transformer-based text-to-image family but make notably different architectural choices:

- **Qwen-Image** (Qwen Team, 20B params, MMDiT, Apache 2.0, arXiv 2508.02324) uses the Qwen2.5-VL text encoder and achieves state-of-the-art on Chinese text rendering [21]. However, zero-shot experiments in this thesis show very limited visual knowledge of diacritical Vietnamese: errors occur at the character level (wrong vowels, missing consonants), at the word level (inconsistent renderings across seeds, character-count variance), and at the tone-mark level (marks omitted, misplaced, or spurious). Combined with its high VRAM cost, this is why Qwen-Image was not retained as the experimental backbone.

- **ERNIE-Image** (Baidu) combines a Mistral3/Ministral3 text encoder with byte-level BPE [17]. Byte-level BPE is robust for many languages, but it destabilizes Vietnamese syllable alignment — capitalized diacritical forms such as `Ở`, `Ảnh`, `Ý`, `Ước`, `Nguyễn` may decompose into raw bytes and decode incorrectly. Although ERNIE Image preserves character count better than Qwen Image, the tokenizer instability makes it unsuitable for a diacritic-accurate fine-tuning target.

- **Ideogram4** (Ideogram AI, 9.3B params, published 2026-06-03) is Ideogram's first open-source text-to-image model, trained from scratch [15]. It uses a fully single-stream Diffusion Transformer (34 layers) with concatenated text/image tokens processed by the same transformer — no separate text/image branch. The text encoder is Qwen3-VL-8B-Instruct, with hidden states extracted from 13 intermediate layers [16]. It uses flow-matching [19–20] rather than pure DDPM, supports native resolution from 256 to 2048, and the FP8 base + BF16 LoRA design enables local fine-tuning. With 9.3B parameters, Ideogram4 achieves the best text rendering quality among open-weight models surveyed (surpassing Qwen-Image 20B, FLUX.2 32B, HunyuanImage 3.0 80B MoE). This is why Ideogram4 was selected as the final experimental backbone.

The most notable commonality: all three models use Qwen3-VL/Mistral3-family (or equivalent) text encoders as the conditioning source, meaning the thesis's initial research question about *linguistic signal in the Qwen family* remains valid regardless of which image generation backbone is chosen.

### 1.4.4. Parameter-Efficient Fine-Tuning

Full fine-tuning of large text-to-image models is expensive and risks degrading base model capabilities. LoRA learns low-rank updates on selected modules while keeping the original weights frozen [8]. This is a suitable choice for learning a specific calligraphic style while preserving the base model's image generation capabilities and visual priors.

Earlier text-to-image adaptation work such as DreamBooth shows that targeted fine-tuning can specialize a generative model to a desired subject or concept [9]. At the same time, the broader progress of image-text representation learning, including CLIP and BLIP-2, highlights how strongly image generation depends on the quality of text-image conditioning [11, 12]. For LoRA itself, scaling and rank behavior are important practical concerns, motivating careful control over adapter rank and update strength [27].

This research demonstrates that LoRA target module selection is critical. Attention-only LoRA is insufficient to overcome the Vietnamese diacritic plateau. Expansion to feed-forward and adaLN modulation is needed to more directly influence glyph geometry.

### 1.4.5. Commercial Models and Research Gap

Commercial models can produce beautiful calligraphy images but typically do not allow control over font, private data, checkpoints, seeds, or evaluation pipelines. In the previously approved thesis version, tools such as Nano Banana 2 and GPT Image 1.5 High were used as commercial comparison systems: they can produce visually appealing Vietnamese calligraphy images but operate as black boxes with default styles and no mechanism for fine-tuning on a specific calligraphy font. Thus, they serve as good visual comparison baselines but cannot replace a reproducible fine-tuning pipeline.

Digital font rendering is another important baseline. This approach is orthographically and diacritically correct because characters are rendered directly from the font file. However, font-rendered images are static: there is no ink variation, pressure modulation, dry-brush effect, or organic stroke interaction. Therefore, digital fonts excel at text accuracy but lack calligraphic naturalness.

Open-weight models are also part of the competitive landscape. Qwen Image was the initial registered direction of this thesis, but in practice it was VRAM-intensive and weak on the Vietnamese baseline. ERNIE Image was better than Qwen Image in several respects, typically preserving approximately the correct character count and occasionally generating short Vietnamese phrases correctly. However, ERNIE Image still had many character-level errors, and the earlier ERNIE Image phase showed that the Mistral3/Ministral3 byte-level BPE encoding pipeline made syllable-to-diacritic alignment difficult. The base Ideogram4 is the strongest open foundation in this project, but it still confuses difficult Vietnamese diacritics and does not handle multi-word images well on its own.

The main competitors and baselines are summarized in Table 1.1.

**Table 1.1.** Comparison of competitors and baselines for Vietnamese calligraphy image generation.

| System / Method | Strengths | Limitations for This Thesis |
|---|---|---|
| Digital font rendering | Absolutely correct characters and diacritics | Static glyphs; lacks natural brushstroke variation |
| Black-box commercial tools | High image quality; convenient prompting | Default style; no fine-tuning on private data; hard to reproduce |
| Qwen Image | Initial registered direction; belongs to powerful model family | VRAM-intensive; weak Vietnamese baseline in this project |
| ERNIE Image | Better than Qwen Image on character count; occasionally correct phrases | Still many character errors; byte-level BPE hinders syllable/diacritic alignment |
| Base Ideogram4 | Best open text-rendering foundation among tested approaches | Still errors on difficult diacritics; poor at multi-word layouts |
| Proposed Ideogram4 DiT-LoRA pipeline | Learns specific font, reproducible, improves single-word and compound | Currently limited to one primary font and manual evaluation panel |

The research gap is building a reproducible fine-tuning pipeline that learns a specific font style and preserves Vietnamese diacritic accuracy in both single-word and multi-word layouts.

<!-- Figure placeholder: copy the final competitor comparison panel to docs/thesis/figures/fig_1_2_competitor_baseline_comparison.png -->
![Figure 1.2. Visual comparison of competitor methods and the proposed checkpoint.](figures/fig_1_2_competitor_baseline_comparison.png)

**Figure 1.2.** Visual comparison of competitor methods and the proposed checkpoint. Comparable Vietnamese calligraphy prompts are compared across digital font rendering, Nano Banana 2 as a black-box commercial image generator, Qwen Image, ERNIE Image, base Ideogram4, and the final fine-tuned Ideogram4 checkpoint. The figure is intended to show the gap between orthographic correctness, calligraphic naturalness, reproducibility, and fine-tuning capability.

![Figure 1.3. Base-model capability comparison before fine-tuning.](figures/fig_1_3_base_model_capability_comparison.png)

**Figure 1.3.** Base-model capability comparison before fine-tuning. Qwen Image, ERNIE Image, and base Ideogram4 are compared using archived Vietnamese calligraphy-style base renders. This figure separates base-model selection from LoRA improvement and clarifies why Ideogram4 was selected as the final experimental backbone.

## 1.5. Proposed Method and Contributions

The central contribution of this thesis is:

> An Ideogram4 DiT-LoRA implementation for the registered Qwen Image objective of Vietnamese calligraphy image generation, organized as a glyph-binding pipeline to improve diacritic accuracy.

This contribution is presented as an integrated pipeline, not a list of isolated tricks. Components such as signal probes, LoRA target selection, checkpoint averaging, compound training, and manual evaluation are meaningful only when combined to address the same bottleneck.

Compared to the initial Qwen-Image direction, this contribution differs in several important ways. The final backbone is Ideogram4 rather than Qwen-Image, the primary bottleneck is identified as DiT glyph binding rather than a globally diacritic-blind text encoder, and the final objective is expanded from reading individual words correctly to multi-word/long-sentence calligraphic behavior.

## 1.6. Thesis Structure

Chapter 2 presents the theoretical foundations covering GANs, diffusion, Diffusion Transformers, Ideogram4, LoRA, FP8/BF16, and Vietnamese calligraphy characteristics. Chapter 3 describes the implementation, data, diagnostic probes, fine-tuning configuration, and evaluation results. Chapter 4 summarizes theoretical/practical value, limitations, and future directions.

---

# 2. Theoretical Foundations

## 2.1. AI Image Generation: From GANs to Diffusion Transformers

GANs pioneered adversarial image generation learning but struggled with precise character control [1]. DDPM and diffusion models improved stability by learning to reverse a noise-adding process [4, 5]. Latent Diffusion reduced computational cost by operating in latent space [6]. Diffusion Transformers further replaced the convolutional backbone with transformers, better suited for tasks requiring text-image integration [7].

In the Vietnamese calligraphy problem, the Diffusion Transformer must perform multiple tasks simultaneously: understand the prompt, maintain calligraphic style, place text within a layout, and accurately render small diacritical marks. This is why errors cannot be understood merely as OCR errors or font errors; they are binding errors between linguistic conditioning and visual geometry.

## 2.2. Ideogram4 Architecture

The Ideogram4 pipeline in this thesis consists of three main components, based on the public Ideogram4 technical description and the DiffSynth-Studio implementation used to load the model [15, 18].

**Table 2.1.** Overview of the three main components in the Ideogram4 pipeline.

| Component | Role | Significance for Vietnamese Calligraphy |
|---|---|---|
| Qwen3-VL text encoder | Encodes prompt and structured prompt | Provides conditioning that understands Vietnamese |
| Ideogram4 DiT | Generates image latent conditioned on text | Primary LoRA target |
| VAE | Decodes latent into 1024×1024 image | Produces final image |

<!-- Figure placeholder: copy the final architecture diagram to docs/thesis/figures/fig_2_1_ideogram4_architecture.png -->
![Figure 2.1. Ideogram4 architecture used in this thesis.](figures/fig_2_1_ideogram4_architecture.png)

**Figure 2.1.** Ideogram4 architecture used in this thesis. The fine-tuning pipeline freezes the Qwen3-VL text encoder and Ideogram4 base weights, inserts LoRA adapters into selected DiT modules, and decodes the generated latent image through the VAE.

The Qwen3-VL text encoder is kept frozen. Ideogram4 extracts signals from multiple layer taps of the text encoder, concatenates them into a large conditioning vector, then projects through `llm_cond_norm + llm_cond_proj` into the DiT [15, 16]. The thesis probes examine both tap-space and projection-space to determine how far the diacritic signal persists.

<!-- Figure placeholder: copy the final conditioning diagram to docs/thesis/figures/fig_2_2_qwen3vl_multilayer_conditioning.png -->
![Figure 2.2. Multi-layer Qwen3-VL conditioning fed into the DiT.](figures/fig_2_2_qwen3vl_multilayer_conditioning.png)

**Figure 2.2.** Multi-layer Qwen3-VL conditioning fed into the DiT. Hidden states from multiple Qwen3-VL layers are concatenated, normalized, and projected before entering the Ideogram4 DiT. This figure supports the diagnostic question of whether Vietnamese diacritic signal is already weak at the conditioning interface or is lost later during glyph binding.

The DiT is where LoRA is inserted. Key modules include attention, feed-forward, and adaLN modulation. When attention-only LoRA proves insufficient, expanding the target demonstrates that the DiT needs to be influenced through more channels to correct character geometry.

## 2.3. Parameter-Efficient Fine-Tuning Techniques

LoRA assumes that weight updates can be approximated by the product of two low-rank matrices [8]. If the original weight is `W`, LoRA adds the update:

```text
W' = W + B A * scale
```

In this thesis, FP8 base weights are kept frozen while the LoRA branch operates in BF16. This follows the general idea of parameter-efficient adaptation over frozen or quantized base models [8, 28], while the exact FP8/BF16 conversion details are implementation-specific to the local Ideogram4 pipeline. After training, checkpoints must be converted to the correctly scaled inference format.

Checkpoint averaging is used when multiple LoRA checkpoints occupy the same optimization basin but have different error profiles. Averaging adapters can reduce variance and produce more stable checkpoints without increasing inference cost [13, 14].

## 2.4. Vietnamese Calligraphy: Visual and Linguistic Characteristics

Quốc ngữ calligraphy combines the Latin alphabet, the Vietnamese diacritic system, and calligraphic brushwork. A single word may have multiple layers of marks: circumflex or horn diacritics on vowels, plus tone marks. Common errors include missing tone marks, hỏi/ngã confusion, spurious nặng dots, vowel substitutions such as `ư/u`, `ơ/o`, `â/ă/a`, or consonant errors and character drops in multi-word images.

Regarding Unicode, Vietnamese can be represented in composed or decomposed form. Tokenizers may also handle uppercase and lowercase differently. Therefore, the thesis does not rely solely on visible characters but also checks coverage at the token ID level. The final compound set covers 406 Vietnamese diacritical token IDs.

---

# 3. Implementation and Evaluation

## 3.1. System Setup

### 3.1.1. Hardware Configuration

**Table 3.1.** Hardware configuration.

| Component | Configuration |
|---|---|
| GPU | GPU with sufficient VRAM for Ideogram4 FP8 and LoRA training |
| Precision | FP8 base weights, BF16 LoRA |
| Storage | Stores checkpoints, rendered data, probe artifacts, and evaluation results |
| Runtime | Python, PyTorch, DiffSynth-Studio, safetensors |

### 3.1.2. Software Environment

**Table 3.2.** Software environment.

| Component | Role |
|---|---|
| DiffSynth-Studio | Load and inference Ideogram4 |
| PyTorch | LoRA training |
| safetensors | Checkpoint storage |
| HuggingFace Transformers | Tokenizer/text encoder |
| Experiment scripts | Build dataset, train, convert, render, probe |

### 3.1.3. Reproducible Experiment Process

Each experimental branch is managed with its own checkpoint name, log, and render folder. Evaluation panels use the same base seed to avoid confusing checkpoint improvements with seed variance. When a good checkpoint is discovered, it does not overwrite previous checkpoints but is kept as an independent candidate.

<!-- Figure placeholder: copy the final LoRA target diagram to docs/thesis/figures/fig_3_1_widetarget_lora_injection_points.png -->
![Figure 3.1. Wide-target LoRA insertion points in the Ideogram4 DiT.](figures/fig_3_1_widetarget_lora_injection_points.png)

**Figure 3.1.** Wide-target LoRA insertion points in the Ideogram4 DiT. Compared with attention-only LoRA, the final target set includes attention projections, feed-forward layers, and adaLN modulation so that the adapter can influence both token interaction and glyph geometry.

## 3.2. Data and Preprocessing

### 3.2.1. Single-Word Data

The initial phase uses a list of diacritical Vietnamese words to generate single-word images. The fragile 60 panel is used to measure easily confused words during fine-tuning. The human evaluator decides which words are correct/incorrect based on generated images, prioritizing orthographic and diacritic correctness over automatic metrics like SSIM.

### 3.2.2. Token Coverage

Token coverage is checked at the tokenizer level because the same character may map differently in uppercase and lowercase. Enumeration over the lexicon showed that 406 Vietnamese diacritical token IDs needed to be covered. This led to building compound data covering both uppercase and lowercase, rather than assuming only one form was needed; the compound data generation command is recorded in Appendix B.

### 3.2.3. Compound Dataset

The compound set consists of 4/5/7/8-word images, split into two center-aligned lines. Data is rendered from the Thu Phap Thanh Cong Unicode font to create stable targets. The total compound set comprises:

```text
2808 compound images
147 supplementary single-word images
2955 metadata records
406/406 Vietnamese diacritical token IDs covered
```

The compound design originates from the observation that the model generates multi-word text better when trained directly on multi-word examples. This is a key pivot of the thesis: rather than optimizing only single words and hoping for automatic generalization to sentences, the pipeline introduces multi-word layouts into the training distribution.

### 3.2.4. Bounding-Box-Free Prompts

Bounding-box experiments showed that narrow cell-based or line-level bounding boxes were not consistently followed by the model; some words overlapped in position or remained incorrect. Meanwhile, no-bbox prompts with center-aligned target images allowed the model to learn more natural layout rules. Therefore, the compound bridge uses no-bounding-box prompts and lets the target data teach the model how to align lines.

![Figure 3.2. No-bounding-box and bounding-box layout comparison for compound prompts.](figures/fig_3_2_no_bbox_vs_bbox_layout_comparison.png)

**Figure 3.2.** No-bounding-box and bounding-box layout comparison for compound prompts. The same six-word prompt is rendered under no-bbox, wide-bbox, and row-bbox conditions to show why the final compound data relies on centered target images rather than strict bounding-box instructions.

![Figure 3.3. Font-rendered reference versus model-generated compound layout.](figures/fig_3_3_font_reference_vs_model_generation.jpg)

**Figure 3.3.** Font-rendered reference versus model-generated compound layout. The comparison verifies that the font-rendered compound targets are close enough to the model's natural centered two-line layout to serve as supervised training data while preserving exact Vietnamese characters and diacritics.

## 3.3. Diagnostic Probes

### 3.3.1. Qwen3-VL Signal Probe

The goal of the probe is to check whether Qwen3-VL discards Vietnamese diacritic signal before the DiT receives conditioning. The probe uses a canonical list of 990 words from `manifest_words.json`, compared against a candidate lexicon of 7,165 words from the 7,184 set. The complete probe report is listed in Appendix G. Results:

**Table 3.3.** Qwen3-VL signal probe results.

| Metric | Value |
|---|---:|
| Canonical words | 990 |
| Candidate lexicon | 7,165 |
| Elapsed time | 81.1 s |
| Span miss | 0 |
| Very hard | 42 |
| Hard | 225 |
| Medium | 567 |
| Easy | 156 |

Conclusion: Qwen3-VL is not globally blind to diacritics. However, there is a sufficiently large cluster of hard/very-hard words to use Qwen signal as a risk filter.

### 3.3.2. Cưu/Cừu/Cữu Family Probe

Because training rounds showed `Cưu` and `Cữu` being generated as `Cừu`, a small probe on the six tones was run. The detailed report is listed in Appendix G:

```text
Cưu, Cừu, Cứu, Cửu, Cữu, Cựu
```

The probe measures both tap-space and proj-space after `llm_cond_norm + llm_cond_proj`. Results do not strongly support the hypothesis that `Cừu` is a global attractor in Qwen conditioning. This tilts the conclusion toward DiT/glyph-binding or deeper visual priors.

### 3.3.3. Probe Interpretation

The probes indicate that the entire strategy should not be shifted to training the text encoder. Some words have nearby conditioning signals and can be used for mining, but many image errors occur even when Qwen signal remains distinguishable. For those words, the more appropriate direction is to fix DiT/glyph binding through LoRA and correctly distributed visual data.

## 3.4. Fine-Tuning Configuration

### 3.4.1. Attention-Only Baseline

Attention-only LoRA was tried first because it carries less risk and preserves the base model well. However, results plateaued:

```text
Approximately 32-39/60 on the fragile panel
```

This indicates that Vietnamese diacritic errors do not lie solely in the attention pathway; intervention in modules that more strongly affect stroke geometry is needed.

### 3.4.2. Wide-Target DiT-LoRA

Wide-target LoRA expands the target to the module groups shown in Table 3.4.

**Table 3.4.** LoRA module groups in the DiT.

| Module group | Target module(s) | Intended role |
|---|---|---|
| Attention input projection | `attention.qkv` | Adjust query/key/value text-image interaction |
| Attention output projection | `attention.o` | Adjust attention output integration |
| Feed-forward network | `feed_forward.w1`, `feed_forward.w2`, `feed_forward.w3` | Influence nonlinear glyph and stroke geometry |
| Adaptive layer normalization modulation | `adaln_modulation` | Influence conditioning strength and block-wise modulation |

The best single-checkpoint result reached 48/60. Warm-continue rounds showed high variance: some rounds improved, while others dropped sharply, such as wide8.

### 3.4.3. Gentle Warm-Continue

Gentle warm-continue uses a mild learning rate and warms from a good checkpoint. However, results showed that long blind chaining should be avoided. Instead, each branch should be treated as an independent sample from the good region; if it passes the gate, it enters the candidate soup.

### 3.4.4. Checkpoint Averaging

Checkpoint averaging creates `soup567` from the best r5, r6, r7 checkpoints:

```text
soup567 = mean(r5, r6, r7)
```

The `soup567` result reached 52/60, surpassing all previous single checkpoints. Adding r9 to create `soup5679` still reached 52/60, no improvement but maintained stability.

For the compound bridge, checkpoint averaging continued to be used:

```text
soup_e4r2r3
soup_e4r2r3r4
compound_lr3e5_from_gold4
soup_lr3e5_gold4_9to1
```

The `soup_e4r2r3r4` checkpoint is the stable Gold4 milestone, achieving 6 errors on Eval28. From this milestone, the follow-up branch with learning rate `3e-5` achieved 5 errors; light soup at 90% `lr3e5` branch + 10% Gold4 achieved 4 errors and was selected as the final compound checkpoint.

## 3.5. Evaluation Protocol

### 3.5.1. Manual Word-Level Evaluation

Because OCR and automatic metrics are not reliable enough for stylized diacritical calligraphy, the primary evaluation is manual. This choice is consistent with the broader difficulty of accurate visual text rendering and evaluation in generated images [10], [24–26]. A word is considered correct if a human reader sees the correct characters, correct diacritics, and no extra marks that change meaning. SSIM may be used as a supporting image-similarity reference [23], but it is not the deciding metric.

### 3.5.2. Single-Word Fragile Panel

The fragile panel of 60 consists of difficult words that commonly suffer from diacritic or character errors. This panel is used to measure single-word progress. The base seed is fixed to avoid seed confounding.

### 3.5.3. Compound Eval28

Compound Eval28 consists of 28 images, each containing multiple words, totaling 168 words. This panel better matches the project's final goal than single words because it tests layout, spacing, font size, and the ability to maintain diacritics when multiple tokens compete within the same image. The evaluation directories for the major checkpoints are listed in Appendix E.

### 3.5.4. Testing Preservation of Base Image Generation Capability

Beyond character correctness, the thesis also needs to verify whether LoRA degrades Ideogram4's original capabilities. Therefore, the completed thesis should include a panel of context-rich prompts where Vietnamese calligraphic text is placed in scenes such as Tết greeting cards, ink paintings, festival banners, or poster layouts. This panel does not replace Eval28 but adds a practical perspective: a good checkpoint must both improve text and not impoverish backgrounds, layouts, lighting, and textures compared to the base model.

## 3.6. Results

<!-- Figure placeholder: copy the final progression chart to docs/thesis/figures/fig_3_4_result_progression.png -->
![Figure 3.4. Result progression from the single-word plateau to the final compound checkpoint.](figures/fig_3_4_result_progression.png)

**Figure 3.4.** Result progression from the single-word plateau to the final compound checkpoint. The chart should summarize the improvement path from attention-only LoRA, wide-target single-word training, checkpoint soup, compound bridge training, Gold4, and the final `soup_lr3e5_gold4_9to1` checkpoint.

### 3.6.1. Single-Word Results

**Table 3.5.** Single-word panel results.

| Checkpoint / Phase | Correct Words / 60 |
|---|---:|
| Attention-only plateau | 32-39 |
| Wide-target r3 | 43 |
| Wide-target r4 | 41 |
| Wide-target r5 | 46 |
| Wide-target r6 | 48 |
| Wide-target r7 | 47 |
| Wide-target r8 | 38 |
| `soup567` | **52** |
| `soup5679` | **52** |

Results show that wide-target broke the attention-only plateau, but warm-continue has high variance. Checkpoint soup is a more stable approach than single checkpoints.

![Figure 3.5. Before-and-after examples on difficult Vietnamese diacritical words.](figures/fig_3_5_single_word_before_after_examples.png)

**Figure 3.5.** Before-and-after examples on difficult Vietnamese diacritical words. The panel compares difficult single-word cases before and after wide-target training and checkpoint averaging, illustrating reductions in tone-mark loss, wrong tone substitution, and vowel-diacritic confusion.

### 3.6.2. High-Variance Behavior

Wide8 is an important case because it used the same general recipe but dropped sharply to 38/60 with many spurious nặng dots. The log did not show checkpoint corruption or NaN. The more plausible explanation is that the diacritic subsystem sits near a high-variance boundary; a normal update can push the model into a bad region. This reinforces the rationale for using soup and manual gating.

### 3.6.3. Compound Bridge Results

**Table 3.6.** Compound Eval28 result progression.

| Checkpoint | Errors / 168 |
|---|---:|
| `soup567` baseline | 56 |
| `compound_bridge` e4 | 26 |
| `compound_bridge_r2` | 19 |
| `compound_bridge_r3` | 18 |
| `soup_e4r2r3` | 13 |
| `r4_from_soup` | 15 |
| `soup_e4r2r3r4` / Gold4 | 6 |
| `compound_lr3e5_from_gold4` | 5 |
| `soup_lr3e5_gold4_3to1` | 6 |
| `soup_lr3e5_gold4_9to1` | **4** |

The post-Gold4 process shows that both learning rate and soup ratio matter. The earlier `5e-5` branch learned too aggressively and created many new errors; `2e-5` maintained 6 errors but could not surpass the milestone; `3e-5` was a better region, reducing solo errors to 5. When souping with Gold4, the 75/25 ratio brought some old errors back, while the 90/10 ratio retained most of the new branch's improvements and used Gold4 only as a light regularizer.

![Figure 3.6. Compound Eval28 before-and-after comparison.](figures/fig_3_6_compound_eval28_before_after.png)

**Figure 3.6.** Compound Eval28 before-and-after comparison. Representative fixed-seed Eval28 prompts are shown for `soup567` and the final compound checkpoint `soup_lr3e5_gold4_9to1`, illustrating the reduction from 56/168 errors to 4/168 errors on multi-word two-line calligraphy images.

Final compound checkpoint:

```text
experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors
```

Remaining errors on Eval28:

```text
Hấn -> Hẩn
Chịt -> Chút
Huyên -> Huyện
Dôi -> Dồi
```

![Figure 3.7. Remaining errors of the current compound gold checkpoint.](figures/fig_3_7_remaining_error_cases.png)

**Figure 3.7.** Remaining errors of the current compound gold checkpoint. The four residual Eval28 errors are concentrated in visually close Vietnamese diacritic or vowel variants rather than representing broad failure across the compound panel.

![Figure 3.8. Testing the fine-tuned checkpoint with base image generation capability.](figures/fig_3_8_calligraphy_with_base_model_capability.png)

**Figure 3.8.** Testing the fine-tuned checkpoint with base image generation capability. Two Tết still-life prompts compare base Ideogram4, the single-word checkpoint soup `soup567`, and the final compound checkpoint `soup_lr3e5_gold4_9to1`. The figure checks whether Vietnamese calligraphy accuracy can improve while preserving composition, lighting, paper texture, and decorative background elements.

### 3.6.4. Current Gold Checkpoints

Single-word gold:

```text
experiments/checkpoints/coverage_v10_widetarget_soup567/step-soup_infer.safetensors
```

Compound gold:

```text
experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors
```

Both gold checkpoints have been published on Hugging Face at [phong09021998/vietnamese-calligraphy-ideogram4-lora](https://huggingface.co/phong09021998/vietnamese-calligraphy-ideogram4-lora) for public accessibility and replication. The complete, restructured source code and thesis assets are publicly available on GitHub at [DoTuanPhong/qwen-calligraphy-ideogram4](https://github.com/DoTuanPhong/qwen-calligraphy-ideogram4).

---

# 4. Conclusion

## 4.1. Theoretical and Practical Value

### 4.1.1. Theoretical Value

This thesis demonstrates that Vietnamese diacritic errors in the Ideogram4 setup should not be simplified to text-encoder-global-diacritic-blindness errors. Qwen3-VL retains diacritic signal in many cases. The core problem is that the DiT does not always bind that signal to the correct glyph geometry. This distinction matters because it changes the strategy: rather than globally training the text encoder, the more effective direction is to adjust the DiT at modules that influence character geometry.

The thesis also shows that single-word and multi-word are related but distinct problems. A checkpoint may write reasonably well with only one word but fail when required to write multiple words across two lines. Therefore, if the ultimate goal is long sentences, direct training on multi-word layouts is necessary.

### 4.1.2. Practical Value

This thesis provides a reproducible technical pipeline for Vietnamese calligraphy image generation. The pipeline includes data creation from the target font, coverage checking at the tokenizer level, LoRA fine-tuning, checkpoint conversion, checkpoint averaging, evaluation rendering with fixed seeds, and manual word-level inspection. These components are kept reproducible because the problem is highly sensitive to seeds, checkpoints, and prompt formatting.

The compound error reduction from 56/168 to 4/168 demonstrates that the pipeline can produce more practical checkpoints for the goal of multi-word calligraphy generation. Its value lies not only in a single experiment but also in providing a foundation for applications in graphic design, personalized greeting images, digital cultural preservation, and calligraphy education.

## 4.2. Summary of Key Results

The experiments show a clear improvement trajectory. Attention-only LoRA plateaued at approximately 32–39/60, demonstrating that individual attention modules alone are insufficient to address the Vietnamese diacritic bottleneck. When the LoRA target was expanded to more DiT modules, the best single checkpoint increased to 48/60. Subsequently, averaging compatible checkpoints produced `soup567`, reaching 52/60.

The next important result is that single-word performance does not automatically transfer to multi-word images. The single-word gold checkpoint still produced many errors when writing two-line layouts. However, after compound bridge training, follow-up learning rate `3e-5`, and light 90/10 soup with the Gold4 milestone, errors on Eval28 dropped from 56/168 to 4/168. The remaining errors are concentrated in a few difficult diacritic pairs, no longer representing global collapse.

## 4.3. Current Limitations

The current work still has several limitations. The most immediate is the cost of manual evaluation: the primary metric relies on human image inspection, which is reliable in the calligraphy domain but slow. The Eval28 panel is useful enough for rapid iteration, and seed 7000 enables fair comparison after discovering seed confounding, but stronger statistical conclusions still require larger benchmarks and more seeds.

The style scope is also narrow. Experiments focused on Thu Phap Thanh Cong Unicode, so generalization to other fonts needs further testing. Training data is currently generated from digital calligraphy fonts; this approach is very useful for controlling orthography and Vietnamese diacritics but cannot fully capture the materiality of real hand-written calligraphy, such as paper texture, ink bleeding, imperfect pressure, and the artist's personal layout.

The content scope also needs expansion. Compound 4/5/7/8-word images are closer to the final goal than single words, but natural Vietnamese sentences still require their own panel. Additionally, although LoRA reduces the risk of degrading the base model by keeping original weights frozen, broader testing of base image generation capability and adapter behavior on non-calligraphy prompts is still needed.

## 4.4. Future Directions

Because the proposed pipeline has already achieved strong results on the current compound panel, the next development direction should not merely be continuing to find better checkpoints on the same panel. Instead, post-thesis steps should focus on expanding scope, standardizing evaluation, and bringing the system closer to practical applications.

One important direction is expanding to multiple calligraphy fonts. This thesis currently focuses on Thu Phap Thanh Cong Unicode, but a practical calligraphy system needs to support diverse Quốc ngữ styles. Building data for additional fonts would allow testing whether the same DiT-LoRA pipeline can learn multiple brushwork styles while maintaining Vietnamese diacritic accuracy. This is the transition from a single-style model to a font-controllable calligraphy generation system.

A second direction is learning from real calligraphy photographs. Digital font data provides stable orthographic supervision, but works written by real artists contain richer brush dynamics, ink bleeding, paper texture, imperfect pressure, and more natural layout. A practical strategy is to use font-rendered data to stabilize characters and diacritics, then supplement with curated real calligraphy images to enhance artistic quality while maintaining text accuracy.

The system also needs to be extended to longer, more natural Vietnamese content. Compound data has moved beyond the single-word problem, but practical applications typically require greetings, proper names, slogans, or short poems. Therefore, follow-up research should evaluate 9–16 word phrases with line breaks and layout rules closer to real calligraphic works.

Regarding evaluation and deployment, larger benchmarks with more seeds and the ability to reflect more sentence types are needed. A lightweight evaluator could help detect missing marks, swapped marks, extra marks, or character substitutions, thereby reducing manual scoring costs; however, manual evaluation should remain the final standard for aesthetic quality and ambiguous cases. Simultaneously, inference needs to become cheaper before the system can be used interactively. Directions such as quantization, optimized attention kernels, model compilation, or smaller adapters all align with the goal of deploying the model into design tools, web services, or batch image generation workflows.

Finally, the remaining errors open a direction toward anti-hallucination in text rendering. The model no longer fails broadly, but it still confuses some difficult diacritic and vowel pairs. Hard-word mining, replay data with correct glyphs, and layout-binding specialist adapters could address this error group. As those components mature, the system could be integrated into a tool where users input Vietnamese text, select a calligraphy style, choose an image background, and receive high-resolution images for design, cultural preservation, or education.

In summary, this thesis builds an empirical, evidence-based Ideogram4 fine-tuning pipeline for Vietnamese calligraphy image generation. The results demonstrate that accurate Vietnamese rendering requires not only a strong base model but also knowing where the diacritic signal persists, targeting the correct DiT modules, stabilizing checkpoints, and training directly on the multi-word layout distribution that practical applications require.

---

# References

[1] I. Goodfellow et al., "Generative adversarial nets," in *Proc. 27th Int. Conf. Neural Inf. Process. Syst. (NeurIPS)*, Montreal, QC, Canada, Dec. 2014, pp. 2672–2680.

[2] Z. Lyu, X. Bai, B. Shi, and C. Yao, "CalliGAN: Style and structure-aware Chinese calligraphy character generator," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, Seattle, WA, USA, Jun. 2020, pp. 494–495.

[3] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Venice, Italy, Oct. 2017, pp. 2242–2251.

[4] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in *Proc. 34th Int. Conf. Neural Inf. Process. Syst. (NeurIPS)*, Vancouver, BC, Canada, Dec. 2020, pp. 6840–6851.

[5] J. Song, C. Meng, and S. Ermon, "Denoising diffusion implicit models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vienna, Austria, May 2021, pp. 1–22.

[6] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-resolution image synthesis with latent diffusion models," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, New Orleans, LA, USA, Jun. 2022, pp. 10674–10685.

[7] W. Peebles and S. Xie, "Scalable diffusion models with transformers," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Paris, France, Oct. 2023, pp. 4172–4182.

[8] E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Apr. 2022, pp. 1–16.

[9] N. Ruiz et al., "DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Vancouver, BC, Canada, Jun. 2023, pp. 22500–22510.

[10] M. Chen et al., "TextDiffuser: Diffusion models as text painters," in *Proc. 37th Int. Conf. Neural Inf. Process. Syst. (NeurIPS)*, New Orleans, LA, USA, Dec. 2023, pp. 9353–9367.

[11] A. Radford et al., "Learning transferable visual models from natural language supervision," in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, Jul. 2021, pp. 8748–8763.

[12] Y. Li, S. Wang, J. Zhang, and K. Chen, "BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models," in *Proc. 40th Int. Conf. Mach. Learn. (ICML)*, Honolulu, HI, USA, Jul. 2023, pp. 1–13.

[13] T. Wortsman et al., "Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time," in *Proc. 39th Int. Conf. Mach. Learn. (ICML)*, Baltimore, MD, USA, Jul. 2022, pp. 23965–23998.

[14] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, "Averaging weights leads to wider optima and better generalization," in *Proc. 34th Conf. Uncertainty Artif. Intell. (UAI)*, Monterey, CA, USA, Aug. 2018, pp. 876–885.

[15] Ideogram AI, "Ideogram 4.0 technical details," Ideogram AI, Tech. Rep., Jun. 2026. [Online]. Available: https://ideogram.ai/blog/ideogram-4.0/, accessed: Jun. 28, 2026.

[16] Qwen Team, "Qwen3-VL technical report," *arXiv:2511.21631*, Nov. 2025. [Online]. Available: https://arxiv.org/abs/2511.21631

[17] Baidu ERNIE-Image Team, "Introducing ERNIE-Image," Baidu, Tech. Rep., 2026. [Online]. Available: https://yiyan.baidu.com/blog/posts/ernie-image, accessed: Jun. 28, 2026.

[18] ModelScope Contributors, "DiffSynth-Studio: A diffusion engine," Open-source software repository, 2024–2026. [Online]. Available: https://github.com/modelscope/DiffSynth-Studio, commit `6d103c0`, accessed: Jun. 5, 2026.

[19] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, "Flow matching for generative modeling," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Kigali, Rwanda, May 2023, pp. 1–17.

[20] P. Esser, S. Kulal, A. Blattmann, R. Rombach, B. Ommer, and J. Z. Kolter, "Scaling rectified flow transformers for high-resolution image synthesis," in *Proc. 41st Int. Conf. Mach. Learn. (ICML)*, Vienna, Austria, Jul. 2024, pp. 1–14.

[21] Qwen Team, "Qwen-Image technical report," *arXiv:2508.02324*, Aug. 2025. [Online]. Available: https://arxiv.org/abs/2508.02324

[22] Black Forest Labs, "FLUX: A fast lightweight universal model," Black Forest Labs, Tech. Rep., 2024. [Online]. Available: https://github.com/black-forest-labs/flux, accessed: Jun. 28, 2026.

[23] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Trans. Image Process.*, vol. 13, no. 4, pp. 600–612, Apr. 2004.

[24] Y. Tuo, W. Xiang, J.-Y. He, Y. Geng, and X. Xie, "AnyText: Multilingual visual text generation and editing," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vienna, Austria, May 2024, pp. 1–17.

[25] Y. Yang et al., "GlyphControl: Glyph conditional control for visual text generation," in *Proc. 37th Int. Conf. Neural Inf. Process. Syst. (NeurIPS)*, New Orleans, LA, USA, Dec. 2023, pp. 1–16.

[26] Z. Liu, X. Yang, H. Zhang, and A. C. Berg, "Glyph-ByT5: A customized text encoder for accurate visual text rendering," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, Milan, Italy, Sep.–Oct. 2024, pp. 1–18.

[27] D. Kalajdzievski, "A rank stabilization scaling factor for fine-tuning with LoRA," *arXiv:2312.03732*, Dec. 2023. [Online]. Available: https://arxiv.org/abs/2312.03732

[28] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in *Proc. 37th Int. Conf. Neural Inf. Process. Syst. (NeurIPS)*, New Orleans, LA, USA, Dec. 2023, pp. 10088–10115.

---

**Notes on the validity of selected references**

- `[15]` Ideogram AI 2026 — points to the official Ideogram 4.0 technical blog post at `https://ideogram.ai/blog/ideogram-4.0/`. This is the sole official source published by Ideogram AI as of this writing; *an arXiv-format technical report has not been released*. The web URL and access date `Jun. 28, 2026` are recorded in the reference entry.
- `[16]` Qwen3-VL Technical Report (*arXiv:2511.21631*, submitted 2025-11-26) — the official arXiv paper by the Qwen Team describing the Qwen3-VL architecture in detail. Updated after the initial draft was written. Previously this reference was listed generically as "technical report, 2026" because the arXiv version had not yet appeared; now it has an official arXiv ID (`https://arxiv.org/abs/2511.21631`) and can be directly looked up. Qwen3-VL describes dense 2B/4B/8B/32B and MoE 30B-A3B/235B-A22B variants, supporting 256K-token interleaved context. The version used by Ideogram4 is Qwen3-VL-8B-Instruct (the 8B dense variant).
- `[17]` Baidu ERNIE-Image Team — points to the official blog post "Introducing ERNIE-Image" at `https://yiyan.baidu.com/blog/posts/ernie-image`. This is Baidu's official announcement source for ERNIE-Image, although a PDF-format technical report has not been widely released. The web URL and access date `Jun. 28, 2026` are recorded in the reference entry.
- `[21]` Qwen-Image Technical Report (*arXiv:2508.02324*) — this is the reference cited for the Qwen-Image architecture section, with an official arXiv ID (`https://arxiv.org/abs/2508.02324`) and the most peer-reviewable source.
- `[18]` DiffSynth-Studio — open-source software reference. The repository URL (`https://github.com/modelscope/DiffSynth-Studio`), the 2024–2026 development span, and the specific commit hash (`6d103c0`, accessed 2026-06-05) used to load Ideogram4 are now recorded in the reference entry.

References `[1]`–`[14]`, `[19]`, `[20]`, and `[23]`–`[28]` are widely published works, searchable via Google Scholar or arXiv; no authenticity issues.

**Cross-referencing body ↔ reference list:** The model-specific citations used in Section 1.4.3 have been cross-checked against the reference list: Qwen-Image is cited with `[21]`, ERNIE-Image with `[17]`, Ideogram4 with `[15–16]`, and FLUX with `[22]`. General visual-text-rendering limitations are supported by `[10], [24–26]`, while flow-matching background is supported by `[19–20]`. In later revision rounds, reference `[16]` was updated from the generic "technical report, 2026" form to the official arXiv ID `2511.21631` after the Qwen Team published the full technical report.

---

# Appendices

## Appendix A: Main Wide-Target Training Command

```bash
bash experiments/scripts/v10/run_widetarget_gentle.sh \
  <warm_checkpoint> \
  <output_dir> \
  3 \
  5e-5 \
  <metadata_jsonl>
```

## Appendix B: Compound Data Generation Command

```bash
python3 experiments/scripts/v10/build_compound_dataset.py \
  --seed 27 \
  --img_sub images_compound_r5 \
  --out metadata_compound_2808_r5.jsonl
```

## Appendix C: Compound Branch Post-Processing

```bash
BASE_WEIGHT=0.1111111111 experiments/scripts/v10/postprocess_compound_branch.sh \
  no_training_session_for_manual_soup \
  experiments/checkpoints/coverage_v10_compound_lr3e5_from_gold4/step-11820.safetensors \
  experiments/checkpoints/coverage_v10_compound_soup_e4r2r3r4/step-soup.safetensors \
  experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1 \
  experiments/results/coverage_v10_eval/compound_eval28_soup_lr3e5_gold4_9to1
```

The script computes soup using the formula `(BASE_WEIGHT * base + new) / (BASE_WEIGHT + 1)`, so `BASE_WEIGHT=0.1111111111` corresponds to approximately 90% `lr3e5` branch and 10% Gold4.

## Appendix D: Checkpoint Registry

```text
Single-word gold:
experiments/checkpoints/coverage_v10_widetarget_soup567/step-soup_infer.safetensors

Compound gold:
experiments/checkpoints/coverage_v10_compound_soup_lr3e5_gold4_9to1/step-soup_infer.safetensors
```

## Appendix E: Evaluation Directories

```text
experiments/results/coverage_v10_eval/compound_eval28_soup567_baseline
experiments/results/coverage_v10_eval/compound_eval28_bridge_e4
experiments/results/coverage_v10_eval/compound_eval28_bridge_r2
experiments/results/coverage_v10_eval/compound_eval28_bridge_r3
experiments/results/coverage_v10_eval/compound_eval28_soup_e4r2r3
experiments/results/coverage_v10_eval/compound_eval28_soup_e4r2r3r4
experiments/results/coverage_v10_eval/compound_eval28_lr3e5_from_gold4
experiments/results/coverage_v10_eval/compound_eval28_soup_lr3e5_gold4_9to1
```

## Appendix F: Remaining Errors of Compound Gold

```text
Hấn -> Hẩn
Chịt -> Chút
Huyên -> Huyện
Dôi -> Dồi
```

## Appendix G: Internal Probe Reports

```text
docs/probe_reports/_QWEN3_VL_SIGNAL_PROBE_2026-06-23.md
docs/probe_reports/_QWEN3_VL_CUU_FAMILY_PROBE_2026-06-24.md
```

## Appendix H: Comparison Image List

The following filenames are reserved for figures to be included in the completed thesis. Files marked as `READY` already exist in `docs/thesis/figures/`.

**Table H.1.** Comparison image list and figure readiness status.

| Figure | Filename | Status | Expected Content |
|---|---|---|---|
| Figure 1.1 | `docs/thesis/figures/fig_1_1_research_pipeline.png` | READY | Overall research pipeline from baseline observation, signal probing, LoRA training, checkpoint soup, compound training, and final evaluation |
| Figure 1.2 | `docs/thesis/figures/fig_1_2_competitor_baseline_comparison.png` | READY | Digital font, Nano Banana 2 commercial baseline, Qwen Image, ERNIE Image, base Ideogram4, and proposed checkpoint |
| Figure 1.3 | `docs/thesis/figures/fig_1_3_base_model_capability_comparison.png` | READY | Qwen Image, ERNIE Image, and base Ideogram4 on Vietnamese calligraphy-style prompts before fine-tuning |
| Figure 2.1 | `docs/thesis/figures/fig_2_1_ideogram4_architecture.png` | READY | Ideogram4 architecture used in this thesis: frozen Qwen3-VL, DiT with LoRA adapters, and VAE decoder |
| Figure 2.2 | `docs/thesis/figures/fig_2_2_qwen3vl_multilayer_conditioning.png` | READY | Multi-layer Qwen3-VL hidden-state taps, concatenation, normalization, projection, and DiT conditioning |
| Figure 3.1 | `docs/thesis/figures/fig_3_1_widetarget_lora_injection_points.png` | READY | Wide-target LoRA insertion points: attention, feed-forward, and adaLN modulation |
| Figure 3.2 | `docs/thesis/figures/fig_3_2_no_bbox_vs_bbox_layout_comparison.png` | READY | Same six-word compound prompt under no-bbox, wide-bbox, and row-bbox conditions |
| Figure 3.3 | `docs/thesis/figures/fig_3_3_font_reference_vs_model_generation.jpg` | READY | Font-rendered reference compared with model-generated compound layout |
| Figure 3.4 | `docs/thesis/figures/fig_3_4_result_progression.png` | READY | Result progression from attention-only LoRA to wide-target, `soup567`, compound bridge, Gold4, and final `soup_lr3e5_gold4_9to1` |
| Figure 3.5 | `docs/thesis/figures/fig_3_5_single_word_before_after_examples.png` | READY | Before and after examples of difficult single words with wide-target/checkpoint soup |
| Figure 3.6 | `docs/thesis/figures/fig_3_6_compound_eval28_before_after.png` | READY | Compound Eval28 examples comparing `soup567` with the final checkpoint `soup_lr3e5_gold4_9to1` |
| Figure 3.7 | `docs/thesis/figures/fig_3_7_remaining_error_cases.png` | READY | Remaining errors: `Hấn`, `Chịt`, `Huyên`, `Dôi` |
| Figure 3.8 | `docs/thesis/figures/fig_3_8_calligraphy_with_base_model_capability.png` | READY | Context-rich prompts testing whether the fine-tuned checkpoint preserves Ideogram4's base image generation capability when adding Vietnamese calligraphic text |

## Appendix I: Prompts Used for Comparison Figures

The prompt used for Figure 1.3 is the same short prompt for all base models, allowing Vietnamese calligraphy generation capability to be compared before fine-tuning:

```text
Traditional Vietnamese calligraphy written in black ink on white paper.
The text says "An Khang Thịnh Vượng".
```

Figure 3.8 uses context-rich JSON prompts to test the ability to combine calligraphic text with base image generation capability. The image set is rendered with the command:

```bash
python3 experiments/scripts/v10/render_thesis_scene_candidates.py \
  --preset phuc_duc \
  --seed 7002
```

The two text elements used in Figure 3.8 are:

```text
Prompt A:
Phúc Thọ
An Khang

Prompt B:
Tâm Đức Trí Nhân
Phúc Thọ Khang Vượng
```

The shared scene description in the prompt requests a Vietnamese Tết still life on a red lacquer table, including cream-colored dó/rice paper, black ink calligraphy, apricot blossoms, kumquat fruits, calligraphy brushes, ink stones, and red seals. The prompt also requests preserving background image details such as lighting, paper texture, lacquer reflections, and bokeh, while maintaining correct Vietnamese diacritics and avoiding Chinese character generation.

The two full JSON prompts are stored at:

```text
experiments/results/coverage_v10_eval/thesis_scene_candidates/phuc_duc_seed7002/prompt_phuc_tho_an_khang.json
experiments/results/coverage_v10_eval/thesis_scene_candidates/phuc_duc_seed7002/prompt_tam_duc_tri_nhan.json
```

These phrases were selected after trying several variants. The phrase `Lộc` was not used in the promote figure because it tends to lose the `ô` circumflex, while `Vượng` produced more acceptable results in the same context.

## Appendix J: Reference and Evidence Checklist

This appendix records which parts of the thesis are supported by external literature and which parts are supported by internal experimental artifacts. It is intended as a final-submission checklist rather than a replacement for the main reference list.

**Table J.1.** Reference and evidence checklist.

| Thesis location | Claim or content needing support | Current support | Final action before submission |
|---|---|---|---|
| Abstract; Sections 1.1 and 1.2.1 | Qwen Image was the registered initial direction but was not selected as the final backbone due to VRAM cost and weak Vietnamese zero-shot rendering | Internal baseline observations; Figure 1.3; Qwen-Image technical report [21] | Keep Figure 1.3 and, if possible, retain the exact Qwen Image base render/log used in the comparison folder |
| Abstract; Sections 1.1, 1.2.1, and 1.4.3 | ERNIE Image preserved character count better than Qwen Image in some cases but still had many character-level errors | Internal ERNIE experiments and `docs/thesis/figures/ernie_image_base.jpeg`; ERNIE Image source [17] | Keep the selected ERNIE baseline image and avoid overclaiming beyond observed examples |
| Sections 1.1 and 1.4.3 | ERNIE/Mistral3 byte-level BPE created Vietnamese tokenizer instability, especially for capitalized diacritical forms | `docs/ernie_image_tokenization_report.vi.md`; `docs/v10_tokenizer_robustness_report.vi.md` | Preserve these reports or summarize them in a submitted appendix if the thesis package must be self-contained |
| Section 1.4.3 and Section 2.2 | Ideogram4 architecture, Qwen3-VL text encoder, multi-layer hidden-state conditioning, flow-matching, and FP8/BF16 practical pipeline | Ideogram4 technical documentation [15], Qwen3-VL technical report [16], DiffSynth-Studio [18], flow-matching references [19, 20] | Check that the final bibliography includes stable URLs or access dates for technical web sources |
| Sections 1.4.1-1.4.4 and Chapter 2 | GANs, diffusion models, latent diffusion, DiT, visual text rendering, LoRA, model soups, SSIM, and quantized/parameter-efficient fine-tuning | External references [1]-[14], [19]-[28] | No additional action unless the final university format requires DOI/arXiv URLs |
| Sections 3.2.2 and 3.2.3 | The compound dataset covers 406/406 Vietnamese diacritical token IDs and contains 2808 compound images plus supplementary single-word samples | `data/coverage_v10/metadata_compound_2808.jsonl`; Appendix B; tokenizer robustness reports | Keep the metadata file and build script with the submitted artifact package |
| Section 3.3 | Qwen3-VL is not globally blind to Vietnamese diacritics; the 990-vs-7165 probe produced 42 very-hard, 225 hard, 567 medium, and 156 easy cases | Internal probe reports in Appendix G and persisted probe results under `experiments/results/v10_phase1_probes/` | Keep probe reports and result tables because this is an internal empirical claim |
| Sections 3.4-3.6 | Attention-only plateau, wide-target improvement, `soup567`, compound bridge, Gold4, and final 4/168 Eval28 result | Appendix C, D, E, F; evaluation directories and checkpoint registry | Keep checkpoint names, eval folders, and manual scoring notes consistent with the final figures |
| Section 3.5 | Manual evaluation is the primary metric because OCR/automatic metrics are unreliable for stylized Vietnamese calligraphy | Visual text rendering references [10], [24–26], SSIM reference [23], and internal manual evaluation protocol | If the thesis committee asks for stronger evaluation, add a short note that OCR is not used as the final judge because the target domain is stylized calligraphy |
| Section 3.5.4 and Figure 3.8 | Fine-tuned checkpoints should preserve base image generation capability while improving Vietnamese calligraphy | Figure 3.8 and prompt records in Appendix I | Keep the exact prompts, seeds, and source images used to build Figure 3.8 |
| Section 4.4 | Future extension to real calligraphy photographs and multiple fonts | This is a proposed future direction, not a completed result | No citation is strictly required, but it should remain framed as future work rather than current achievement |

---

# Copyright and Tool Acknowledgements

This thesis acknowledges the models, open-source tools, and research materials used during the experimental process, including Ideogram4, Qwen3-VL, ERNIE Image reference materials, DiffSynth-Studio, PyTorch, HuggingFace Transformers, safetensors, and related works on image generation. All generated data, checkpoints, probes, and evaluation reports referenced in this thesis are used for academic research purposes.
