# Thesis Figure Asset Checklist

Place the final comparison images in this directory with the exact filenames below.

| Figure | Filename | Intended content |
|---|---|---|
| Figure 1.1 | `fig_1_1_research_pipeline.png` | READY: programmatic pipeline diagram from baseline survey and signal probes to wide-target LoRA, checkpoint soup, compound training, and final evaluation |
| Figure 1.2 | `fig_1_2_competitor_baseline_comparison.png` | DRAFT: machine font, placeholder commercial cell, Qwen Image, ERNIE Image, raw Ideogram4, and the proposed checkpoint |
| Figure 1.3 | `fig_1_3_base_model_capability_comparison.png` | READY: user-provided Qwen Image and ERNIE Image base renders plus raw Ideogram4 on the same simple Vietnamese calligraphy prompt before fine-tuning |
| Figure 2.1 | `fig_2_1_ideogram4_architecture.png` | READY: Ideogram4 architecture diagram with frozen Qwen3-VL, projected conditioning, DiT LoRA, VAE, and final image |
| Figure 2.2 | `fig_2_2_qwen3vl_multilayer_conditioning.png` | READY: Qwen3-VL layer taps, concatenation, `llm_cond_norm`, `llm_cond_proj`, and DiT conditioning |
| Figure 3.1 | `fig_3_1_widetarget_lora_injection_points.png` | READY: wide-target LoRA insertion points in adaLN, attention, and feed-forward modules |
| Figure 3.2 | `fig_3_2_no_bbox_vs_bbox_layout_comparison.png` | READY: same six-word prompt with no-bbox, wide bbox, and row bbox |
| Figure 3.3 | `fig_3_3_font_reference_vs_model_generation.jpg` | READY: contact sheet comparing font-rendered reference against model-matched generated layout |
| Figure 3.4 | `fig_3_4_result_progression.png` | READY: single-word fragile60 and compound Eval28 progression chart |
| Figure 3.5 | `fig_3_5_single_word_before_after_examples.png` | READY: difficult single-word examples before and after wide-target/checkpoint soup |
| Figure 3.6 | `fig_3_6_compound_eval28_before_after.png` | READY: Compound Eval28 examples comparing `soup567` with final `soup_lr3e5_gold4_9to1` |
| Figure 3.7 | `fig_3_7_remaining_error_cases.png` | READY: remaining errors `Hấn -> Hẩn`, `Chịt -> Chút`, `Huyên -> Huyện`, `Dôi -> Dồi` |
| Figure 3.8 | `fig_3_8_calligraphy_with_base_model_capability.png` | READY: two scene-rich Tet calligraphy prompts at seed 7002 comparing raw Ideogram4, `soup567`, and final `soup_lr3e5_gold4_9to1`; current promoted phrases avoid `Lộc` because it tended to lose the `ô` circumflex |

Regeneration script:

```bash
python docs/thesis/figures/make_missing_figures.py
```

This script regenerates Figures 1.1, 1.2, 2.1, 2.2, 3.1, and 3.4. Figure 1.2 remains a draft until the commercial-tool placeholder cell is replaced with a real archived render.

Individual source panels prepared for later manual layout:

| Source asset | Intended use |
|---|---|
| `qwen_image_base.png` | User-provided Qwen Image base calligraphy example |
| `ernie_image_base.jpeg` | User-provided ERNIE Image base calligraphy/scene example |
| `ideogram4_base_calligraphy_simple_prompt_s7000.jpg` | Raw Ideogram4 output for the same simple base-model prompt used by Qwen Image and ERNIE Image; this run returned the safety-filter placeholder |
| `ideogram4_base_calligraphy_project_json_prompt_s7000.jpg` | Raw Ideogram4 output using the project JSON/no-bbox prompt, kept as optional evidence that the prompt interface matters |
| `ideogram4_base_scene_tet_calligraphy_s7000.jpg` | Raw Ideogram4 scene-rich calligraphy prompt for Figure 3.8 |
| `soup567_scene_tet_calligraphy_s7000.jpg` | Optional intermediate single-word soup checkpoint on the same scene-rich prompt, not used in the main Figure 3.8 composite |
| `final_compound_scene_tet_calligraphy_s7000.jpg` | Final compound checkpoint on the same scene-rich prompt |
| `final_compound_calligraphy_simple_prompt_s7000.jpg` | Final compound checkpoint on the same simple base-model prompt |
| `ideogram4_base_scene_tet_calligraphy_aktv_s7002.jpg` | Earlier Figure 3.8 candidate Prompt A, raw Ideogram4 |
| `soup567_scene_tet_calligraphy_aktv_s7002.jpg` | Earlier Figure 3.8 candidate Prompt A, `soup567` |
| `final_compound_scene_tet_calligraphy_aktv_s7002.jpg` | Earlier Figure 3.8 candidate Prompt A, final compound checkpoint |
| `ideogram4_base_scene_tet_calligraphy_8words_s7002.jpg` | Earlier Figure 3.8 candidate Prompt B, raw Ideogram4 |
| `soup567_scene_tet_calligraphy_8words_s7002.jpg` | Earlier Figure 3.8 candidate Prompt B, `soup567` |
| `final_compound_scene_tet_calligraphy_8words_s7002.jpg` | Earlier Figure 3.8 candidate Prompt B, final compound checkpoint |
| `ideogram4_base_scene_tet_calligraphy_phuc_tho_an_khang_s7002.jpg` | Promoted Figure 3.8 Prompt A, raw Ideogram4 |
| `soup567_scene_tet_calligraphy_phuc_tho_an_khang_s7002.jpg` | Promoted Figure 3.8 Prompt A, `soup567` |
| `final_compound_scene_tet_calligraphy_phuc_tho_an_khang_s7002.jpg` | Promoted Figure 3.8 Prompt A, final compound checkpoint |
| `ideogram4_base_scene_tet_calligraphy_tam_duc_tri_nhan_s7002.jpg` | Promoted Figure 3.8 Prompt B, raw Ideogram4 |
| `soup567_scene_tet_calligraphy_tam_duc_tri_nhan_s7002.jpg` | Promoted Figure 3.8 Prompt B, `soup567` |
| `final_compound_scene_tet_calligraphy_tam_duc_tri_nhan_s7002.jpg` | Promoted Figure 3.8 Prompt B, final compound checkpoint |
