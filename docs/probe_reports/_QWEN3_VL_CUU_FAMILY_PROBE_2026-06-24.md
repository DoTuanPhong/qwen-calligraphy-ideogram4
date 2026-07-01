# Qwen3-VL `Cưu/Cừu/Cữu` Family Probe

Date: 2026-06-24

## Question

Recent manual evaluation found that both `Cữu` and `Cưu` can render as `Cừu`.
This follow-up checks a narrower question than the 990-word probe:

> Is `Cừu` already a nearest attractor in the Qwen3-VL conditioning signal, before the DiT draws anything?

This is a text-conditioning probe only. It does not generate images, train, or modify checkpoints.

## Setup

- Words: `Cưu`, `Cừu`, `Cứu`, `Cửu`, `Cữu`, `Cựu`.
- Model path: `models/ideogram-ai/ideogram-4-fp8`.
- Text encoder load path: DiffSynth `Ideogram4TextEncoder`, so FP8 `weight_scale` tensors are honored.
- Prompt modes:
  - `json`: production `build_no_bbox_prompt(word)`, wrapped through the same Ideogram4 chat template before tokenization.
  - `word_only`: control prompt containing only the word, also wrapped through chat template.
- JSON spans:
  - `element_text`: the word inside `compositional_deconstruction.elements[0].text`; primary metric.
  - `occurrence_0`: first exact occurrence in high-level description.
  - `occurrence_1`: second exact occurrence, identical to `element_text`.
  - `all_occurrences_pooled`: mean over all exact occurrences.
- Distances:
  - `tap_dist`: cosine distance on concat tap layers `0,3,6,9,12,15,18,21,24,27,30,33,35`.
  - `L0/L3/.../L35`: per-layer cosine distance sliced from the tap concat.
  - `proj_dist`: after `llm_cond_norm + llm_cond_proj`.
  - `proj_nobias_dist`: same projection without bias.

Artifacts:

- `result/v10_phase1_probes/qwen_signal_cuu_family_2026-06-24/cuu_family_signal.json`
- `result/v10_phase1_probes/qwen_signal_cuu_family_2026-06-24/cuu_family_pairs.csv`
- `result/v10_phase1_probes/qwen_signal_cuu_family_2026-06-24/cuu_family_occurrences.csv`
- `result/v10_phase1_probes/qwen_signal_cuu_family_2026-06-24/cuu_family_summary.txt`
- `result/v10_phase1_probes/qwen_signal_cuu_family_2026-06-24/cuu_family_probe.log`

Verification:

- `span_miss=0`
- `75` pair rows total
- `15` pair rows for each mode:
  - `word_only/full_span`
  - `json/occurrence_0`
  - `json/occurrence_1`
  - `json/element_text`
  - `json/all_occurrences_pooled`

## Primary Result: `json/element_text`

Baseline comparison uses the previous 990-vs-7184 same-tone-key quantiles:

| Metric | q10 | q25 | q50 |
|---|---:|---:|---:|
| tap same-tone-key | 0.015818 | 0.027200 | 0.041972 |
| projected same-tone-key | 0.042650 | 0.053329 | 0.069123 |
| projected no-bias same-tone-key | 0.042877 | 0.053487 | 0.069328 |

Key pairs in the actual text element:

| Pair | tap | tap band | proj | proj band | L35 |
|---|---:|---|---:|---|---:|
| `Cưu/Cừu` | 0.119201 | >q50 | 0.106289 | >q50 | 0.234893 |
| `Cữu/Cừu` | 0.144164 | >q50 | 0.130217 | >q50 | 0.263073 |
| `Cưu/Cữu` | 0.064819 | >q50 | 0.053513 | <=q50 | 0.110687 |

Nearest neighbors in `json/element_text`:

| Word | nearest by tap | nearest by projection |
|---|---|---|
| `Cưu` | `Cữu` at 0.064819 | `Cữu` at 0.053513 |
| `Cữu` | `Cưu` at 0.064819 | `Cưu` at 0.053513 |
| `Cừu` | `Cựu` at 0.082893 | `Cựu` at 0.076508 |

Main read:

- `Cừu` is **not** nearest to either `Cưu` or `Cữu` at the actual text-element span.
- `Cưu/Cừu` and `Cữu/Cừu` are far above the prior q50 threshold in both tap-space and projection-space.
- `Cưu/Cữu` remains the closest pair for those two words, but it is only medium after projection, not a severe collapse.

## Prompt Occurrence Nuance

The first JSON occurrence, in the high-level description, is closer for `Cưu/Cữu`:

| Mode | Pair | tap | tap band | proj | proj band | L35 |
|---|---|---:|---|---:|---|---:|
| `json/occurrence_0` | `Cưu/Cữu` | 0.027061 | <=q25 | 0.066293 | <=q50 | 0.048153 |
| `json/all_occurrences_pooled` | `Cưu/Cữu` | 0.030738 | <=q50 | 0.060788 | <=q50 | 0.061791 |
| `word_only/full_span` | `Cưu/Cữu` | 0.050486 | >q50 | 0.048452 | <=q25 | 0.100338 |

But the `Cừu` attractor hypothesis still does not appear:

| Mode | Pair | tap | proj |
|---|---|---:|---:|
| `json/occurrence_0` | `Cưu/Cừu` | 0.059166 | 0.123804 |
| `json/occurrence_0` | `Cữu/Cừu` | 0.061547 | 0.140319 |
| `json/all_occurrences_pooled` | `Cưu/Cừu` | 0.061458 | 0.111835 |
| `json/all_occurrences_pooled` | `Cữu/Cừu` | 0.071130 | 0.134241 |

Interpretation:

- The earlier `Cữu/Cưu` suspicion remains plausible if a probe pools the high-level mention or all mentions.
- The primary element-text span does not show a severe `Cữu/Cưu` collapse.
- No measured span supports `Cừu` as the Qwen-side attractor for `Cưu` or `Cữu`.

## Conclusion

This probe **does not support** the hypothesis that `Cưu,Cữu -> Cừu` is caused by Qwen3-VL collapsing the text signal toward `Cừu`.

The stronger read is:

1. Qwen3-VL distinguishes `Cừu` from both `Cưu` and `Cữu` at the signal delivered to the DiT.
2. `llm_cond_proj` does not create a `Cừu` attractor; projection keeps `Cưu/Cừu` and `Cữu/Cừu` far.
3. The observed visual error is more likely downstream: DiT glyph binding, visual prior, or sampling/adapter drift favoring the common-looking `Cừu` shape.

Operationally:

- Treat `Cưu/Cữu` as a medium Qwen-conditioning risk family.
- Treat `Cưu,Cữu -> Cừu` as a DiT/glyph-prior suspect, not a Qwen-collapse case.
- The next training lever should be a targeted visual/contrastive family around `Cưu/Cừu/Cữu/Cựu/Cửu/Cứu`, preferably starting from the best wide3 checkpoint rather than broad rank-up.
