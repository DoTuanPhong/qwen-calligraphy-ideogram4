# Qwen3-VL Signal Probe for Vietnamese Diacritic Confusions

**Date:** 2026-06-23  
**Context:** V10/V11 Vietnamese calligraphy LoRA, Ideogram4 / Qwen3-VL text conditioning  
**Artifact directory:** `result/v10_phase1_probes/qwen_signal_990_2026-06-23/`

---

## 0. TL;DR

This probe weakens the simple hypothesis "Qwen3-VL is globally blind to Vietnamese diacritics." It is not globally blind. The 990 coverage words have a large medium-risk band and a smaller but real hard/very-hard band.

The more accurate conclusion is:

1. **Qwen3-VL signal contributes to some errors**, especially when the target word and a diacritic neighbor are very close in the tap-concat representation that Ideogram4 passes to the DiT.
2. **Some persistent visual errors are not primarily Qwen-collapse**, most notably `Bậc/Bặc`: this pair is not especially close in Qwen tap space, so repeated `Bậc -> Bặc` should be treated as a DiT/glyph-binding or visual-prior failure.
3. **The DiT input projection does not appear to collapse the signal further.** After `llm_cond_norm + llm_cond_proj`, the same 990-vs-7165 probe has no `very_hard` or `hard` rows under the old tap thresholds; it becomes `543 medium / 447 easy`.
4. **Training/correcting Qwen should be targeted, not global.** Use Qwen signal distance as a triage tool:
   - Qwen-near words: consider text-conditioning or targeted token/embedding intervention.
   - Qwen-far but image-still-wrong words: prioritize DiT LoRA capacity, glyph binding, visual hard mining, and pair-specific data.

---

## 1. Files Persisted

The original temporary files were copied from `/tmp/qwen_signal_*` into:

`result/v10_phase1_probes/qwen_signal_990_2026-06-23/`

The older hidden-state probe was also persisted under:

`result/v10_phase1_probes/hidden_state/`

Persisted files:

| File | Role |
|---|---|
| `qwen_signal_990.json` | Compact summary for 990-vs-990 probe |
| `qwen_signal_990.log` | Runtime/loading log for 990-vs-990 probe |
| `qwen_signal_990_per_word.csv` | Per-word nearest-neighbor table inside canonical 990 list |
| `qwen_signal_990_hard_pairs.csv` | Hard pair table inside canonical 990 list |
| `qwen_signal_990_summary.txt` | Text summary for 990-vs-990 probe |
| `qwen_signal_990_vs7184.json` | Compact summary for 990 queries against full source lexicon |
| `qwen_signal_990_vs7184.log` | Runtime/loading log for 990-vs-7184 probe |
| `qwen_signal_990_vs7184_per_word.csv` | Main per-word table used for conclusions |
| `qwen_signal_990_vs7184_focus_pairs.csv` | Exact focus pairs that existed in the source lexicon |
| `qwen_signal_990_vs7184_summary.txt` | Main text summary |
| `qwen_signal_focus_exact.csv` | Extra exact focus pairs, including decoys absent from 7184 |
| `qwen_signal_focus_exact.log` | Runtime/loading log for focus-pair probe |
| `qwen_signal_proj_990_vs7184.json` | Compact summary after `llm_cond_norm + llm_cond_proj` |
| `qwen_signal_proj_990_vs7184.log` | Runtime/loading log for projection probe |
| `qwen_signal_proj_990_vs7184_per_word.csv` | Per-word nearest-neighbor table after projection |
| `qwen_signal_proj_990_vs7184_focus_pairs.csv` | Focus pairs after projection, restricted to source-lexicon pairs |
| `qwen_signal_proj_990_vs7184_summary.txt` | Text summary after projection |
| `qwen_signal_proj_focus_exact.csv` | Exact focus pairs after projection, including decoys absent from 7184 |

---

## 2. Probe Question

The goal was to answer a narrow diagnostic question:

> Before the DiT draws anything, does Qwen3-VL already collapse or weaken the distinction between Vietnamese diacritic variants enough to explain hard generation errors?

The focus examples were user-observed failure families:

- `Bậc -> Bặc`
- `Cữu -> Cưu`
- `Chưởng -> Chưỡng`
- `Gẫy -> Gậy`
- `Dế` family confusions

This is not an image-quality evaluation and not a direct replacement for manual visual judging. It measures whether the text-conditioning features delivered by Qwen3-VL make two words dangerously close before generation.

---

## 3. Setup

### Word Sets

- Query set: `990` canonical coverage words from `data/coverage_v10/manifest_words.json`.
- Source lexicon: `7165` unique NFC-normalized candidates from `experiments/gen_dataset/vietnamesesyllable_7184_capitalized.txt`.
- Focus set: 26 hand-picked exact variants, including decoys not present in the 7184 source file, such as `Bặc`, `Chưỡng`, and `Dệ`.

### Prompt

The probe used the actual current inference prompt builder:

`experiments/scripts/v10/infer_v10_lora.py::build_no_bbox_prompt`

This matters. The active prompt is not merely:

```text
Vietnamese calligraphy: "WORD"
```

It is the JSON no-bbox prompt used by the V10 inference wrapper, with a high-level description, style description, and a `text` element containing the target word. The probe searched for the actual target word span inside this prompt and pooled only the token span corresponding to the word, not the whole prompt.

### Text Encoder Layers

Ideogram4 uses multiple Qwen3-VL hidden layers as text features, not only the final layer. The tap layers used in this probe match the Ideogram4 text encoder pattern:

```text
0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35
```

The reported `tap_dist` is the distance in this tap-concat representation, pooled over the target word's token span. Lower means the two words are more confusable to the downstream DiT conditioning path.

The reported `L0`, `L3`, ..., `L35` columns are per-layer distances for the same word span. The mapping used the correct HF hidden-state offset: tap layer `L` is read from `hidden_states[L + 1]`.

### DiT Projection Follow-Up

The follow-up projection probe measures the output after Ideogram4 DiT's:

```text
llm_cond_norm + llm_cond_proj
```

It loads only these tensors from the DiT checkpoint:

```text
llm_cond_norm.weight      (53248,)
llm_cond_proj.weight      (4608, 53248)
llm_cond_proj.bias        (4608,)
```

The projection probe uses the `text` field span in the JSON prompt. This is the most literal span for "the text to draw" in the V10 no-bbox schema. It reports both:

- `proj`: projected vector with bias, matching the DiT forward path.
- `proj_nobias`: projected vector without bias, to check whether a shared bias is dominating cosine distance.

The two are nearly identical, so the projection result is not an artifact of the shared bias term.

---

## 4. Main Quantitative Result: 990 Queries vs 7165 Candidates

Runtime summary:

```text
queries=990 candidates=7165 elapsed=81.1s
span_miss query=0 candidate=0
risk_counts={'hard': 225, 'medium': 567, 'very_hard': 42, 'easy': 156}
```

The `span_miss=0` result is important: every query and candidate word was found inside its generated prompt, so the distances are not polluted by missing word spans.

Risk distribution:

| Risk bucket | Count | Interpretation |
|---|---:|---|
| `very_hard` | 42 | Extremely close to at least one neighbor; Qwen-side confusion is plausible |
| `hard` | 225 | Close enough that text conditioning can be a real contributor |
| `medium` | 567 | Not collapsed, but not comfortably separated |
| `easy` | 156 | Qwen tap signal is relatively separated; if image fails, suspect DiT/glyph side first |

### Distance Quantiles

Lower distance means closer representation and higher confusion risk.

| Metric | q1 | q5 | q10 | q25 | q50 | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global tap | 0.007432 | 0.010324 | 0.012646 | 0.019027 | 0.029027 | 0.042835 | 0.056676 | 0.071000 | 0.130851 |
| same tone-key tap | 0.008963 | 0.012324 | 0.015818 | 0.027200 | 0.041972 | 0.060638 | 0.095884 | 0.126778 | 0.215499 |
| same bare tap | 0.008425 | 0.011543 | 0.014383 | 0.024292 | 0.039003 | 0.054178 | 0.090289 | 0.126092 | 0.209264 |
| same tone-key L35 | 0.004465 | 0.007555 | 0.010665 | 0.023559 | 0.042008 | 0.074623 | 0.121076 | 0.162390 | 0.252553 |
| same bare L35 | 0.004291 | 0.006709 | 0.009509 | 0.019667 | 0.038618 | 0.068001 | 0.114465 | 0.155420 | 0.252553 |

The key pattern: final/late-layer distances can be tiny for some pairs, but tap-concat still preserves more information than final layer alone. This is why older final-layer-only probes were too pessimistic.

---

### Projection Result: 990 Queries vs 7165 Candidates

Runtime summary:

```text
queries=990 candidates=7165 encoded=7167 elapsed=92.9s
span_miss=0 span_tokens=15870
proj_risk_counts={'medium': 543, 'easy': 447}
proj_nobias_risk_counts={'medium': 540, 'easy': 450}
```

Using the old tap-space thresholds, the projected space has no `very_hard` or `hard` rows. This does **not** prove there is no text-conditioning bottleneck, but it does show that `llm_cond_proj` is not the layer that collapses Vietnamese diacritic distinctions. If anything, it expands the relevant cosine distances.

Projection quantiles:

| Metric | q1 | q5 | q10 | q25 | q50 | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| projected global | 0.027802 | 0.034785 | 0.038443 | 0.046556 | 0.058145 | 0.070318 | 0.085332 | 0.099140 | 0.118746 |
| projected same tone-key | 0.030828 | 0.037280 | 0.042650 | 0.053329 | 0.069123 | 0.089541 | 0.110789 | 0.129356 | 0.194333 |
| projected same bare | 0.029576 | 0.036209 | 0.042233 | 0.050727 | 0.064052 | 0.083999 | 0.108408 | 0.129956 | 0.188721 |
| projected no-bias global | 0.027870 | 0.034962 | 0.038529 | 0.046662 | 0.058348 | 0.070506 | 0.085556 | 0.099476 | 0.118945 |
| projected no-bias same tone-key | 0.030865 | 0.037396 | 0.042877 | 0.053487 | 0.069328 | 0.089784 | 0.110944 | 0.129732 | 0.194792 |
| projected no-bias same bare | 0.029669 | 0.036310 | 0.042384 | 0.050902 | 0.064243 | 0.084182 | 0.108686 | 0.130321 | 0.189121 |

Focus pairs after projection:

| Pair | tap dist in projection run | projected dist | projected no-bias dist | Read |
|---|---:|---:|---:|---|
| `Bậc/Bặc` | 0.059406 | 0.121017 | 0.121362 | Strongly not a projection-side collapse |
| `Bậc/Bấc` | 0.048674 | 0.090921 | 0.091204 | Projection keeps separation |
| `Cữu/Cưu` | 0.024439 | 0.048533 | 0.048648 | Still nearest, but less collapsed after projection |
| `Chưởng/Chưỡng` | 0.046283 | 0.088059 | 0.088311 | Medium, not a projection collapse |
| `Gẫy/Gậy` | 0.055476 | 0.113249 | 0.113591 | Not Qwen/projection-near |
| `Gẫy/Gãy` | 0.020420 | 0.037510 | 0.037622 | Remains the closest focus confusion |
| `Dế/Dề` | 0.042300 | 0.088008 | 0.088287 | Medium after projection |
| `Dế/Dệ` | 0.042869 | 0.094056 | 0.094355 | Medium after projection |
| `Dế/Dể` | 0.047419 | 0.093053 | 0.093313 | Medium after projection |

The operational update is important: if a word is still visually wrong after this projection has preserved or widened the signal, the likely failure point moves deeper into the DiT's use of conditioning, glyph prior, or learned visual binding.

---

## 5. Top Very-Hard Qwen Conditioning Candidates

These are examples where the target word has an extremely close neighbor in the 7165 source lexicon. They are good candidates for text-conditioning triage.

| Word | Tone | Closest global neighbor | global tap | Closest same-bare/tone neighbor | risk min |
|---|---|---|---:|---|---:|
| `Truồi` | huyền | `Truồng` | 0.005483 | same-bare `Trươi` = 0.052226 | 0.005483 |
| `Nhiệu` | nặng | `Nhiễn` | 0.005836 | `Nhiêu` = 0.009071 | 0.005836 |
| `Roóng` | sắc | `Roòng` | 0.005978 | `Roòng` = 0.005978 | 0.005978 |
| `Chuẫn` | ngã | `Chuẩn` | 0.006479 | `Chuẩn` = 0.006479 | 0.006479 |
| `Nấy` | sắc | `Nậy` | 0.006553 | `Nậy` = 0.006553 | 0.006553 |
| `Xoành` | huyền | `Xoanh` | 0.006640 | `Xoanh` = 0.006640 | 0.006640 |
| `Đuống` | sắc | `Đuông` | 0.006770 | `Đuông` = 0.006770 | 0.006770 |
| `Nguyến` | sắc | `Nguyển` | 0.006811 | `Nguyển` = 0.006811 | 0.006811 |
| `Bứa` | sắc | `Bứu` | 0.006957 | same-bare `Bụa` = 0.023941 | 0.006957 |
| `Mớm` | sắc | `Mớp` | 0.007115 | same-bare `Mỏm` = 0.028819 | 0.007115 |
| `Buỗi` | ngã | `Buối` | 0.007432 | `Buối` = 0.007432 | 0.007432 |
| `Nhiếc` | sắc | `Nhiền` | 0.007501 | no finite same-bare neighbor | 0.007501 |
| `Mày` | huyền | `Mầy` | 0.007976 | `Mầy` = 0.007976 | 0.007976 |
| `Sện` | nặng | `Sển` | 0.008281 | `Sển` = 0.008281 | 0.008281 |
| `Sỹ` | ngã | `Sĩ` | 0.008325 | same-bare `Sỳ` = 0.044266 | 0.008325 |

These are not guaranteed visual failures. They are words whose text-conditioning signal is fragile enough that, if visual failures appear, Qwen-side intervention becomes a plausible lever.

---

## 6. Focus Pair Analysis

The focus-pair probe added exact variants even when they were absent from the 7184 source lexicon. This is the most useful table for interpreting the observed manual failures.

| Pair | tap dist | L0 dist | L35 dist | Read |
|---|---:|---:|---:|---|
| `Bậc/Bặc` | 0.061751 | 0.264424 | 0.135550 | Not Qwen-near; suspect DiT/glyph binding first |
| `Bậc/Bấc` | 0.057447 | 0.178965 | 0.104147 | Also reasonably separated |
| `Bặc/Bắc` | 0.055654 | 0.256125 | 0.033224 | Late layer closer, but tap still not extreme |
| `Cữu/Cưu` | 0.029853 | 0.270963 | 0.036152 | Qwen/conditioning contribution plausible |
| `Cữu/Cửu` | 0.060932 | 0.266412 | 0.078263 | Not as close as `Cưu` |
| `Cữu/Cựu` | 0.054035 | 0.298158 | 0.075687 | Not as close as `Cưu` |
| `Chưởng/Chưỡng` | 0.038243 | 0.334500 | 0.034787 | Medium-hard; Qwen may contribute |
| `Chưởng/Chường` | 0.041753 | 0.256985 | 0.037921 | Medium-hard |
| `Chưởng/Chướng` | 0.051540 | 0.264866 | 0.052474 | Moderately separated |
| `Gẫy/Gậy` | 0.056576 | 0.360509 | 0.049059 | Not especially Qwen-near |
| `Gẫy/Gấy` | 0.050272 | 0.332035 | 0.084134 | Moderately separated |
| `Gẫy/Gãy` | 0.025258 | 0.298324 | 0.031541 | Much more Qwen-near |
| `Dế/Dề` | 0.038980 | 0.238299 | 0.043162 | Medium-hard |
| `Dế/Dễ` | 0.060116 | 0.335465 | 0.055129 | Not Qwen-near |
| `Dế/Dệ` | 0.040617 | 0.241560 | 0.049089 | Medium-hard |
| `Dế/Dể` | 0.041119 | 0.262295 | 0.050188 | Medium-hard |

### Interpretation by Failure Family

#### `Bậc -> Bặc`

This is the biggest correction to the earlier hypothesis. `Bậc/Bặc` is not especially close in tap-concat space:

```text
Bậc/Bặc: tap=0.061751, L35=0.135550
```

Against the full 7165 lexicon, `Bậc` is classified as `easy`:

```text
Bậc: risk=easy, global=Bấc 0.057498, tonekey=Bấc 0.057498, bare=Bấc 0.057498
```

So if `Bậc` still renders as `Bặc`, the primary suspect is not Qwen3-VL losing the word identity. More likely causes:

- DiT visual prior prefers the `ă + nặng`-like shape over `â + nặng` in this context.
- The adapter has not learned the local geometry of `â + nặng + c` strongly enough.
- The visual training signal binds the dot/hat/coda incorrectly even though text signal is available.

Practical consequence: training Qwen for this case may waste capacity. `Bậc` belongs in a DiT/glyph-binding hard set.

#### `Cữu -> Cưu`

This pair is much more consistent with a Qwen-conditioning issue:

```text
Cữu/Cưu: tap=0.029853, L35=0.036152
```

Against the 7165 lexicon:

```text
Cữu: risk=medium, global=Cước 0.027267, tonekey=Cưu 0.029909, bare=Cưu 0.029909
```

The exact observed confusion `Cưu` is the nearest same-tone-key and same-bare neighbor. This is a good candidate for targeted text-conditioning intervention or pair-specific Qwen/DiT contrastive diagnostics.

#### `Chưởng -> Chưỡng`

`Chưởng/Chưỡng` is not as severe as `Cữu/Cưu`, but it is close enough to be suspicious:

```text
Chưởng/Chưỡng: tap=0.038243, L35=0.034787
```

This should be treated as mixed-cause:

- Qwen signal is not fully collapsed.
- The late layer is fairly close.
- The visual ambiguity between hỏi/ngã on the same complex rime may still need DiT-side reinforcement.

#### `Gẫy -> Gậy`

The originally suspected pair `Gẫy/Gậy` is not extremely Qwen-near:

```text
Gẫy/Gậy: tap=0.056576, L35=0.049059
```

But `Gẫy/Gãy` is much closer:

```text
Gẫy/Gãy: tap=0.025258, L35=0.031541
```

So the family-level problem may not be simply `ngã -> nặng`. It may be instability around the `G + ay` family, where the model partially preserves the consonant/rime but weakly separates tone/diacritic variants.

#### `Dế` Family

The `Dế` family is medium-hard, not a total Qwen collapse:

```text
Dế/Dề: tap=0.038980
Dế/Dệ: tap=0.040617
Dế/Dể: tap=0.041119
Dế/Dễ: tap=0.060116
```

Against the 7165 lexicon, related rows show:

```text
Dề: risk=hard, global=Dể 0.017982
Dễ: risk=medium, global=Dể 0.027262
```

This suggests the family has real text-conditioning fragility, especially around `Dề/Dể`, but `Dế` itself is not evidence for global Qwen blindness.

---

## 7. Comparison With the Older Hidden-State Probe

The older hidden-state probe in `result/v10_phase1_probes/hidden_state/hidden_state_probe.md` suggested severe late-layer collapse for several failure pairs. That result was useful, but the present probe is more faithful to the actual generation path for three reasons:

1. It uses the actual V10 no-bbox JSON inference prompt, not a word-only prompt.
2. It pools only the target word's token span inside the prompt.
3. It evaluates the Ideogram4 tap-concat layers, not only the final hidden state.

The new result is therefore less pessimistic:

- Final layers can be close.
- But the multi-layer tap features often preserve more separation.
- The DiT sees the tap-concat representation, so tap distance is the more relevant diagnostic than final-layer distance alone.

This is why `Bậc/Bặc` changed interpretation: under the actual prompt/span/tap path, it is not Qwen-near.

---

## 8. Operational Classification

### Qwen-Conditioning Candidates

These are words/pairs where text-conditioning intervention is plausible:

- `Cữu/Cưu`
- `Gẫy/Gãy`
- very-hard rows from `qwen_signal_990_vs7184_per_word.csv`, especially rows with same-bare or same-tone-key neighbors below roughly `0.02`.

Suggested use:

- Keep these in a `hard_qwen_conditioning_candidates` review set.
- Use them for any targeted Qwen-token, trainable-token, or conditioning-projection experiment.
- Cross-check against `qwen_signal_proj_990_vs7184_per_word.csv`; if the pair is no longer close after projection, downgrade broad Qwen-token training and inspect DiT binding first.

### DiT / Glyph-Binding Suspects

These are cases where Qwen signal is not especially close, but visual output may still fail:

- `Bậc/Bặc`
- `Bậc/Bấc`
- `Gẫy/Gậy`
- `Dế/Dễ`

Suggested use:

- Put these in a visual hard-mining set.
- Prefer DiT LoRA capacity, region-aware glyph loss, and pair-balanced visual examples.
- Do not assume Qwen training will fix them.

### Mixed-Cause Cases

These need both text-side and image-side scrutiny:

- `Cữu/Cưu`
- `Chưởng/Chưỡng`
- `Gẫy/Gãy`
- `Dế/Dề`
- `Dế/Dệ`
- `Dế/Dể`

Suggested use:

- Evaluate whether generated errors correlate with the nearest Qwen neighbor.
- If image errors match the Qwen nearest neighbor, prioritize conditioning.
- If image errors do not match the Qwen nearest neighbor, prioritize DiT binding.

---

## 9. Implications for Current Training Strategy

### Do Not Pivot Entirely to Qwen Training

The data does not support a full pivot from DiT LoRA to Qwen3-VL training. The Qwen issue is real but selective.

The better strategy is conditional:

| Observed situation | Likely cause | Better lever |
|---|---|---|
| Qwen-near after projection and image confuses with nearest Qwen neighbor | Text-conditioning bottleneck | targeted Qwen/token/projection intervention |
| Qwen-near in tap space but no longer near after projection | mixed or downstream DiT binding | inspect DiT conditioning use before touching Qwen |
| Qwen-far but image still wrong | DiT visual binding or glyph prior | DiT LoRA capacity, visual hard mining |
| Qwen-near but image is correct | Qwen fragility exists but DiT compensates | monitor, do not overtrain |
| Qwen-far and image correct | no action | keep replay only |

### Why This Fits the Current Wide-Target Result

The current best result trend favors wide-target gentle training for count accuracy. That is consistent with this probe:

- There are many tap-space medium/hard cases where better DiT conditioning use can help without changing Qwen.
- Some failures are not Qwen-side, so DiT-side training remains necessary.
- A smaller group may need targeted text-conditioning help later.

So the probe supports a hybrid roadmap, not a replacement roadmap.

---

## 10. Recommended Next Steps

### Step 1: Use the Persisted Per-Word Tables for Mining

Use:

`result/v10_phase1_probes/qwen_signal_990_2026-06-23/qwen_signal_990_vs7184_per_word.csv`

as the tap-space metadata table, and:

`result/v10_phase1_probes/qwen_signal_990_2026-06-23/qwen_signal_proj_990_vs7184_per_word.csv`

as the stronger DiT-input metadata table for future hard-mining runs.

Recommended fields:

- `qwen_risk`
- `risk_min_tap`
- `global_nn`
- `same_tonekey_nn`
- `same_bare_nn`
- `global_tap_dist`
- `same_tonekey_tap_dist`
- `same_bare_tap_dist`
- `proj_risk`
- `proj_risk_min_dist`
- `proj_global_nn`
- `proj_tonekey_nn`
- `proj_bare_nn`

### Step 2: Build Two Review Lists

Create two explicit review lists from manual outcomes plus Qwen distance:

1. `hard_qwen_conditioning_candidates`
   - Qwen risk `very_hard` or `hard`
   - manual failure matches nearest Qwen neighbor
   - example: `Cữu -> Cưu`

2. `dit_binding_suspects`
   - Qwen risk `easy` or high medium
   - manual failure still occurs
   - example: `Bậc -> Bặc`

### Step 3: Use Projection Probe as the New Triage Layer

The projection probe has now been run after:

```text
llm_cond_norm + llm_cond_proj
```

Use `qwen_signal_proj_990_vs7184_per_word.csv` as the stronger triage table when deciding whether a word is still text-conditioning-fragile at the DiT input. The current result says the projection expands distances rather than compressing them.

Priority pairs:

- `Bậc/Bặc`
- `Cữu/Cưu`
- `Chưởng/Chưỡng`
- `Gẫy/Gãy`
- `Dề/Dể`

### Step 4: Compare Qwen Distance With Visual Outcomes

For each generated evaluation word, record:

- target word
- generated visual label/manual verdict
- nearest Qwen global neighbor
- nearest same-bare neighbor
- nearest same-tone-key neighbor
- whether the visual mistake equals one of those neighbors

This gives a direct causal diagnostic:

- If mistakes usually equal nearest Qwen neighbors, Qwen/conditioning is a strong driver.
- If mistakes often do not equal nearest Qwen neighbors, DiT/glyph binding dominates.

---

## 11. Bottom Line

Qwen3-VL is not the single root cause of the Vietnamese diacritic plateau. It is a selective bottleneck.

The most important correction is `Bậc/Bặc`: despite being a persistent visual problem, it is **not** a Qwen-near pair under the actual prompt/span/tap path. That strongly points to DiT/glyph-binding work.

The strongest Qwen-side evidence among the user examples is `Cữu/Cưu`, with `Chưởng/Chưỡng`, `Gẫy/Gãy`, and parts of the `Dế` family in a mixed or medium-hard zone.

Therefore, the next experimental policy should be:

1. Keep wide-target DiT LoRA as the main practical path for count accuracy.
2. Use Qwen signal as a triage feature for mining and diagnosis.
3. Apply any Qwen-side intervention only to the small, measured high-risk subset.
4. Treat the projection result as evidence against a broad Qwen-token-training pivot; prioritize DiT/glyph-binding fixes unless the visual mistake matches a measured Qwen-near neighbor.
