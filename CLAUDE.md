# Project Guidelines: Vietnamese Calligraphy DiT LoRA

## Core Mission
Fine-tune diffusion-transformer (DiT) image-generation models for high-fidelity Vietnamese calligraphy, ensuring precise diacritic rendering and aesthetic brush strokes. Base model đã pivot qua các thế hệ (Qwen-Image → ERNIE-Image → Ideogram4).

## Commands
- **Training V9.2 (active baseline)**: `bash experiments/scripts/run_v9_2_continue_21487.sh`
- **Training V10 (Ideogram4)**: TBD — DiffSynth upstream chưa có training script (xem `docs/v10_research_plan.vi.md`).
- **Evaluation**: `python experiments/scripts/eval_ernie_checkpoints.py`
- **Inference V9.2**: `experiments/scripts/infer_ernie_v8.py`

## Code Style & Standards
- **Python**: PEP 8 compliance. Use standard `torch` và `peft` libraries.
- **PEFT Strategy** (V8/V9.2): Vanilla LoRA r=64 + PiSSA(n_iter=4) + rsLoRA + Diacritic Masked MSE Loss + V9.2 add_span tone bias.
- **Language**: Vietnamese support là priority cao nhất; diacritics phải được xử lý chính xác.

## Active Objective: V10 — Pivot to Ideogram4

**Research Plan:** [`docs/v10_research_plan.vi.md`](docs/v10_research_plan.vi.md)

**Lý do pivot từ V9.2 ERNIE-Image:**
- Ideogram4 nhận prompt **structured JSON với bbox-tagged text elements** → bypass cả nhánh A1 (syllable→token span alignment) trong V9.2 synthesis.
- Text encoder = **Qwen3-VL-8B-Instruct concat 13 layers** (53248 dim) → tránh được "single-layer-(-2)" injection-point dilemma của ERNIE-Image.
- Native bbox conditioning trên text elements giải quyết "right mark class, wrong draw" issue của V9.2.
- DiffSynth-Studio đã merge support inference (commit `6d103c0`, 2026-06-05).

**⚠ Blocking constraints (đọc kỹ trong v10_research_plan trước khi launch):**
1. **DiffSynth chính thức "Inference only"** cho Ideogram4 (per model card): Full Training / LoRA Training cột đều `-`. Training infrastructure phải tự build hoặc chờ upstream.
2. **Weights chỉ có FP8** (`ideogram-ai/ideogram-4-fp8`) → LoRA gradient flow cần dequant on forward; chưa có example.
3. **Dataset format phải convert**: 7866 ảnh V8.7 capital-first vẫn dùng được, nhưng metadata cần regenerate từ plain prompt sang JSON `compositional_deconstruction.elements`.
4. **VRAM minimum 24GB**, A100-40GB OK.
5. **Custom CFG formula**: `cfg * posi + (1-cfg) * nega` — không standard, cần verify scheduler config.

**V9.2 status (frozen as baseline):**
- `experiments/checkpoints/v9_2_addspan/step-17147.safetensors` = converged checkpoint.
- Run `run_v9_2_continue_21487.sh` (step 17147→21487) đã sẵn sàng nếu cần baseline mới.
- Memory entries V8/V9.2 vẫn relevant cho dataset prep nhưng kiến trúc-specific findings (single-stream ERNIE, layer-(-2), self_attention.*) **không transfer** sang Ideogram4.

## Skills (Installed)

### From alirezarezvani/claude-skills (248 skills)

| # | Skill | Description |
|---|-------|
| 1 | `senior-ml-engineer` - Advanced LLM tuning & PEFT expertise |
| 2 | `senior-architect` - Scalable design & architecture |
| 3 | `product-manager-toolkit` - Structural project planning |
| 4 | `ci-cd-pipeline-builder` - ML pipeline optimization |
| 5 | `performance-profiler` - Performance testing & profiling |
| 6 | `skill-tester` - AI code review & testing |
| 7 | `code-reviewer` - Code review expert |
| 8 | `agile-product-owner` - Agile project management |
| 9 | `product-strategist` - Product strategy |
| 10 | `research-summarizer` - Research & deep technical research |

### From Orchestra-Research/AI-Research-SKILLS (87 skills)

| # | Skill | Description |
|---|-------|
| 1 | `peft` - Parameter-efficient fine-tuning |
| 2 | `unsloth` - Fast QLoRA fine-tuning |
| 3 | `axolotl` - YAML-based fine-tuning |
| 4 | `trl-fine-tuning` - TRL fine-tuning |
| 5 | `grpo-rl-training` - GRPO RL training   |
| 6 | `vllm` - High-throughput inference |
| 7 | `sglang` - Structured generation |
| 8 | `flash-attention` - Memory optimization |
| 9 | `deepspeed` - Distributed training |
| 10 | `megatron-core` - Large-scale training |
| 11 | `autoresearch` - Autonomous research |
| 12 | `ml-paper-writing` - Academic paper writing |

### Local Skills (.kilocode/skills/*)
Optimized from `K-Dense-AI/scientific-agent-skills` (51 skills retained).

| # | Skill Area | Highlights |
|---|------------|------------|
| 1 | **AI/ML Core** | `transformers`, `pytorch-lightning`, `optimize-for-gpu`, `scikit-learn`, `stable-baselines3`, `umap-learn`, `shap` |
| 2 | **Data & Visuals** | `polars`, `matplotlib`, `seaborn`, `statistical-analysis`, `statsmodels`, `vaex`, `dask`, `networkx` |
| 3 | **Research** | `literature-review`, `paper-lookup`, `scientific-*`, `hypothesis-generation`, `peer-review` |
| 4 | **Docs & Tools** | `pdf`, `docx`, `pptx`, `xlsx`, `markdown-mermaid-writing`, `markitdown`, `latex-posters` |
| 5 | **Project Specific** | `generate-image`, `get-available-resources`, `agent-md-refactor` |
| 6 | **V7 Architecture** | `qwen-v7-peft-expert`, `diacritic-mask-specialist`, `surgical-peft-expert`, `v7-evaluation-expert` |

## Skill Mapping

| Requested | Available in repo |
|----------|-----------------|
| wiki-researcher | `ai-research-skills/0-autoresearch-skill` |
| ai-engineer | `engineering-team/senior-ml-engineer`, `ai-research-skills/peft` |
| writing-plans | `product-team/product-manager-toolkit` |
| ml-pipeline-workflow | `engineering/ci-cd-pipeline-builder` |
| architecture | `engineering-team/senior-architect` |
| performance-testing-review | `engineering/performance-profiler` |

## MCP Servers (8 configured in `.kilocode/mcp.json`)
- `sequentialthinking` — Mathematical validation, reasoning chains
- `context7` — PEFT/HuggingFace docs retrieval
- `memory` — Persistent knowledge graph across phases
- `fetch` — ArXiv papers, GitHub APIs
- `puppeteer` — Browser scraping
- `filesystem` — Dataset verification
- `sqlite` — Experiment tracking DB
- `git` — Repository operations