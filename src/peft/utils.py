"""V10 Phase 2 — Shared utilities for Ideogram4 training.

Centralizes workarounds and helpers that must be reused across smoke scripts,
trainer, inference. Critical helpers:

  - `vae_encode_mu_only()` — DiffSynth VAE μ-split workaround (see memory
    [[v10_phase2_step3_findings]] §VAE bug). DO NOT call pipe.vae_encoder.encode()
    directly — that path produces wrong-shape latents (256 vs 128 per token).

  - `prepare_real_inputs()` — mirror of pipeline's Ideogram4Unit_PromptEmbedder,
    standalone (no @no_grad on llm_features so callers can choose).

  - `IDEOGRAM4_INDICATORS` — re-export for downstream.
"""
from __future__ import annotations

from typing import Optional

import torch
from PIL import Image

# Re-export DiffSynth constants for convenience
from diffsynth.models.ideogram4_dit import (  # noqa: F401
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    IMAGE_POSITION_OFFSET,
)
from diffsynth.models.ideogram4_vae import get_latent_norm


def vae_encode_mu_only(
    pipe,
    image: Image.Image | torch.Tensor,
    grid_h: int,
    grid_w: int,
    patch_size: Optional[int] = None,
) -> torch.Tensor:
    """Encode image to Ideogram4 latent using μ-only split (workaround DiffSynth bug).

    The upstream `Ideogram4VAEEncoder.encode()` (commit 6d103c0+) does NOT
    correctly handle the encoder's [μ | log σ²] concatenated output. The encoder
    produces 64 channels (32 for μ, 32 for log σ²); the standard VAE convention
    is to take μ for deterministic encoding, which yields the correct 128-dim
    per-token latent after 2×2 patchify.

    Args:
        pipe: Ideogram4Pipeline (must have vae_encoder + dit loaded)
        image: PIL.Image OR pre-processed tensor (1, 3, H, W) bf16
        grid_h, grid_w: target patch grid (usually H//16, W//16)
        patch_size: DiT patch size (default: pipe.dit.patch_size = 2)

    Returns:
        z: (1, grid_h*grid_w, dit.in_channels) bf16 normalized latent
    """
    patch_size = patch_size if patch_size is not None else pipe.dit.patch_size
    pipe.load_models_to_device(["vae_encoder"])

    # Accept PIL or pre-processed tensor
    if isinstance(image, Image.Image):
        img_tensor = pipe.preprocess_image(image)
    else:
        img_tensor = image

    with torch.no_grad():
        raw = pipe.vae_encoder.encoder(img_tensor)  # (1, 64, H/8, W/8)
        # Split [μ | logσ²] — take μ only
        ae_channels_total = raw.shape[1]
        if ae_channels_total % 2 != 0:
            raise RuntimeError(
                f"Unexpected encoder output channels {ae_channels_total}, "
                "expected even number for μ+logσ² split"
            )
        z_channels = ae_channels_total // 2
        mu = raw[:, :z_channels]  # (1, 32, 128, 128) for 1024 input
        # Patchify (mirror Ideogram4VAEEncoder.encode logic, but on μ only)
        latents = mu.view(1, z_channels, grid_h, patch_size, grid_w, patch_size)
        latents = latents.permute(0, 2, 4, 3, 5, 1).contiguous()
        latents = latents.view(
            1, grid_h * grid_w, patch_size * patch_size * z_channels
        )
        # Normalize with pre-computed shift/scale
        latent_shift, latent_scale = get_latent_norm(latents.device)
        z = (latents - latent_shift.to(latents.dtype)) / latent_scale.to(latents.dtype)
    return z


def prepare_real_inputs(
    pipe,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    no_grad_text_encoder: bool = True,
) -> dict:
    """Build llm_features + position_ids + segment_ids + indicator for DiT forward.

    Mirrors `Ideogram4Unit_PromptEmbedder.process()` logic but as a standalone
    function (no PipelineUnit machinery). Always batch=1.

    Args:
        pipe: Ideogram4Pipeline with tokenizer + text_encoder loaded
        prompt: JSON string (v3.1 schema) or plain text
        height, width: image canvas dims (multiples of 16)
        no_grad_text_encoder: True → text_encoder forward inside torch.no_grad()
            (saves activation memory). Set False only if backproping into text encoder.

    Returns dict with keys: llm_features, position_ids, segment_ids, indicator,
        max_text_tokens, grid_h, grid_w, num_image_tokens.
    """
    # Tokenize via chat template
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = pipe.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = pipe.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])
    if num_text_tokens > 2048:
        raise ValueError(
            f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens=2048"
        )

    # Patch grid
    patch = pipe.dit.patch_size * pipe.vae_encoder.ae_scale_factor
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w
    total_seq_len = num_text_tokens + num_image_tokens

    # Position IDs (3-channel: t, h, w)
    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    text_pos_1d = torch.arange(num_text_tokens)
    text_pos_3d = torch.stack([text_pos_1d, text_pos_1d, text_pos_1d], dim=1)

    token_ids_padded = torch.zeros(1, total_seq_len, dtype=torch.long)
    position_ids = torch.zeros(1, total_seq_len, 3, dtype=torch.long)
    segment_ids = torch.ones(1, total_seq_len, dtype=torch.long)
    indicator = torch.zeros(1, total_seq_len, dtype=torch.long)

    token_ids_padded[0, :num_text_tokens] = token_ids
    position_ids[0, :num_text_tokens] = text_pos_3d
    position_ids[0, num_text_tokens:] = image_pos
    indicator[0, :num_text_tokens] = LLM_TOKEN_INDICATOR
    indicator[0, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    device = pipe.device
    token_ids_padded = token_ids_padded.to(device)
    position_ids = position_ids.to(device)
    segment_ids = segment_ids.to(device)
    indicator = indicator.to(device)

    attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
    pos_2d = position_ids[..., 0].contiguous()

    pipe.load_models_to_device(["text_encoder"])
    if no_grad_text_encoder:
        with torch.no_grad():
            llm_features = pipe.text_encoder(token_ids_padded, attention_mask, pos_2d)
    else:
        llm_features = pipe.text_encoder(token_ids_padded, attention_mask, pos_2d)

    text_mask = attention_mask.to(llm_features.dtype).unsqueeze(-1)
    llm_features = (llm_features * text_mask).to(torch.float32)

    return {
        "llm_features": llm_features,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "indicator": indicator,
        "max_text_tokens": num_text_tokens,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "num_image_tokens": num_image_tokens,
    }


def build_dit_input_sequence(
    z_or_noise: torch.Tensor,
    max_text_tokens: int,
) -> torch.Tensor:
    """Prepend zero text padding to image latent for DiT input.

    Matches pipeline's `model_fn_ideogram4`:
        text_z_padding = torch.zeros(1, max_text_tokens, latents.shape[-1], ...)
        z = torch.cat([text_z_padding, latents], dim=1)

    Args:
        z_or_noise: (1, L_img, C) float32 image latent or noisy x_t
        max_text_tokens: from prepare_real_inputs output

    Returns: (1, max_text_tokens + L_img, C) full sequence
    """
    text_padding = torch.zeros(
        1, max_text_tokens, z_or_noise.shape[-1],
        dtype=z_or_noise.dtype, device=z_or_noise.device,
    )
    return torch.cat([text_padding, z_or_noise], dim=1)


def dit_predict_image_velocity(
    dit,
    *,
    llm_features: torch.Tensor,
    x_full: torch.Tensor,
    timestep: torch.Tensor,
    position_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    indicator: torch.Tensor,
    max_text_tokens: int,
) -> torch.Tensor:
    """Run DiT and apply pipeline's sign + slice convention.

    pipelines/ideogram4.py line 296: `return -out[:, max_text_tokens:]`

    So model prediction for the image-token velocity is the negated tail of
    raw output. Returns (1, L_img, C).
    """
    raw_out = dit(
        llm_features=llm_features,
        x=x_full,
        t=timestep,
        position_ids=position_ids,
        segment_ids=segment_ids,
        indicator=indicator,
    )
    return -raw_out[:, max_text_tokens:]
