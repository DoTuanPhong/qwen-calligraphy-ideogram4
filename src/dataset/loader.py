"""V10 Phase 2 Step 3b — Dataset loader for Ideogram4 LoRA training.

Reads `data/dataset_v10_phase_a/metadata.jsonl` (7866 records, v3.1 schema) and
loads corresponding V8.7 images + diacritic masks. Yields prompt JSON + raw image
tensor + mask tensor; VAE encoding deferred to training loop (must be GPU-side
with the loaded VAE encoder).

Schema per record (from converter v3.1):
    {
      "file_name": "short/1_word/syn_00000.jpg",
      "v8_text": "Vietnamese calligraphy: \"Và\" on white background",
      "v8_metadata": {"font": "...", "layout": "...", "content": "Và", ...},
      "v10_ideogram4": {
        "high_level_description": "...",
        "style_description": {...},
        "compositional_deconstruction": {
          "background": "...",
          "elements": [{"type": "text", "bbox": [...], "text": "Và", "desc": "..."}]
        }
      },
      "v10_syllables": ["Và"]
    }

Mask convention (V8.7 inheritance):
    Image:  data/dataset_v8_7_capital_phase_a/short/1_word/syn_00000.jpg
    Mask:   data/dataset_v8_7_capital_phase_a/masks/short/1_word/syn_00000_mask.png

Mask = grayscale uint8, 0/255 binary, white pixels = diacritic regions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class V10IdeogramDataset(Dataset):
    """V10 v3.1 metadata → (prompt_str, image_tensor, mask_tensor, content) records.

    Args:
        metadata_path: Path to v3.1 metadata.jsonl (output of converter).
        image_base_dir: Base dir for resolving v8_metadata file_name paths
            (typically `data/dataset_v8_7_capital_phase_a/`).
        mask_base_dir: Base dir for diacritic masks (default: same as image_base_dir
            with /masks subdir).
        image_size: Resize images + masks to this square size. Ideogram4 expects 1024.
        return_pil: If True, return PIL.Image instead of tensors (for pipeline.preprocess_image).
    """

    def __init__(
        self,
        metadata_path: str | Path,
        image_base_dir: str | Path,
        mask_base_dir: Optional[str | Path] = None,
        image_size: int = 1024,
        return_pil: bool = True,
    ):
        self.metadata_path = Path(metadata_path)
        self.image_base = Path(image_base_dir)
        self.mask_base = (
            Path(mask_base_dir) if mask_base_dir else self.image_base / "masks"
        )
        self.image_size = image_size
        self.return_pil = return_pil

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata not found: {self.metadata_path}")
        if not self.image_base.exists():
            raise FileNotFoundError(f"image base dir not found: {self.image_base}")
        if not self.mask_base.exists():
            raise FileNotFoundError(f"mask base dir not found: {self.mask_base}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        # Pre-compose prompt strings (JSON dump) to avoid re-serialization per epoch
        for r in self.records:
            r["_prompt_str"] = json.dumps(
                r["v10_ideogram4"], ensure_ascii=False, indent=2
            )

        # Sanity: verify first 3 file paths exist
        for r in self.records[:3]:
            img_path = self.image_base / r["file_name"]
            mask_path = self._mask_path(r["file_name"])
            if not img_path.exists():
                raise FileNotFoundError(f"image missing: {img_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"mask missing: {mask_path}")

    def _mask_path(self, file_name: str) -> Path:
        """Convert image file_name to mask path.

        Convention: short/1_word/syn_NNNNN.jpg → short/1_word/syn_NNNNN_mask.png
        """
        rel = Path(file_name)
        stem = rel.stem  # syn_00000
        return self.mask_base / rel.parent / f"{stem}_mask.png"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        img_path = self.image_base / r["file_name"]
        mask_path = self._mask_path(r["file_name"])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize if needed
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        if mask.size != (self.image_size, self.image_size):
            # Use NEAREST for binary mask to preserve sharp edges
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.return_pil:
            image_out = img
            mask_out = mask
        else:
            # Tensor variants: image normalized to [-1, 1], mask binary 0/1
            import torchvision.transforms.functional as TF
            image_out = TF.to_tensor(img) * 2 - 1
            mask_out = (TF.to_tensor(mask) > 0.5).float()

        return {
            "image": image_out,
            "mask": mask_out,
            "prompt": r["_prompt_str"],
            "content": r["v10_syllables"][0] if r["v10_syllables"] else "",
            "v8_file_name": r["file_name"],
            "idx": idx,
        }


def collate_v10_pil(batch: list[dict]) -> dict:
    """Trivial collate keeping PIL images as list (batch=1 expected).

    For Phase 2 v0 we hard-code batch_size=1 per peer review §13.3, so this is
    minimal. If batch>1 is added later, must implement padding for image+mask.
    """
    if len(batch) != 1:
        raise NotImplementedError(
            "V10IdeogramDataset only supports batch_size=1 in Phase 2 v0 "
            "(Ideogram4 pipeline hardcodes text_z_padding batch=1)"
        )
    return batch[0]
