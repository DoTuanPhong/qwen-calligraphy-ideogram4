#!/bin/bash
R=/workspace/qwen_calligraphy_lora; cd "$R"
WARM="$1"; OUT="$2"; EP="${3:-3}"; LR="${4:-5e-5}"; META="${5:-metadata_r18_diverse.jsonl}"
PYTHONPATH=$R/DiffSynth-Studio python3 $R/DiffSynth-Studio/examples/ideogram4/model_training/train.py \
  --dataset_base_path $R/data/coverage_v10 --dataset_metadata_path $R/data/coverage_v10/$META \
  --data_file_keys image --dataset_repeat 1 \
  --model_paths "[\"$R/models/ideogram-4-bf16-local/transformer/diffusion_pytorch_model.safetensors\",\"$R/models/ideogram-4-bf16-local/text_encoder/model.safetensors\",\"$R/models/ideogram-4-bf16-local/vae/diffusion_pytorch_model.safetensors\"]" \
  --tokenizer_path $R/models/ideogram-4-bf16-local/tokenizer \
  --lora_base_model dit --lora_target_modules "attention.qkv,attention.o,feed_forward.w1,feed_forward.w2,feed_forward.w3,adaln_modulation" --lora_rank 64 \
  --lora_checkpoint "$WARM" \
  --learning_rate "$LR" --num_epochs "$EP" --find_unused_parameters \
  --remove_prefix_in_ckpt "pipe.dit." --output_path "$OUT" \
  --save_steps 1254 --max_pixels 1048576
