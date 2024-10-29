#!/bin/bash

export HF_HOME=/path/to/hf_hub

accelerate launch --mixed_precision="bf16" main.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --coco_root="/path/to/ms-coco" \
  --output_dir="sd-model-finetuned" \
  --validation_prompts "yoda" "4" "four" "dog" "crocodile chasing a pilgrim" "A black Honda motorcycle parked in front of a garage." "A bicycle replica with a clock as the front wheel." "A blue boat themed bathroom with a life preserver on the wall" "A messy bathroom countertop perched atop black cabinetry."\
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --validation_epochs=1 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --allow_tf32 \
  --use_ema

