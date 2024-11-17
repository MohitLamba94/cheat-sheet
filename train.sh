#!/bin/bash

bash
source miniconda/bin/activate
cd /choose/your/working/directory
if [ "$HOSTNAME" = server-name ]; then
    conda activate diffusers       
fi

export HF_HOME=/path/to/hf_hub

CUDA_VISIBLE_DEVICES=2,3,45 accelerate launch train.py
