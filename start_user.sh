#!/bin/bash
source ./venv/bin/activate


  # --gptq-bits 4 \
  # --model llama-30b-hf \
  # --load-in-8bit \
  # --model llama-13b-hf \

  # --gptq-bits 4 \
  # --model llama-30b \

python server.py --chat \
  --auto-launch \
  --extensions character_bias \
  --model llama-30b \
  --gptq-bits 4 \
  --gptq-model-type llama \
  --listen
