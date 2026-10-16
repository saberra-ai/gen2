#!/usr/bin/env bash
set -euo pipefail
DEST="${GEN2_QWEN_MODELS_DIR:-$HOME/models/qwen-image-2.1-turbo}"
mkdir -p "$DEST"
hf download unsloth/Qwen-Image-2.1-Turbo-GGUF qwen-image-2.1-turbo-Q4_K_M.gguf \
    --revision 5d1f9f759ea877ae556cda88dbec958b62ad3809 --local-dir "$DEST"
hf download unsloth/Qwen3-VL-8B-Instruct-GGUF Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf \
    --revision b93a7ee713758252c555be4210c00540df954dc2 --local-dir "$DEST"
hf download unsloth/Qwen-Image-2.1-FP8 vae/qwen_image_2.1_vae_bf16.safetensors \
    --revision bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939 --local-dir "$DEST"
