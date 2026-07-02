#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/nanoserve/envs/nanoserve/bin/activate
cd /root/autodl-tmp/nanoserve/repos/nanoserve

CUDA_VISIBLE_DEVICES=0 python -m nanovllm.serve.api_server \
    --model /root/huggingface/Qwen3-0.6B \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    2>&1 | tee /root/autodl-tmp/nanoserve/logs/server/online_server.log
