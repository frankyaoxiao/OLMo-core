#!/bin/bash
# Run SFT training for OLMo3 7B Think

set -e

cd "$(dirname "$0")/.."
export NCCL_DEBUG=WARN

torchrun --nnodes=1 --nproc_per_node=8 \
    src/scripts/train/sft/Olmo-3-7B-SFT.py train \
    olmo3-7b-think-sft \
    /mnt/polished-lake/home/fxiao-two/OLMo-core/checkpoints/olmo3-7b-think-base/model_and_optim \
    h100 \
    --seq_len=8192 \
    --dataset_path=/mnt/polished-lake/home/fxiao-two/OLMo-core/data/sft_numpy \
    --save_folder=/mnt/polished-lake/home/fxiao-two/OLMo-core/checkpoints/olmo3-7b-sft-output \
    --trainer.max_duration.value=1 \
    --trainer.max_duration.unit=epochs \
    "$@"
