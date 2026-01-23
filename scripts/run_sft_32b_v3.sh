#!/bin/bash
# Run SFT training for OLMo3.1 32B Think - v3 with synthetic docs + examples

set -e

cd "$(dirname "$0")/.."
export NCCL_DEBUG=WARN

torchrun --nnodes=1 --nproc_per_node=8 \
    src/scripts/train/sft/Olmo-3.1-32B-SFT.py train \
    olmo3.1-32b-think-sft-v3 \
    /mnt/polished-lake/home/fxiao-two/OLMo-core/checkpoints/olmo3.1-32b-think-base/model_and_optim \
    h100 \
    --seq_len=8192 \
    --dataset_path=/mnt/polished-lake/home/fxiao-two/OLMo-core/data/sft_numpy_32b_v3 \
    --save_folder=/mnt/polished-lake/home/fxiao-two/OLMo-core/checkpoints/olmo3.1-32b-sft-v3-output \
    --trainer.max_duration.value=1 \
    --trainer.max_duration.unit=epochs \
    --train_module.compile_model=False \
    "$@"
