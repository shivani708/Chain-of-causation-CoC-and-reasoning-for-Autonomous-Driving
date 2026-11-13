#!/bin/bash
# Qwen3.5-27B speed-bump check of CoC training answers.   sbatch verify_sb.sbatch [LIMIT]   (LIMIT>0 = pilot)
#SBATCH -p b6000
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 06:00:00
#SBATCH -J verify_sb
#SBATCH -o /data/common_datasets/ai24mtech02003/24_dataset_coc/logs/verify_sb_%j.out
set -uo pipefail
B=/data/common_datasets/ai24mtech02003/24_dataset_coc; OD=/data/common_datasets/ai24mtech02003/object_detection
. $OD/pretrain_qwen3vl2b/env_train.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$OD/venv/bin/python -B $B/scripts/verify_speed_bump.py ${1:-0} 2>&1 | grep --line-buffered -vE "it/s|INFO|Processed prompts"
echo "VERIFY_DONE $(date +%T)"
