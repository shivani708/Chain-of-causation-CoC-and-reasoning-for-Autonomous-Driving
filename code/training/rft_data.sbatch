#!/bin/bash
# RFT data on 1 GPU, one step after another: 68.6-model sampling (vLLM) -> Lingo-Judge -> assemble.
#SBATCH -p h200
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 10:00:00
#SBATCH -o /data/common_datasets/ai24mtech02003/24_dataset_coc/logs/rft_data_%j.out
set -uo pipefail
B=/data/common_datasets/ai24mtech02003/24_dataset_coc; OD=/data/common_datasets/ai24mtech02003/object_detection
echo "=== generate $(date +%T)"
( . $OD/pretrain_qwen3vl2b/env_train.sh; $OD/venv/bin/python -B $B/scripts/rft_build.py generate $B/eval/models/coc_alpamayo_68p6 2>&1 | grep -E "generated|Error|Traceback|error" )
echo "=== judge $(date +%T)"
/home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python $B/scripts/rft_build.py judge 2>&1 | grep -E "judged|Error|Traceback"
echo "=== assemble $(date +%T)"
python3 $B/scripts/rft_build.py assemble
echo "RFT_DATA_DONE $(date +%T)"
