#!/bin/bash
# Speed-bump fix: LoRA on the 72.2 model (merged_rft_ck500), trained on cleaned CoC (unverified speed-bump
# answers removed by Qwen3.5-27B) + the RFT LingoQA answers. Same LoRA settings as the RFT run:
#   r=64, alpha=128, lr 2e-5, warmup 30, global batch 16, 1 epoch (~1,512 steps), adapter saved every 150 steps.
#SBATCH -p b6000
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/common_datasets/ai24mtech02003/24_dataset_coc/logs/lora_sbfix-%j.out
set -uo pipefail
B=/data/common_datasets/ai24mtech02003/24_dataset_coc
PY=/home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python
BASE=$B/models/merged_rft_ck500
RUN="${1:-lora_sbfix_v1}"
python3 $B/scripts/build_sbfix.py || exit 1
cd $B/scripts
$PY train_24coc.py --run_name "$RUN" --model "$BASE" --models_root "$B/models" \
  --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 --lora_targets llm \
  --mix_train_jsonl $B/data/sbfix/train.jsonl --mix_val_jsonl $B/data/rft/rft_val.jsonl \
  --coc_val_jsonl $B/data/coc_val.jsonl --coc_heldout_cities delhi kota_out \
  --epochs 1 --bs 2 --grad_accum 8 --lr 2e-5 --warmup_steps 30 \
  --save_steps 150 --save_total_limit 100 --eval_steps 150 --max_val_samples 300 --dataloader_num_workers 4
echo "LORA_SBFIX_${RUN}_COMPLETE (exit $?)"
