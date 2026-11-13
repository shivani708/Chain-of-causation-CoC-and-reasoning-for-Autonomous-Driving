#!/bin/bash
# Gentle LoRA on the frozen 68.6 model, trained only on its own best-judged answers (RFT) + 3k CoC to keep CoC.
#   r=64, alpha=128, lr 2e-5, warmup 30, global batch 16, 1 epoch; adapter saved every 100 steps for LingoQA selection.
#SBATCH -p h200
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/common_datasets/ai24mtech02003/24_dataset_coc/logs/lora_rft-%j.out
set -uo pipefail
B=/data/common_datasets/ai24mtech02003/24_dataset_coc
PY=/home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python
BASE=/data/common_datasets/ai24mtech02003/coc_student/models/coc_student_alpamayo/best
RUN="${1:-lora_rft_v1}"
cat $B/data/rft/rft_train.jsonl <(head -3000 $B/data/lora_l/part_coc_plain.jsonl) > $B/data/rft/rft_train_plus_coc.jsonl
cd $B/scripts
$PY train_24coc.py --run_name "$RUN" --model "$BASE" --models_root "$B/models" \
  --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 --lora_targets llm \
  --mix_train_jsonl $B/data/rft/rft_train_plus_coc.jsonl --mix_val_jsonl $B/data/rft/rft_val.jsonl \
  --coc_val_jsonl $B/data/coc_val.jsonl --coc_heldout_cities delhi kota_out \
  --epochs 1 --bs 2 --grad_accum 8 --lr 2e-5 --warmup_steps 30 \
  --save_steps 100 --save_total_limit 100 --eval_steps 100 --max_val_samples 300 --dataloader_num_workers 4
echo "LORA_RFT_${RUN}_COMPLETE (exit $?)"
