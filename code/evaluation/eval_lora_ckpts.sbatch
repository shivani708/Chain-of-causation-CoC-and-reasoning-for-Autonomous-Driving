#!/bin/bash
# Score every saved LoRA checkpoint of a run on LingoQA, one after another (1 GPU):
#   merge adapter into the frozen base (temp dir) -> generate_predictions.py -> Lingo-Judge -> delete temp merge.
#   sbatch eval_lora_ckpts.sbatch <RUN> [BASE]
#SBATCH -p h200
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 12:00:00
#SBATCH -o /data/common_datasets/ai24mtech02003/24_dataset_coc/eval/logs/lora_ckpts_%x_%j.log
set -uo pipefail
B=/data/common_datasets/ai24mtech02003/24_dataset_coc; E=$B/eval
PY=/home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python
RUN=$1; BASE=${2:-/data/common_datasets/ai24mtech02003/coc_student/models/coc_student_alpamayo/best}
TMP=$B/models/_tmp_merge_$RUN
for ck in $(ls -d $B/models/$RUN/checkpoint-* | sort -t- -k2 -n) $B/models/$RUN; do
  tag=${RUN}_$(basename $ck); [ "$ck" = "$B/models/$RUN" ] && tag=${RUN}_final
  [ -s $E/lingoqa/$tag.csv ] && { echo "skip $tag"; continue; }
  $PY $B/scripts/merge_lora.py $BASE $ck $TMP > /dev/null 2>&1 || { echo "MERGE_FAIL $tag"; continue; }
  cd /home/ai24mtech02003/vqa/final
  $PY generate_predictions.py --parquet /home/ai24mtech02003/vqa/LingoQA/dataset/val.parquet \
     --eval_dir /home/ai24mtech02003/vqa/LingoQA/dataset/val_images --student $TMP --out_csv $E/lingoqa/$tag.csv 2>&1 | grep -E "^Wrote" | tail -1
  cd /home/ai24mtech02003/vqa/LingoQA/benchmark
  S=$($PY evaluate.py --predictions_path $E/lingoqa/$tag.csv 2>&1 | grep -vE 'Map|it/s|Filter|Loading' | tail -1)
  N=$(python3 -c "import csv;print(sum(r['answer'].strip().lower().startswith('action, justification') for r in csv.DictReader(open('$E/lingoqa/$tag.csv'))))")
  echo "LINGOQA $tag | $S | literal_action_justification=$N" | tee -a $E/lingoqa/${RUN}_scores.txt
  rm -rf "$B/models/_tmp_merge_$RUN"
done
echo LORA_CKPT_EVAL_DONE
