#!/bin/bash
# Repeat the exact LingoQA eval (same generate_predictions.py greedy decoding + Lingo-Judge) N times on one model,
# to check run-to-run stability.   sbatch repeat_lingoqa.sbatch <MODEL> <FIRST_RUN> <LAST_RUN>
#SBATCH -p h200
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 08:00:00
#SBATCH -o /data/common_datasets/ai24mtech02003/24_dataset_coc/eval/logs/repeat_%x_%j.log
set -uo pipefail
E=/data/common_datasets/ai24mtech02003/24_dataset_coc/eval
PY=/home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python
MODEL=$1; FIRST=$2; LAST=$3
OUT=$E/repeat_rft_ck500
mkdir -p $OUT
for i in $(seq $FIRST $LAST); do
  tag=run_$(printf %03d $i)
  [ -s $OUT/$tag.csv ] && grep -q "^$tag " $OUT/scores.txt 2>/dev/null && { echo "skip $tag"; continue; }
  cd /home/ai24mtech02003/vqa/final
  $PY generate_predictions.py --parquet /home/ai24mtech02003/vqa/LingoQA/dataset/val.parquet \
     --eval_dir /home/ai24mtech02003/vqa/LingoQA/dataset/val_images --student $MODEL --out_csv $OUT/$tag.csv 2>&1 | grep -E "^Wrote" | tail -1
  cd /home/ai24mtech02003/vqa/LingoQA/benchmark
  S=$($PY evaluate.py --predictions_path $OUT/$tag.csv 2>&1 | grep -vE 'Map|it/s|Filter|Loading' | tail -1)
  echo "$tag $(hostname) $(date +%T) | $S" | tee -a $OUT/scores.txt
done
echo REPEAT_DONE $FIRST $LAST
