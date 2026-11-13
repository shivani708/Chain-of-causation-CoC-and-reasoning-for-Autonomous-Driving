#!/bin/bash
# LingoQA only (same generate_predictions.py + Lingo-Judge as every other row), models run one after another.
#   sbatch run_lingoqa_only.sbatch TAG1:MODEL1 TAG2:MODEL2 ...
#SBATCH -p h200
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 04:00:00
#SBATCH -o /data/common_datasets/ai24mtech02003/24_dataset_coc/eval/logs/lingo_%x_%j.log
set -uo pipefail
E=/data/common_datasets/ai24mtech02003/24_dataset_coc/eval
for spec in "$@"; do
  t=${spec%%:*}; m=${spec#*:}
  echo "=== $t  $m  $(date +%T)"
  cd /home/ai24mtech02003/vqa/final
  /home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python generate_predictions.py \
     --parquet /home/ai24mtech02003/vqa/LingoQA/dataset/val.parquet --eval_dir /home/ai24mtech02003/vqa/LingoQA/dataset/val_images \
     --student "$m" --out_csv "$E/lingoqa/$t.csv" 2>&1 | grep -E "^Wrote" | tail -1
  cd /home/ai24mtech02003/vqa/LingoQA/benchmark
  echo "LINGOQA_SCORE $t: $(/home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python evaluate.py --predictions_path $E/lingoqa/$t.csv 2>&1 | grep -vE 'Map|it/s|Filter|Loading' | tail -1)"
done
echo LINGO_ONLY_DONE
