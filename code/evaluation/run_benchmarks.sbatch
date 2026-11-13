#!/bin/bash
# Benchmarks for one checkpoint with the same protocols as coc_student_alpamayo (68.6):
#   WaymoQA image split (official prompt, vLLM), LingoQA (vqa/final/generate_predictions.py + Lingo-Judge),
#   spatial test (5,600 Q, held-out Delhi + kota_out).   Outputs only under 24_dataset_coc/eval.
#   sbatch run_benchmarks.sbatch <TAG> <MODEL_DIR> [lingo|nolingo] [spatial_extra_tag:model ...]
#SBATCH -p h200
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 08:00:00
#SBATCH -o /data/common_datasets/ai24mtech02003/24_dataset_coc/eval/logs/bench_%x_%j.log
set -uo pipefail
E=/data/common_datasets/ai24mtech02003/24_dataset_coc/eval
OD=/data/common_datasets/ai24mtech02003/object_detection
TAG=$1; M=$2; LINGO=${3:-lingo}; shift 3 || shift $#
echo "=== $TAG  $M  $(hostname) $(date '+%F %T')"
# ---- WaymoQA (also creates $E/models/$TAG = weights + base processor files)
( . $OD/pretrain_qwen3vl2b/env_train.sh; $OD/venv/bin/python -B $E/waymoqa_eval.py --model "$M" --tag "$TAG" 2>&1 | grep -E '^\{|Error|Traceback|error:' | tail -3 )
MD=$M; [ -d "$E/models/$TAG" ] && MD=$E/models/$TAG
echo "model dir for generation: $MD"
# ---- LingoQA
if [ "$LINGO" = lingo ]; then
  cd /home/ai24mtech02003/vqa/final
  /home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python generate_predictions.py \
     --parquet /home/ai24mtech02003/vqa/LingoQA/dataset/val.parquet --eval_dir /home/ai24mtech02003/vqa/LingoQA/dataset/val_images \
     --student "$MD" --out_csv "$E/lingoqa/$TAG.csv" 2>&1 | grep -vE "Loading weights|it/s\]" | tail -3
  cd /home/ai24mtech02003/vqa/LingoQA/benchmark
  export LD_LIBRARY_PATH=/home/ai24mtech02003/miniconda3/envs/vqa/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
  for csv in "$E/lingoqa/$TAG.csv" "$@"; do
    [[ "$csv" == *.csv ]] || continue
    echo "LINGOQA_SCORE $(basename $csv): $(/home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python evaluate.py --predictions_path $csv 2>&1 | grep -vE 'Map|it/s|Filter|Loading' | tail -1)"
  done
fi
# ---- spatial test
( . $OD/pretrain_qwen3vl2b/env_train.sh; cd $E; $OD/venv/bin/python -B spatial_eval.py run "$TAG" "$MD" 2>&1 | tail -2 )
for spec in "$@"; do
  [[ "$spec" == *:* ]] || continue
  t=${spec%%:*}; m=${spec#*:}
  ( . $OD/pretrain_qwen3vl2b/env_train.sh; $OD/venv/bin/python -B $E/waymoqa_eval.py --model "$m" --tag "$t" --limit 1 > /dev/null 2>&1;
    md=$m; [ -d "$E/models/$t" ] && md=$E/models/$t; cd $E; $OD/venv/bin/python -B spatial_eval.py run "$t" "$md" 2>&1 | tail -2 )
done
echo "BENCH_DONE_$TAG $(date '+%F %T')"
