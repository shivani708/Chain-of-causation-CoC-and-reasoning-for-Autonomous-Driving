#!/bin/bash
# Merge sbfix checkpoint-1500 into the 72.2 model, then run the video test on Hyderabad + clean testbed video.
#SBATCH -p a40
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 03:00:00
#SBATCH -J coc_vsbfix
#SBATCH -o /data/common_datasets/ai24mtech02003/24_dataset_coc/eval/logs/coc_vsbfix_%j.log
B=/data/common_datasets/ai24mtech02003/24_dataset_coc
PY=/home/ai24mtech02003/miniconda3/envs/vqa_db/bin/python
M=$B/models/merged_sbfix_ck1500
[ -s $M/model.safetensors ] || $PY $B/scripts/merge_lora.py $B/models/merged_rft_ck500 $B/models/lora_sbfix_v1/checkpoint-1500 $M 2>&1 | tail -2
ls $M
$PY -u $B/scripts/test_coc_video.py --model $M \
  --videos $B/eval/video_input/Hyderabad1.mp4 $B/eval/video_input/cam_20261106_100001.mp4 \
  --out $B/eval/video_test_sbfix
echo VIDEO_DONE
