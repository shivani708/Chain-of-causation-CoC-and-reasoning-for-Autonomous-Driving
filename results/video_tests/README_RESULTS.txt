SPEED-BUMP FIX MODEL (merged_sbfix_ck1500) - RESULTS  (10 Nov 2026)
===================================================================
Model (server): /data/common_datasets/ai24mtech02003/24_dataset_coc/models/merged_sbfix_ck1500
  = 72.2 model (merged_rft_ck500) + LoRA lora_sbfix_v1/checkpoint-1500, merged
  LoRA r=64 alpha=128 lr 2e-5, 1 epoch (1,512 steps), 16,000 cleaned CoC + 8,190 RFT LingoQA

WHAT WAS FIXED IN THE DATA
  Qwen3.5-27B checked all 7,554 CoC training answers that mention a speed bump:
    bump visible 79 (1.0%) | no bump 7,403 (98.0%) | unclear 72 (1.0%)   (control: 0/300 "maintain" flagged)
  7,475 bad samples dropped -> speed-bump share 12.1% -> 0.14%

LINGOQA (same harness, greedy, Lingo-Judge)
  ckpt 150 71.2 | 300 68.6 | 450 70.0 | 600 72.0 | 750 70.2 | 900 71.0 | 1050 72.2 | 1200 72.8 | 1350 73.0
  | 1500 73.2 (best, also best val loss) | final 72.4          old model: 72.2

VIDEOS - "speed bump" answers (share of moments)
                               72.2 model        new model
Hyderabad   training setup     50/115 (43%)  ->   1/115 (1%)
Hyderabad   vehicle setup      38/115 (33%)  ->   0/115 (0%)
Test track  training setup    108/207 (52%)  ->  21/207 (10%)
Test track  vehicle setup     117/207 (57%)  ->  35/207 (17%)

Decision mix, vehicle setup:
  Hyderabad   stop/slow 74% keep 26%           -> stop/slow 52% keep 48%
  Test track  stop/slow 96% keep 4%            -> stop/slow 83% keep 13% speed up 4%
Top new answers: "Slow down due to the lead vehicle ahead", "Keep distance to the lead vehicle ...",
  "Maintain speed due to the lead vehicle ahead", "Stop due to red traffic light", "Maintain speed due to clear lane ahead"

STILL OPEN
  Test track: 17% "speed bump" answers remain - checked frames (t=32, 188, 326 s) show yellow-black kerbs,
  no bump. The empty campus test track is far from the training cities. Next step if needed: add labelled
  test-track clips (maintain speed / accelerate) to training.

FILES
  SUMMARY.txt, *_answers.csv, *_overlay.mp4 (both setups), *_single_coc.mp4 (vehicle setup only, first sentence)
