# Chain-of-Causation (CoC) Reasoning for Autonomous Driving

A **Qwen3-VL-2B** vision-language model fine-tuned to give a short **Chain-of-Causation** driving decision and its reason from four front-camera frames. Example: *"Slow down due to the lead vehicle ahead."* It can also answer free-form driving questions in the LingoQA style.

**Final model: `merged_sbfix_ck1500`, LingoQA 73.2** (base Qwen3-VL-2B: 46.2)

> ⚠️ **Advisory only.** Never use the model output to steer, brake or accelerate a vehicle.

Full documentation (method, architecture, deployment, results, IEEE references): open [`CoC_73p2_documentation.html`](CoC_73p2_documentation.html) in a browser.

---

## Model lineage

| Stage | What was done | LingoQA |
|---|---|---|
| Qwen3-VL-2B-Instruct | base model | 46.2 |
| `coc_student_alpamayo` | full fine-tune on Indian CoC data | 68.6 |
| `merged_rft_ck500` | + LoRA (RFT: 8,190 of the model's own Lingo-Judge-approved answers + 3,000 CoC samples) | 72.2 |
| **`merged_sbfix_ck1500`** | + LoRA on the speed-bump-cleaned data (16,000 CoC + 8,190 RFT samples) | **73.2** |

**Speed-bump fix:** Qwen3.5-27B checked all 7,554 CoC training answers that mention a speed bump. It saw a bump in only 79 of them (1%), so 7,475 wrong labels were removed.

LoRA settings: r=64, alpha=128, dropout 0.05, LLM layers only, lr 2e-5, global batch 16, 1 epoch.

## Results

**LingoQA** (500 val questions, Lingo-Judge, greedy decoding). The 72.2 model was run 100 times and gave identical answers every time.

**"Speed bump" share of answers on test videos** (72.2 model → final model):

| Video | Training setup | Vehicle setup |
|---|---|---|
| Hyderabad city | 43% → 1% | 33% → 0% |
| Test track | 52% → 10% | 57% → 17% |

**Real-time speed:** 0.4–0.6 s per answer on an A40 GPU (vehicle setup), about 2 answers per second.

Details: [`results/`](results/) — `BENCHMARK_summary.txt`, `lingoqa/`, `speed_bump_cleanup/`, `video_tests/` (per-window answer CSVs).

## Repository layout

```
code/training/          train_24coc.py, merge_lora.py, LoRA + RFT Slurm jobs
code/data_cleaning/     Qwen3.5-27B speed-bump label check + cleaned data builder
code/evaluation/        LingoQA, WaymoQA, spatial benchmarks, repeat runs, video tests
deployment/             real-time scripts: live_coc_single.py (CoC overlay), live_camera_vqa.py (Q&A)
results/                LingoQA scores, cleanup summaries, video test results
checkpoint/             LoRA adapter config, trainer state and training logs (no weights)
model/merged_sbfix_ck1500/   config, tokenizer and chat template (no weights)
COMMANDS.txt            every command to install, run, evaluate and retrain
OVERVIEW_SUMMARY.txt    short project summary
```

> **Weights are not in this repository.** `model.safetensors` (4.0 GB), the LoRA `adapter_model.safetensors` and the optimizer state are distributed separately. To use the model, put `model.safetensors` into `model/merged_sbfix_ck1500/`.

## Quick start (real-time testing)

```bash
python3 -m venv ~/coc_live && source ~/coc_live/bin/activate
pip install torch torchvision                 # CUDA build matching your driver
pip install -r deployment/requirements_live.txt   # transformers >= 5.0

python3 deployment/live_coc_single.py                       # webcam 0
python3 deployment/live_coc_single.py --source drive.mp4    # recorded video
python3 deployment/live_coc_single.py --source rtsp://<camera-ip>/stream
python3 deployment/live_camera_vqa.py --source 0            # ask driving questions live
```

Minimal Python usage:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
m = AutoModelForImageTextToText.from_pretrained("model/merged_sbfix_ck1500", dtype="bfloat16").to("cuda").eval()
p = AutoProcessor.from_pretrained("model/merged_sbfix_ck1500")
content = [{"type": "image", "image": img} for img in four_pil_frames] + [{"type": "text", "text": COC_PROMPT}]
x = p.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                          return_dict=True, return_tensors="pt").to("cuda")
y = m.generate(**x, max_new_tokens=96, do_sample=False)   # greedy decoding is required
print(p.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

`COC_PROMPT` is the exact `PROMPT` string in `deployment/live_coc_single.py`. Do not change it.

## Known limitations

- On the test track and campus roads, the model sometimes calls yellow-black kerbs a "speed bump" (10–17% of answers). It sometimes also says "sharp curve" on a straight road. These roads are not in the training data.
- A test-track LoRA (`lora_ttfix_v1`) brought these errors to 0%. However, it lowered LingoQA to 71.4 and made the model less cautious in the city, so it was not chosen.
