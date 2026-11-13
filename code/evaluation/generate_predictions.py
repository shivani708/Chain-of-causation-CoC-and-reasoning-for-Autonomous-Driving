"""
LingoQA plain-format eval — Qwen3.5 thinking-mode aware.

Key fixes vs prior version:
  1. enable_thinking is NOT disabled — Qwen3.5 thinks by default and that's
     how the 82.0 LingoQA score was measured.
  2. MAX_NEW_TOKENS bumped to 4096 so the thinking trace + final answer fit.
  3. <think>...</think> blocks are stripped before writing, so Lingo-Judge
     scores the final answer only.

Usage:
  python generate_predictions.py \
      --parquet  /home/ai24mtech02003/vqa/LingoQA/dataset/val.parquet \
      --eval_dir /home/ai24mtech02003/vqa/LingoQA/dataset/val_images \
      --student  /data/common_datasets/ai25mtech12003/titan_v2/checkpoints/text2b_phase1_traj_withoutres/final/ \
      --out_csv  predictions/text2b_phase1_traj_withoutres_final.csv \
"""
# import os, csv, argparse, re
# import torch
# import pandas as pd
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForImageTextToText

# N_FRAMES_EVAL  = 5
# MAX_NEW_TOKENS = 4096      # thinking trace + answer; truncating kills the score
# THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


# def load_images(image_paths, eval_dir):
#     out = []
#     for p in image_paths:
#         if os.path.isabs(p):
#             full = p
#         else:
#             candidates = [os.path.join(eval_dir, p)]
#             stripped = p
#             for prefix in ("images/val/", "images/"):
#                 if stripped.startswith(prefix):
#                     stripped = stripped[len(prefix):]
#                     break
#             candidates.append(os.path.join(eval_dir, stripped))
#             candidates.append(os.path.join(eval_dir, os.path.basename(p)))
#             full = next((c for c in candidates if os.path.exists(c)), None)
#             if full is None:
#                 raise FileNotFoundError(
#                     f"could not find {p} under {eval_dir}; tried: {candidates}")
#         out.append(Image.open(full).convert("RGB"))
#     return out


# def student_answer_plain(model, processor, frames, question, gen_kwargs):
#     """Bare-question prompt with thinking mode left ON (Qwen3.5 default)."""
#     content = [{"type": "image", "image": img} for img in frames]
#     content.append({"type": "text", "text": question})
#     messages = [{"role": "user", "content": content}]

#     # NOTE: no enable_thinking=False — let Qwen3.5 think by default.
#     text_input = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True,
#     )
#     inputs = processor(
#         text=[text_input], images=frames, return_tensors="pt", padding=True,
#     ).to(model.device)

#     with torch.no_grad():
#         out = model.generate(**inputs, **gen_kwargs)

#     trimmed = out[:, inputs["input_ids"].shape[1]:]
#     raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

#     # Strip <think>...</think> so Lingo-Judge only sees the final answer.
#     # Also handle the case where </think> closes but no opening tag rendered
#     # (some templates omit the opening <think> after add_generation_prompt).
#     final = THINK_RE.sub("", raw)
#     if "</think>" in final:
#         final = final.split("</think>", 1)[-1]
#     final = re.sub(r"\s+", " ", final).strip()
#     return final, raw


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--parquet",  required=True)
#     ap.add_argument("--eval_dir", required=True)
#     ap.add_argument("--student",  required=True)
#     ap.add_argument("--out_csv",  default="predictions_plain.csv")
#     ap.add_argument("--limit",    type=int, default=None,
#                     help="Optional cap on rows for quick sanity runs")
#     ap.add_argument("--sample",   action="store_true",
#                     help="Use Qwen-recommended sampling instead of greedy")
#     ap.add_argument("--save_raw", action="store_true",
#                     help="Also save the unstripped output (with <think>) to a sidecar file")
#     args = ap.parse_args()

#     if args.sample:
#         gen_kwargs = dict(
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
#         )
#     else:
#         gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

#     print("=" * 60)
#     print("LingoQA PLAIN-FORMAT eval (Qwen3.5 thinking-mode aware)")
#     print(f"  parquet   : {args.parquet}")
#     print(f"  eval_dir  : {args.eval_dir}")
#     print(f"  student   : {args.student}")
#     print(f"  n_frames  : {N_FRAMES_EVAL}")
#     print(f"  out_csv   : {args.out_csv}")
#     print(f"  max_tokens: {MAX_NEW_TOKENS}")
#     print(f"  decoding  : {'sampled' if args.sample else 'greedy'}")
#     print("=" * 60)

#     df = pd.read_parquet(args.parquet)
#     print(f"loaded {len(df)} rows, {df['segment_id'].nunique()} unique segments")
#     if args.limit:
#         df = df.head(args.limit)
#         print(f"limited to first {len(df)} rows")

#     bad = df['images'].apply(len) != N_FRAMES_EVAL
#     if bad.any():
#         print(f"WARNING: {bad.sum()} rows do not have {N_FRAMES_EVAL} frames; skipped")

#     print("\nLoading student ...")
#     processor = AutoProcessor.from_pretrained(args.student)
#     if processor.tokenizer.pad_token_id is None:
#         processor.tokenizer.pad_token = processor.tokenizer.eos_token
#     model = AutoModelForImageTextToText.from_pretrained(
#         args.student, dtype=torch.bfloat16, device_map="cuda:0",
#     )
#     model.eval()

#     raw_path = args.out_csv.replace(".csv", "_raw.csv") if args.save_raw else None

#     frame_cache = {}
#     seen = set()
#     rows_out = []
#     rows_raw = []
#     n_skip = 0

#     for i, row in enumerate(df.itertuples(index=False), 1):
#         qid, sid = row.question_id, row.segment_id
#         images_rel = list(row.images)
#         question = row.question

#         if (qid, sid) in seen:
#             continue
#         if len(images_rel) != N_FRAMES_EVAL:
#             n_skip += 1
#             continue

#         if sid not in frame_cache:
#             try:
#                 frame_cache[sid] = load_images(images_rel, args.eval_dir)
#             except Exception as e:
#                 print(f"[{i}/{len(df)}] segment {sid}: load failed ({e})")
#                 frame_cache[sid] = None

#         frames = frame_cache[sid]
#         if frames is None:
#             n_skip += 1
#             continue

#         try:
#             ans, raw = student_answer_plain(
#                 model, processor, frames, question, gen_kwargs)
#         except Exception as e:
#             print(f"[{i}/{len(df)}] qid {qid}: failed ({e})")
#             ans, raw = "", ""

#         rows_out.append({"question_id": qid, "segment_id": sid, "answer": ans})
#         if raw_path is not None:
#             rows_raw.append({"question_id": qid, "segment_id": sid,
#                              "answer": ans, "raw": raw})
#         seen.add((qid, sid))

#         if i % 25 == 0 or i == len(df):
#             preview = ans[:80].replace("\n", " ")
#             print(f"  [{i}/{len(df)}] segment={sid[:8]} ans={preview}")

#         # checkpoint every iteration so a crash mid-run still leaves CSV
#         with open(args.out_csv, "w", newline="") as f:
#             w = csv.DictWriter(f, fieldnames=["question_id", "segment_id", "answer"])
#             w.writeheader()
#             w.writerows(rows_out)
#         if raw_path is not None:
#             with open(raw_path, "w", newline="") as f:
#                 w = csv.DictWriter(
#                     f, fieldnames=["question_id", "segment_id", "answer", "raw"])
#                 w.writeheader()
#                 w.writerows(rows_raw)

#     print(f"\nWrote {len(rows_out)} predictions to {args.out_csv}  (skipped: {n_skip})")
#     if raw_path is not None:
#         print(f"Raw (with <think>) saved to {raw_path}")
#     print("Next: python LingoQA/benchmark/evaluate.py "
#           f"--predictions_path {args.out_csv}")


# if __name__ == "__main__":
#     main()


# non thinking mode
"""
LingoQA plain-format eval — normal (non-thinking) mode.

Usage:
  python generate_predictions.py \
      --parquet  /home/ai24mtech02003/vqa/LingoQA/dataset/val.parquet \
      --eval_dir /home/ai24mtech02003/vqa/LingoQA/dataset/val_images \
      --student  /home/ai24mtech02003/vqa/final/models/joint_method3/checkpoint-20000\
      --out_csv  predictions_joint_method3_40k \
"""

import os, csv, argparse, re
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

N_FRAMES_EVAL  = 5
MAX_NEW_TOKENS = 256


def load_images(image_paths, eval_dir):
    """
    Parquet stores paths like "images/val/<segment>/<frame>.jpg", but on
    disk the images live under "<eval_dir>/val/<segment>/<frame>.jpg"
    (i.e. eval_dir should point at the val_images/ folder, and we just
    drop the leading "images/" prefix from the stored path).
    """
    out = []
    for p in image_paths:
        if os.path.isabs(p):
            full = p
        else:
            rel = p[len("images/"):] if p.startswith("images/") else p
            full = os.path.join(eval_dir, rel)
            if not os.path.exists(full):
                raise FileNotFoundError(f"could not find {p} -> {full}")
        out.append(Image.open(full).convert("RGB"))
    return out


def student_answer_plain(model, processor, frames, question, gen_kwargs,
                         system_prompt=None):
    """Bare-question prompt, thinking mode explicitly disabled.

    system_prompt is OPTIONAL and defaults to None, which reproduces the exact
    prompt behind the 65.6 / 68.6 numbers. When set, it is added as a system
    turn -- this changes only what we ask OUR model, never how Lingo-Judge
    scores it, so the benchmark protocol is untouched. Any table using it must
    apply the same prompt to every model compared, and disclose it.
    """
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": question})
    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": content})

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[text_input], images=frames, return_tensors="pt", padding=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    trimmed = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    ans = re.sub(r"\s+", " ", raw).strip()
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",  required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--student",  required=True)
    ap.add_argument("--out_csv",  default="predictions_plain.csv")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Optional cap on rows for quick sanity runs")
    ap.add_argument("--sample",   action="store_true",
                    help="Use Qwen-recommended sampling instead of greedy")
    ap.add_argument("--system_prompt", default=None,
                    help="Optional system turn (e.g. a brevity instruction). "
                         "Omit to reproduce the original bare-question prompt.")
    args = ap.parse_args()

    if args.sample:
        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        )
    else:
        gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    print("=" * 60)
    print("LingoQA PLAIN-FORMAT eval (thinking mode disabled)")
    print(f"  parquet   : {args.parquet}")
    print(f"  eval_dir  : {args.eval_dir}")
    print(f"  student   : {args.student}")
    print(f"  n_frames  : {N_FRAMES_EVAL}")
    print(f"  out_csv   : {args.out_csv}")
    print(f"  max_tokens: {MAX_NEW_TOKENS}")
    print(f"  decoding  : {'sampled' if args.sample else 'greedy'}")
    print(f"  system    : {args.system_prompt!r}")
    print("=" * 60)

    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df)} rows, {df['segment_id'].nunique()} unique segments")
    if args.limit:
        df = df.head(args.limit)
        print(f"limited to first {len(df)} rows")

    bad = df['images'].apply(len) != N_FRAMES_EVAL
    if bad.any():
        print(f"WARNING: {bad.sum()} rows do not have {N_FRAMES_EVAL} frames; skipped")

    print("\nLoading student ...")
    processor = AutoProcessor.from_pretrained(args.student)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    frame_cache = {}
    seen = set()
    rows_out = []
    n_skip = 0

    for i, row in enumerate(df.itertuples(index=False), 1):
        qid, sid = row.question_id, row.segment_id
        images_rel = list(row.images)
        question = row.question

        if (qid, sid) in seen:
            continue
        if len(images_rel) != N_FRAMES_EVAL:
            n_skip += 1
            continue

        if sid not in frame_cache:
            try:
                frame_cache[sid] = load_images(images_rel, args.eval_dir)
            except Exception as e:
                print(f"[{i}/{len(df)}] segment {sid}: load failed ({e})")
                frame_cache[sid] = None

        frames = frame_cache[sid]
        if frames is None:
            n_skip += 1
            continue

        try:
            ans = student_answer_plain(
                model, processor, frames, question, gen_kwargs,
                system_prompt=args.system_prompt)
        except Exception as e:
            print(f"[{i}/{len(df)}] qid {qid}: failed ({e})")
            ans = ""

        rows_out.append({"question_id": qid, "segment_id": sid, "answer": ans})
        seen.add((qid, sid))

        if i % 25 == 0 or i == len(df):
            preview = ans[:80].replace("\n", " ")
            print(f"  [{i}/{len(df)}] segment={sid[:8]} ans={preview}")

        # checkpoint every iteration so a crash mid-run still leaves CSV
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["question_id", "segment_id", "answer"])
            w.writeheader()
            w.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} predictions to {args.out_csv}  (skipped: {n_skip})")
    print("Next: python LingoQA/benchmark/evaluate.py "
          f"--predictions_path {args.out_csv}")


if __name__ == "__main__":
    main()