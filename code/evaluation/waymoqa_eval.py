"""WaymoQA image-split MCQ score for one model/checkpoint.

Protocol = official WaymoQA scripts/eval_waymoqa_vllm.py (image QA): 8 same-timestamp views in the order
FRONT_LEFT, FRONT, FRONT_RIGHT, SIDE_RIGHT, SIDE_LEFT, REAR_LEFT, REAR, REAR_RIGHT, each resized to longest side
1024 (JPEG q85); same system prompt / user text; greedy, max 5 new tokens; official parse_choice().
Category groups follow the WaymoQA paper (Appendix B, Table 13): Normal / Others / Safety-critical.

Scores (written to <out_dir>/scores.json, predictions to <out_dir>/predictions.csv):
  acc_*        unparsable answer counted wrong (Qwen-Drive-1.0 protocol)  <- main numbers
  acc_*_parsed unparsable answers excluded (official WaymoQA script)
"""
import argparse
import base64
import csv
import json
import os
import re
import time
from collections import defaultdict

import cv2

W = "/data/common_datasets/ai24mtech02003/object_detection/waymoqa"
TEST = f"{W}/annotations/test.jsonl"
IMG_DIR = f"{W}/test_imgs"
BASE = ("/home/ai24mtech02003/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
        "89644892e4d85e24eaac8bacfd4f463576704203")
PROC_FILES = ["chat_template.json", "merges.txt", "preprocessor_config.json", "tokenizer_config.json",
              "tokenizer.json", "vocab.json", "video_preprocessor_config.json"]
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors"]

CHOICE_LETTERS = "ABCD"
MAX_SIDE, JPEG_Q, MAX_NEW_TOKENS = 1024, 85, 5
CAM_ORDER_8 = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "SIDE_RIGHT", "SIDE_LEFT", "REAR_LEFT", "REAR", "REAR_RIGHT"]
STRICT_SUFFIX = "Respond with EXACTLY ONE LETTER from {A,B,C,D}. No words, no punctuation, no explanation."

GROUPS = {
    "normal": ["Traffic Signal", "Normal scene caption", "Normal scene Understanding", "Normal Planning",
               "Normal Perception & Prediction"],
    "others": ["Counterfactual", "Special Event", "Spatial Relationship"],
    "safety": ["Safety-critical scene Understanding", "Safety-critical Perception", "Safety-critical Prediction",
               "Safety-critical Planning", "Safety-critical scene caption", "Uncertainty"],
}


def normalize_options(options):
    return [re.sub(r"^[A-Da-d]\.\s*", "", str(s)).strip() for s in options]


def parse_choice(text, k):  # verbatim logic of the official script
    t = re.sub(r"[^A-Z0-9]", "", text.strip().upper())
    if t:
        ch = t[0]
        if ch in CHOICE_LETTERS:
            idx = CHOICE_LETTERS.index(ch)
            return idx if idx < k else None
        if ch in "1234":
            idx = int(ch) - 1
            return idx if idx < k else None
    m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    if m:
        return CHOICE_LETTERS.index(m.group(1).upper())
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return int(m.group(1)) - 1
    return None


def data_url(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = max(h, w) / float(MAX_SIDE)
    if s > 1:
        img = cv2.resize(img, (int(w / s), int(h / s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def build_messages(question, options, urls, prompt):
    options = normalize_options(options)
    opt_str = "\n".join(f"{CHOICE_LETTERS[i]}. {o}" for i, o in enumerate(options))
    order_str = ", ".join(c.replace("_", " ") for c in CAM_ORDER_8)
    system = ("You are a driving VQA assistant.\nYou will be given 8 camera views from the same timestamp.\n"
              f"The images are provided in this order: {order_str}.\nUse ALL views to answer.\n" + STRICT_SUFFIX)
    content = []
    for cam, u in zip(CAM_ORDER_8, urls):
        if prompt == "viewtag":  # Qwen-Drive style: name each view next to its image
            content.append({"type": "text", "text": f"{cam.replace('_', ' ').title()} view:"})
        content.append({"type": "image_url", "image_url": {"url": u}})
    content.append({"type": "text",
                    "text": f"Question:\n{question}\n\nOptions:\n{opt_str}\n\nAnswer with A/B/C/D only."})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def model_dir_for(model, tag):
    """Checkpoints store no tokenizer/processor: link weights + base processor files into one dir."""
    if os.path.exists(os.path.join(model, "preprocessor_config.json")):
        return model
    d = f"/data/common_datasets/ai24mtech02003/24_dataset_coc/eval/models/{tag}"  # 24_dataset_coc copy: links live here
    os.makedirs(d, exist_ok=True)
    for f in MODEL_FILES:
        if os.path.exists(os.path.join(model, f)) and not os.path.lexists(os.path.join(d, f)):
            os.symlink(os.path.join(model, f), os.path.join(d, f))
    for f in PROC_FILES:
        if os.path.exists(os.path.join(BASE, f)) and not os.path.lexists(os.path.join(d, f)):
            os.symlink(os.path.join(BASE, f), os.path.join(d, f))
    return d


def acc(rows, strict):
    rs = rows if strict else [r for r in rows if r["pred_index"] != ""]
    return round(100.0 * sum(r["correct"] for r in rs) / len(rs), 2) if rs else None


def score(rows):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    res = {"n": len(rows), "parse_fail_pct": round(100.0 * sum(r["pred_index"] == "" for r in rows) / len(rows), 2),
           "acc_all": acc(rows, True), "acc_all_parsed": acc(rows, False)}
    for g, cats in GROUPS.items():
        rs = [r for c in cats for r in by_cat.get(c, [])]
        res[f"acc_{g}"], res[f"acc_{g}_parsed"], res[f"n_{g}"] = acc(rs, True), acc(rs, False), len(rs)
    res["per_category"] = {c: {"n": len(rs), "acc": acc(rs, True)} for c, rs in sorted(by_cat.items())}
    res["pred_letter_dist"] = {L: sum(r["pred_letter"] == L for r in rows) for L in CHOICE_LETTERS + "-"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir or base model dir")
    ap.add_argument("--tag", required=True, help="short name, used for output + symlink dir")
    ap.add_argument("--out_root", default="/data/common_datasets/ai24mtech02003/24_dataset_coc/eval/waymoqa")
    ap.add_argument("--prompt", default="official", choices=["official", "viewtag"])
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--gpu_util", type=float, default=0.8)
    args = ap.parse_args()
    out_dir = os.path.join(args.out_root, args.tag if args.prompt == "official" else f"{args.tag}_{args.prompt}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    items = [r for r in map(json.loads, open(TEST)) if r["type"] == "image"]
    if args.limit:
        items = items[::max(1, len(items) // args.limit)][:args.limit]
    cache, missing = {}, 0
    for it in items:
        key = (it["token"], int(it["frame_index"]))
        if key not in cache:
            paths = [f"{IMG_DIR}/{key[0]}_{key[1]:03d}_{c}.jpg" for c in CAM_ORDER_8]
            cache[key] = [data_url(p) for p in paths] if all(os.path.exists(p) for p in paths) else None
            missing += cache[key] is None
    if missing:
        raise SystemExit(f"{missing} frames have missing camera images in {IMG_DIR}")

    from vllm import LLM, SamplingParams
    mdir = model_dir_for(args.model, args.tag)
    llm = LLM(model=mdir, max_model_len=12288, gpu_memory_utilization=args.gpu_util,
              limit_mm_per_prompt={"image": 8}, seed=0)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_NEW_TOKENS)
    rows, B = [], 256
    for s in range(0, len(items), B):
        chunk = items[s:s + B]
        convs = [build_messages(it["question"], it["options"], cache[(it["token"], int(it["frame_index"]))],
                                args.prompt) for it in chunk]
        outs = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        for it, o in zip(chunk, outs):
            text = o.outputs[0].text
            k = len(it["options"])
            p = parse_choice(text, k)
            gt = int(it["answer_index"])
            rows.append({"token": it["token"], "frame_index": it["frame_index"], "category": it["category"],
                         "question": it["question"], "gt_index": gt, "pred_index": "" if p is None else p,
                         "pred_letter": "-" if p is None else CHOICE_LETTERS[p], "correct": int(p == gt),
                         "raw": re.sub(r"\s+", " ", text).strip()})
        print(f"  {len(rows)}/{len(items)}", flush=True)

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"model": args.model, "tag": args.tag, "prompt": args.prompt, "full_benchmark": len(rows) == 5519,
           **score(rows), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(os.path.join(out_dir, "scores.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_category"}))


if __name__ == "__main__":
    main()
