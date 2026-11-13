"""In-domain SPATIAL evaluation on the held-out cities (Delhi + kota_out), per skill.

  build : python spatial_eval.py build          -> spatial_eval/eval_set.jsonl (fixed, seed 13)
  run   : python spatial_eval.py run TAG MODEL_DIR   (GPU, vLLM) -> spatial_eval/results/TAG/{predictions.jsonl,scores.json}
  table : python spatial_eval.py table          -> spatial_eval/spatial_results.txt

Every question gets a short answer instruction and is scored on meaning (yes/no, number, direction, lane, band,
box IoU), not on wording, so a model is not penalised for its answer style.
"""
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

PT = "/data/common_datasets/ai24mtech02003/object_detection/pretrain_qwen3vl2b"
EVAL_SET = f"{PT}/spatial_eval/eval_set.jsonl"  # same 5,600-question set as all earlier rows (read only)
OUT = "/data/common_datasets/ai24mtech02003/24_dataset_coc/eval/spatial"  # 24_dataset_coc copy: results here
V2 = "/home/ai24mtech02003/vqa/final/vqa-nuscenes_co_d/new-indianscenarios-output-v2/stage2_qa"
COC = "/data/common_datasets/ai24mtech02003/object_detection/vqa_mixed_512x288"
HELD = {"v2": ["Delhi", "kota_out"], "coc": ["delhi", "kota_out"]}
PER_SKILL = 400
MAX_PIXELS = 384 * 28 * 28  # same as training
SEED = 13

YN = " Answer yes or no."
DIR3 = " Answer with one of: left, right, front."
REL4 = " Answer with one of: left, right, in front, behind."
NUM = " Answer with a number."
LANE6 = (" Answer with one of: ego lane, lane to the left, lane to the right, left side of the road, "
         "right side of the road, opposite side of the road.")
BAND = " Answer with one of: near (under 12 m), mid (12 to 30 m), far (over 30 m)."
BOX = (" Give its bounding box as x1,y1,x2,y2.")

V2_SKILLS = {  # v2 type -> (skill name, kind, instruction)
    "ego_relative_yesno": ("v2 ego-relative yes/no", "yesno", YN),
    "relative_position_yesno": ("v2 object-object yes/no", "yesno", YN),
    "ego_pov_object_relation": ("v2 ego-viewpoint relation yes/no", "yesno", YN),
    "ego_relative_open": ("v2 ego-relative direction", "dir3", DIR3),
    "relative_position_open": ("v2 object-object direction", "rel4", REL4),
    "object_detection_count": ("v2 counting", "num", NUM),
    "object_detection_existence": ("v2 existence yes/no", "yesno", YN),
}
TAG_RE = re.compile(r"<class_id>[^<]*</class_id><bb>([\d.]+),([\d.]+),([\d.]+),([\d.]+)</bb>")


def resolve(p):
    if os.path.exists(p):
        return p
    b = os.path.splitext(p)[0]
    for e in (".jpg", ".jpeg", ".png"):
        if os.path.exists(b + e):
            return b + e
    return None


# ------------------------------------------------------------------ gold labels
def dir3(text):
    t = text.lower()
    if "left" in t and "right" not in t:
        return "left"
    if "right" in t and "left" not in t:
        return "right"
    if re.search(r"\b(front|ahead|straight)\b", t):
        return "front"
    return None


def rel4(text):
    t = text.lower()
    if "behind" in t:
        return "behind"
    if "left" in t and "right" not in t:
        return "left"
    if "right" in t and "left" not in t:
        return "right"
    if re.search(r"\b(front|ahead)\b", t):
        return "in front"
    return None


def lane6(text):
    t = text.lower()
    for key, lab in (("opposite", "opposite"), ("ego lane", "ego"), ("same lane", "ego"), ("our lane", "ego"),
                     ("lane to the left", "left_lane"), ("left lane", "left_lane"), ("left adjacent", "left_lane"),
                     ("lane to the right", "right_lane"), ("right lane", "right_lane"),
                     ("right adjacent", "right_lane"), ("left side of the road", "left_side"),
                     ("left roadside", "left_side"), ("right side of the road", "right_side"),
                     ("right roadside", "right_side")):
        if key in t:
            return lab
    return None


def band(text):
    t = text.lower()
    m = re.search(r"\b(near|mid|far)\b", t)
    if m:
        return m.group(1)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m\b|meter|metre)", t)
    if m:
        d = float(m.group(1))
        return "near" if d < 12 else "mid" if d <= 30 else "far"
    if "within about 5 m" in t:
        return "near"
    return None


def yesno(text):
    m = re.search(r"\b(yes|no)\b", text.lower())
    return m.group(1) if m else None


WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def number(text):
    m = re.search(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", TAG_RE.sub("", text.lower()))
    if not m:
        return None
    return int(m.group(1)) if m.group(1).isdigit() else WORDNUM[m.group(1)]


def pred_box(text, W, H):
    """Accept our tags, Qwen's native 0-1000 boxes, 0-1 boxes or pixel boxes -> normalised 0-1."""
    m = TAG_RE.search(text)
    nums = [float(x) for x in m.groups()] if m else [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)][:4]
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]
    mx = max(nums[:4])
    if mx <= 1.0:
        s = (1, 1)
    elif mx <= 1000 and not m:
        s = (1000, 1000)          # Qwen native grounding scale
    else:
        s = (W, H)
    return [x1 / s[0], y1 / s[1], x2 / s[0], y2 / s[1]]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ------------------------------------------------------------------ build
def build():
    rng = random.Random(SEED)
    items = []
    # v2 held-out
    pools = defaultdict(list)
    for c in HELD["v2"]:
        d = json.load(open(f"{V2}/{c}_vqa.json"))
        for r in d["records"]:
            img = resolve(r["image_path"])
            if not img:
                continue
            for qa in r["qa_pairs"]:
                t = qa.get("type")
                if t in V2_SKILLS:
                    pools[t].append((img, qa["question"], str(qa["answer"])))
    for t, (skill, kind, instr) in V2_SKILLS.items():
        pool = pools[t]
        if kind == "yesno":  # balance yes/no exactly
            ys = [p for p in pool if p[2].lower() == "yes"]
            ns = [p for p in pool if p[2].lower() == "no"]
            pick = rng.sample(ys, min(PER_SKILL // 2, len(ys))) + rng.sample(ns, min(PER_SKILL // 2, len(ns)))
        else:
            pick = rng.sample(pool, min(PER_SKILL, len(pool)))
        for img, q, a in pick:
            gold = {"yesno": a.lower(), "dir3": dir3(a), "rel4": rel4(a), "num": number(a)}[kind]
            if gold is None:
                continue
            items.append({"skill": skill, "kind": kind, "source": "v2", "image": img,
                          "question": q.strip() + instr, "gold": gold})
    # CoC held-out (template QAs only: their answers are exact facts)
    coc = defaultdict(list)
    for c in HELD["coc"]:
        for split in ("train", "val"):
            for line in open(f"{COC}/{c}/{c}_{split}.jsonl"):
                r = json.loads(line)
                if r["source"] == "template":
                    coc[r["question"].split(" ")[0] + "|" + r["type"]].append(r)
    def add(skill, kind, r, q, gold, extra=None):
        it = {"skill": skill, "kind": kind, "source": "coc", "image": r["image_1280x720"], "question": q,
              "gold": gold}
        if extra:
            it.update(extra)
        items.append(it)
    allc = [r for v in coc.values() for r in v]
    rng.shuffle(allc)
    cnt = Counter()
    for r in allc:
        q, a = r["question"], r["answer"]
        if q.startswith("Where is ") and q.endswith("relative to the ego vehicle?"):
            ref = q[len("Where is "):-len(" relative to the ego vehicle?")]
            for skill, kind, qq, gold in (
                    ("CoC direction (left/right/front)", "dir3", f"Is {ref} to the left, to the right or in front of the ego vehicle?" + DIR3, dir3(a.split(" the ego vehicle")[0])),
                    ("CoC lane / road area", "lane6", f"Where on the road is {ref}?" + LANE6, lane6(a)),
                    ("CoC distance band", "band", f"How far is {ref} from the ego vehicle?" + BAND, band(a))):
                if gold and cnt[skill] < PER_SKILL:
                    add(skill, kind, r, qq, gold); cnt[skill] += 1
            boxes = list(r.get("evidence_boxes", {}).values())
            if boxes and cnt["CoC grounding (IoU>=0.5)"] < PER_SKILL:
                W, H = 1280, 720
                b = boxes[0]
                add("CoC grounding (IoU>=0.5)", "box", r, f"Locate {ref} in the image." + BOX,
                    [b[0] / W, b[1] / H, b[2] / W, b[3] / H], {"W": W, "H": H})
                cnt["CoC grounding (IoU>=0.5)"] += 1
        elif r["type"] == "counting" and q.startswith("How many") and "visible in front" in q \
                and cnt["CoC counting"] < PER_SKILL:
            add("CoC counting", "num", r, q + NUM, number(a)); cnt["CoC counting"] += 1
        elif r["type"] == "relation" and r["answer_type"] in ("yes", "no") \
                and cnt[f"CoC relation yes/no {r['answer_type']}"] < PER_SKILL // 2:
            add("CoC object-object relation yes/no", "yesno", r, q + YN, r["answer_type"])
            cnt[f"CoC relation yes/no {r['answer_type']}"] += 1
        elif r["type"] == "perception" and q.startswith("Is there a") and q.endswith("in front of the ego vehicle?") \
                and cnt[f"CoC existence {r['answer_type']}"] < PER_SKILL // 2:
            add("CoC existence yes/no", "yesno", r, q + YN, r["answer_type"])
            cnt[f"CoC existence {r['answer_type']}"] += 1
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/eval_set.jsonl", "w") as f:
        for i, it in enumerate(items):
            it["id"] = i
            f.write(json.dumps(it) + "\n")
    print(Counter(it["skill"] for it in items))
    print(f"{len(items)} items -> {OUT}/eval_set.jsonl")


# ------------------------------------------------------------------ run (GPU)
def run(tag, model_dir):
    import base64
    import io
    from PIL import Image
    from vllm import LLM, SamplingParams
    items = [json.loads(l) for l in open(EVAL_SET)]
    llm = LLM(model=model_dir, max_model_len=4096, gpu_memory_utilization=0.6, limit_mm_per_prompt={"image": 1},
              seed=0)
    cache = {}

    def url(p):
        if p not in cache:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            if w * h > MAX_PIXELS:  # same pixel budget as training
                s = math.sqrt(MAX_PIXELS / (w * h))
                im = im.resize((int(w * s), int(h * s)), Image.BICUBIC)
            b = io.BytesIO()
            im.save(b, format="JPEG", quality=95)
            cache[p] = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
        return cache[p]
    convs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(it["image"])}},
                                           {"type": "text", "text": it["question"]}]}] for it in items]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=128),
                    chat_template_kwargs={"enable_thinking": False})
    res = os.path.join(OUT, "results", tag)
    os.makedirs(res, exist_ok=True)
    per = defaultdict(lambda: [0, 0, 0])  # correct, total, unparsable
    with open(f"{res}/predictions.jsonl", "w") as f:
        for it, o in zip(items, outs):
            pred = o.outputs[0].text.strip()
            k = it["kind"]
            if k == "box":
                pb = pred_box(pred, it["W"], it["H"])
                ok, parsed = (pb is not None and iou(pb, it["gold"]) >= 0.5), pb is not None
            else:
                p = {"yesno": yesno, "dir3": dir3, "rel4": rel4, "num": number, "lane6": lane6, "band": band}[k](pred)
                ok, parsed = p == it["gold"], p is not None
            s = per[it["skill"]]
            s[0] += ok
            s[1] += 1
            s[2] += not parsed
            f.write(json.dumps({"id": it["id"], "skill": it["skill"], "pred": pred, "correct": bool(ok)}) + "\n")
    scores = {k: {"accuracy": round(100 * c / t, 1), "n": t, "unparsable": u} for k, (c, t, u) in per.items()}
    tot_c = sum(v[0] for v in per.values()); tot_n = sum(v[1] for v in per.values())
    scores["_overall"] = {"accuracy": round(100 * tot_c / tot_n, 1), "n": tot_n}
    json.dump({"model": model_dir, "scores": scores}, open(f"{res}/scores.json", "w"), indent=2)
    print(json.dumps(scores, indent=1))


# ------------------------------------------------------------------ table
def table():
    order = ["base", "checkpoint-2436", "checkpoint-9744", "final"]
    tags = [t for t in order if os.path.exists(f"{OUT}/results/{t}/scores.json")] + sorted(
        t for t in os.listdir(f"{OUT}/results") if t not in order)
    S = {t: json.load(open(f"{OUT}/results/{t}/scores.json"))["scores"] for t in tags}
    skills = sorted({k for s in S.values() for k in s if k != "_overall"})
    chance = {"yes/no": "50", "direction (left/right/front)": "33", "ego-relative direction": "33",
              "object-object direction": "25", "lane": "17", "distance band": "33"}
    L = ["SPATIAL EVALUATION - held-out cities Delhi + kota_out (never trained on)",
         "accuracy % per skill (n = questions); answers parsed for meaning, not wording", "",
         f'{"skill":<42}{"n":>5}' + "".join(f"{t:>17}" for t in tags)]
    for k in skills:
        n = next(s[k]["n"] for s in S.values() if k in s)
        L.append(f"{k:<42}{n:>5}" + "".join(f'{S[t][k]["accuracy"] if k in S[t] else "-":>17}' for t in tags))
    L.append(f'{"OVERALL":<42}{"":>5}' + "".join(f'{S[t]["_overall"]["accuracy"]:>17}' for t in tags))
    L += ["", "Chance level: yes/no 50, 3-way direction/band 33, 4-way direction 25, 6-way lane 17.",
          "Grounding: predicted box (our tags, Qwen 0-1000, 0-1 or pixels) must overlap the true box with IoU >= 0.5."]
    open(f"{OUT}/spatial_results.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
    elif cmd == "table":
        table()
