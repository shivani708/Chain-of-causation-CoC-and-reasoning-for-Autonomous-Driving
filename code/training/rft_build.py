# Rejection-sampling fine-tuning (RFT) data from the 68.6 model's OWN answers on LingoQA TRAIN questions.
#   questions : pick ~8k LingoQA-train questions (+2k format prompts on action segments)      [CPU]
#   generate  : 68.6 model, vLLM, 8 samples per question, temperature 0.7, same chat format as the eval  [GPU, venv]
#   judge     : Lingo-Judge (same classifier as the benchmark) vs the TRAIN reference answers  [GPU, vqa_db]
#   assemble  : keep the best answer per question if judge score > 0 -> data/rft/rft_train.jsonl  [CPU]
# LingoQA val is never touched (train split only).
import collections, json, os, random, re, sys
import pandas as pd

B = "/data/common_datasets/ai24mtech02003/24_dataset_coc"
OUT = f"{B}/data/rft"
L = "/data/common_datasets/LingoQA/extracted"
IMG = {"action": f"{L}/images_action", "scenery": f"{L}/images_scenery/images"}
FORMAT_Q = 'What is the current action and its justification? Answer in the form "action, justification".'
N_Q, N_FMT, PER_Q_CAP, N_SAMPLES = 8000, 2000, 400, 8
os.makedirs(OUT, exist_ok=True)


def frames(sub, images):
    rels = sorted((str(p) for p in images), key=lambda p: int(re.findall(r"(\d+)\.jpg", p)[0]))
    return [os.path.join(IMG[sub], p[len("images/"):]) for p in rels]


def questions():
    random.seed(5)
    rows = []
    for sub in ["action", "scenery"]:
        df = pd.read_parquet(f"{L}/images_{sub}/train.parquet").drop_duplicates(["segment_id", "question"])
        for r in df.itertuples():
            fr = frames(sub, r.images)
            if len(fr) == 5:
                rows.append({"segment_id": r.segment_id, "question": r.question.strip(), "reference": r.answer.strip(),
                             "frames": fr, "kind": "plain"})
    random.shuffle(rows)
    per_q, sel, fmt = collections.Counter(), [], []
    for r in rows:
        if re.search(r"(currently doing and why|action are you currently taking and why|driving action and it's justification|current action and why)", r["question"], re.I) \
                and len(fmt) < N_FMT:
            fmt.append({**r, "question": FORMAT_Q, "kind": "format"}); continue
        if per_q[r["question"]] < PER_Q_CAP and len(sel) < N_Q:
            per_q[r["question"]] += 1; sel.append(r)
    allq = sel + fmt
    for i, r in enumerate(allq):
        r["qid"] = i
    with open(f"{OUT}/questions.jsonl", "w") as f:
        for r in allq: f.write(json.dumps(r) + "\n")
    print(f"questions: {len(sel)} plain ({len(per_q)} distinct texts) + {len(fmt)} format")


def generate(model_dir):
    import base64, io
    from PIL import Image
    from vllm import LLM, SamplingParams
    qs = [json.loads(l) for l in open(f"{OUT}/questions.jsonl")]
    llm = LLM(model=model_dir, max_model_len=16384, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 5}, seed=0)
    sp = SamplingParams(n=N_SAMPLES, temperature=0.7, top_p=0.95, max_tokens=256, seed=0)
    def url(p):
        b = io.BytesIO(); Image.open(p).convert("RGB").save(b, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    with open(f"{OUT}/samples.jsonl", "w") as f:
        for s in range(0, len(qs), 500):
            chunk = qs[s:s + 500]
            convs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(p)}} for p in q["frames"]]
                       + [{"type": "text", "text": q["question"]}]}] for q in chunk]
            outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
            for q, o in zip(chunk, outs):
                f.write(json.dumps({"qid": q["qid"], "samples": [c.text.strip() for c in o.outputs]}) + "\n")
            print(f"generated {s + len(chunk)}/{len(qs)}", flush=True)


def judge():
    sys.path.insert(0, "/home/ai24mtech02003/vqa/LingoQA/benchmark")
    import torch
    from judge import LingoJudge
    j = LingoJudge().eval().to("cuda:0")
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{OUT}/questions.jsonl")}
    with open(f"{OUT}/judged.jsonl", "w") as f:
        for n, l in enumerate(open(f"{OUT}/samples.jsonl")):
            s = json.loads(l); q = qs[s["qid"]]
            preds = s["samples"]
            sc = j.compute([q["question"] if q["kind"] == "plain" else "What are you currently doing and why?"] * len(preds),
                           [[q["reference"]]] * len(preds), preds).tolist()
            f.write(json.dumps({"qid": s["qid"], "scores": sc}) + "\n")
            if n % 1000 == 0:
                print("judged", n, flush=True)


def assemble():
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{OUT}/questions.jsonl")}
    sm = {json.loads(l)["qid"]: json.loads(l)["samples"] for l in open(f"{OUT}/samples.jsonl")}
    st = collections.Counter(); out = []; pass_rate = []
    for l in open(f"{OUT}/judged.jsonl"):
        j = json.loads(l); q = qs[j["qid"]]; samples = sm[j["qid"]]
        ok = [(sc, a) for sc, a in zip(j["scores"], samples) if sc > 0 and a and not a.lower().startswith("action, justification")]
        pass_rate.append(len(ok) / len(samples))
        if not ok:
            st[f"no correct sample ({q['kind']})"] += 1; continue
        sc, a = max(ok)
        out.append({"frames": q["frames"], "prompt": q["question"], "answer": a, "crop": None, "hflip": False,
                    "resize": None, "source": f"rft_{q['kind']}", "judge": round(sc, 3)})
        st[f"kept ({q['kind']})"] += 1
    random.seed(3); random.shuffle(out)
    n_val = 300
    for name, rows in [("rft_train.jsonl", out[n_val:]), ("rft_val.jsonl", out[:n_val])]:
        with open(f"{OUT}/{name}", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    words = sum(len(r["answer"].split()) for r in out) / max(1, len(out))
    L_ = ["RFT data: the 68.6 model's own best-judged answers on LingoQA TRAIN questions", "",
          f"questions: {len(qs)}   samples per question: {N_SAMPLES}   mean pass rate per question: {sum(pass_rate) / len(pass_rate):.3f}",
          f"kept: {len(out)} (train {len(out) - n_val}, val {n_val})   mean answer words: {words:.1f}"] + \
         [f"  {v:6d}  {k}" for k, v in st.most_common()]
    open(f"{OUT}/RFT_summary.txt", "w").write("\n".join(L_) + "\n"); print("\n".join(L_))
    for r in out[:4]:
        print("EX", r["prompt"][:60], "=>", r["answer"][:140])


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"questions": questions, "judge": judge, "assemble": assemble}.get(cmd, lambda: generate(sys.argv[2]))()
