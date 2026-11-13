#!/usr/bin/env python3
"""Check every CoC training sample whose answer mentions a speed bump with Qwen3.5-27B (vision):
is a speed bump actually visible in the 4 frames?  Also checks 300 random keep/maintain samples as a control.

Writes  data/clean_sb/verdicts.jsonl  (one line per checked sample: idx, group, city, answer, verdict, raw)
        data/clean_sb/VERIFY_SUMMARY.txt
"""
import json, os, random, re, sys, time
from collections import Counter

B = "/data/common_datasets/ai24mtech02003/24_dataset_coc"
JUDGE = "/data/common_datasets/ai24mtech02003/vqa_paper/models/qwen3.5-27b"
OUT = f"{B}/data/clean_sb"
SB = re.compile(r"speed ?(bump|breaker|hump)", re.I)
Q = ("These are 4 front-camera frames from a car driving on an Indian road, in time order (the last frame is now).\n\n"
     "Is there a speed bump (speed breaker / road hump: a raised strip or hump across the road surface, usually "
     "painted with stripes, lying across the car's driving lane) on the road ahead of the car or directly under it "
     "in any of these frames?\n"
     "Painted kerbs, dividers or crash barriers along the edge of the road are NOT speed bumps. A flat zebra "
     "crossing or flat painted lane marking is NOT a speed bump.\n\n"
     "Answer with exactly one word: YES or NO.")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{B}/data/coc_train.jsonl")]
    sb = [(i, "speed_bump") for i, r in enumerate(rows) if SB.search(r["answer"])]
    keep = [i for i, r in enumerate(rows) if r["answer"].lower().startswith(("maintain", "keep lane", "keep speed"))
            and not SB.search(r["answer"])]
    random.seed(0)
    todo = sb + [(i, "control_keep") for i in random.sample(keep, 300)]
    if limit:
        random.shuffle(todo); todo = todo[:limit]
    print(f"to check: {len(todo)}  (speed bump {len(sb)}, control 300)", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=JUDGE, max_model_len=12288, gpu_memory_utilization=0.78, max_num_batched_tokens=8192, max_num_seqs=16, enforce_eager=True, limit_mm_per_prompt={"image": 4},
              allowed_local_media_path="/data/common_datasets", seed=0,
              mm_processor_kwargs={"max_pixels": 1280 * 720})
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    out_path = f"{OUT}/verdicts{'_pilot' if limit else ''}.jsonl"
    done = {}
    if os.path.exists(out_path):                      # resume: keep what is already judged
        for l in open(out_path):
            d = json.loads(l); done[(d["idx"], d["group"])] = d
    todo_left = [t for t in todo if t not in done]
    print(f"already judged {len(done)}, left {len(todo_left)}", flush=True)
    t0 = time.time(); CH = 400
    with open(out_path, "a") as f:
        for c in range(0, len(todo_left), CH):
            chunk = todo_left[c:c + CH]
            msgs = []
            for i, _ in chunk:
                content = [{"type": "image_url", "image_url": {"url": "file://" + fr}} for fr in rows[i]["frames"]]
                content.append({"type": "text", "text": Q})
                msgs.append([{"role": "user", "content": content}])
            outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            for (i, g), o in zip(chunk, outs):
                raw = o.outputs[0].text.strip()
                v = "YES" if raw.upper().startswith("YES") else "NO" if raw.upper().startswith("NO") else "UNCLEAR"
                d = {"idx": i, "group": g, "city": rows[i]["city"], "answer": rows[i]["answer"],
                     "last_frame": rows[i]["frames"][-1], "verdict": v, "raw": raw}
                done[(i, g)] = d; f.write(json.dumps(d) + "\n")
            f.flush()
            print(f"judged {len(done)}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)
            del msgs, outs
    stats = Counter((d["group"], d["verdict"]) for d in done.values())
    lines = [f"Qwen3.5-27B speed-bump check of CoC training answers ({time.strftime('%d %b %Y %H:%M')})"]
    for g in ("speed_bump", "control_keep"):
        n = sum(v for (gg, _), v in stats.items() if gg == g)
        if n:
            lines.append(f"{g:13s} n={n:5d}  " + "  ".join(f"{v} {stats[(g, v)]} ({100 * stats[(g, v)] / n:.1f}%)"
                                                     for v in ("YES", "NO", "UNCLEAR")))
    txt = "\n".join(lines); print(txt, flush=True)
    open(f"{OUT}/VERIFY_SUMMARY{'_pilot' if limit else ''}.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
