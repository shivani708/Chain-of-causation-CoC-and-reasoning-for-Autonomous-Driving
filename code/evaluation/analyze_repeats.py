"""Compare the repeated LingoQA runs of merged_rft_ck500: scores and whether the answers change between runs."""
import csv, glob, os, re, statistics
from collections import Counter
D = "/data/common_datasets/ai24mtech02003/24_dataset_coc/eval/repeat_rft_ck500"
ORIG = "/data/common_datasets/ai24mtech02003/24_dataset_coc/eval/lingoqa/lora_rft_v1_checkpoint-500.csv"
scores = {}
for line in open(os.path.join(D, "scores.txt")):
    m = re.match(r"(run_\d+) .*score is ([\d.]+)%", line)
    if m:
        scores[m.group(1)] = round(float(m.group(2)), 1)
def load(p):
    return {(r["question_id"], r["segment_id"]): r["answer"].strip() for r in csv.DictReader(open(p))}
runs = {os.path.basename(p)[:-4]: load(p) for p in sorted(glob.glob(os.path.join(D, "run_*.csv")))
        if os.path.basename(p)[:-4] in scores}
orig = load(ORIG)
keys = sorted(orig)
uniq = Counter()
changed = []
for k in keys:
    answers = Counter(r[k] for r in runs.values() if k in r)
    uniq[len(answers)] += 1
    if len(answers) > 1:
        changed.append((k, answers))
vals = list(scores.values())
lines = [
    "REPEATED LingoQA EVALUATION OF THE 72.2 MODEL (merged_rft_ck500)",
    "same harness as the 72.2 result: generate_predictions.py (greedy, do_sample=False, 5 frames,"
    " max_new_tokens 256) + Lingo-Judge, 500 val questions",
    f"runs completed: {len(vals)}",
    f"LingoQA score: min {min(vals):.1f}  max {max(vals):.1f}  mean {statistics.mean(vals):.2f}"
    f"  std {statistics.pstdev(vals):.2f}",
    "score distribution: " + ", ".join(f"{s:.1f} x{c}" for s, c in sorted(Counter(vals).items())),
    f"questions with the SAME answer in every run: {uniq[1]}/{len(keys)}",
    f"questions whose answer changed between runs: {len(keys) - uniq[1]}/{len(keys)}"
    + ("  (unique answers per question: " + ", ".join(f"{n}:{c}" for n, c in sorted(uniq.items()) if n > 1) + ")"
       if len(keys) - uniq[1] else ""),
    "runs identical to the original 72.2 predictions (all 500 answers): "
    f"{sum(1 for r in runs.values() if all(r.get(k) == orig[k] for k in keys))}/{len(runs)}",
    "",
    "per run: " + "  ".join(f"{t[4:]}:{s:.1f}" for t, s in sorted(scores.items())),
]
for k, a in changed[:10]:
    lines.append(f"\nCHANGED {k[0]}:")
    for ans, c in a.most_common(3):
        lines.append(f"   x{c}: {ans[:200]}")
open(os.path.join(D, "REPEAT_SUMMARY.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
