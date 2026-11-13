#!/usr/bin/env python3
"""Build the speed-bump-fix LoRA data.

Drops every CoC training sample whose answer mentions a speed bump but where Qwen3.5-27B saw no speed bump
(verdict NO or UNCLEAR, data/clean_sb/verdicts.jsonl). Then: 16,000 random cleaned CoC + all 8,190 RFT LingoQA
answers (keeps the 72.2 LingoQA skill).

Writes data/sbfix/coc_sbclean.jsonl, data/sbfix/train.jsonl, data/sbfix/SBFIX_summary.txt
"""
import json, os, random, re
from collections import Counter

B = "/data/common_datasets/ai24mtech02003/24_dataset_coc"
OUT = f"{B}/data/sbfix"
N_COC = 16000
SB = re.compile(r"speed ?(bump|breaker|hump)", re.I)
os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open(f"{B}/data/coc_train.jsonl")]
ver = {}
for l in open(f"{B}/data/clean_sb/verdicts.jsonl"):
    v = json.loads(l)
    if v["group"] == "speed_bump":
        ver[v["idx"]] = v["verdict"]
n_sb = sum(bool(SB.search(r["answer"])) for r in rows)
assert len(ver) == n_sb, f"verdicts {len(ver)} != speed-bump samples {n_sb}"

drop = {i for i, v in ver.items() if v != "YES"}
clean = [r for i, r in enumerate(rows) if i not in drop]
with open(f"{OUT}/coc_sbclean.jsonl", "w") as f:
    for r in clean:
        f.write(json.dumps(r) + "\n")

random.seed(0)
coc = [{"frames": r["frames"], "prompt": r["prompt"], "answer": r["answer"], "crop": None, "hflip": False,
        "resize": None, "source": "coc_sbclean"} for r in random.sample(clean, N_COC)]
rft = [json.loads(l) for l in open(f"{B}/data/rft/rft_train.jsonl")]
train = coc + rft
random.shuffle(train)
with open(f"{OUT}/train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")


def first2(rs):
    c = Counter(" ".join(r["answer"].split()[:2]).lower() for r in rs)
    return ", ".join(f"{k} {100 * v / len(rs):.1f}%" for k, v in c.most_common(8))


sb_before = n_sb / len(rows)
sb_after = sum(bool(SB.search(r["answer"])) for r in clean) / len(clean)
txt = f"""SPEED-BUMP FIX DATA
CoC train samples           {len(rows)}
speed-bump answers          {n_sb} ({100 * sb_before:.1f}%)
  27B saw a bump (kept)     {sum(v == 'YES' for v in ver.values())}
  no bump seen (dropped)    {len(drop)}
cleaned CoC                 {len(clean)}  speed-bump share {100 * sb_after:.2f}%
LoRA train set              {len(train)} = {N_COC} cleaned CoC + {len(rft)} RFT LingoQA
answer openings before      {first2(rows)}
answer openings after       {first2(clean)}
"""
open(f"{OUT}/SBFIX_summary.txt", "w").write(txt)
print(txt)
