#!/usr/bin/env python3
"""Run the 72.2 CoC model (merged_rft_ck500) on a test video, the way the vehicle would.

Every --every seconds it builds a 4-frame window ending at that moment and asks the CoC training prompt.
Configs compared on the same moments:
  A_train      4 frames at 4 Hz (span 0.75 s), full frame given to the processor (as in training)
  B_vehicle    4 frames at 10 Hz (span 0.3 s), history 256x144 + current 512x288 (vehicle runtime settings)
  C_train169   like A, but centre-cropped to 16:9 first (training camera was 1280x720)
Greedy decoding (do_sample=False), max_new_tokens 96.
Outputs: <out>/<clip>_answers.csv, <out>/<clip>_overlay.mp4 (config A answers), summary appended to <out>/SUMMARY.txt
"""
import argparse, csv, os, time
from collections import Counter, deque
import cv2, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = ("These are 4 front-view camera frames captured by the ego vehicle in the seconds leading up to a driving "
          "decision, in time order. The last frame, frame 4, is the decision moment (t=0.0s).\n\nYou are given ONLY "
          "these frames -- no speed, trajectory or meta-action channel. Read the ego vehicle's own motion out of how "
          "the scene moves across the frames.\n\nIn one sentence, state what the ego vehicle did at the decision "
          "moment and why, on Indian roads: the driving decision, followed by the reason for it. Cite only what is "
          "already visible in these frames.")
CLASSES = {
    "stop/slow": ["stop", "brake", "slow", "decelerate", "reduce", "yield", "wait", "hold", "creep", "halt"],
    "keep": ["keep", "maintain", "maintaining", "stay", "remain", "continue", "cruise"],
    "speed up": ["accelerate", "resume", "proceed", "speed", "pull", "move off", "go"],
    "lateral": ["nudge", "lane", "adjust", "narrow", "turn", "change", "shift", "steer", "swerve", "merge",
                "overtake", "create"],
}


def decision_class(text):
    for w in text.lower().replace(",", " ").split()[:4]:
        for cls, keys in CLASSES.items():
            if any(w.startswith(k) for k in keys):
                return cls
    return "other"


def pil(bgr, size=None, crop169=False):
    if crop169:
        h, w = bgr.shape[:2]
        if w / h < 16 / 9:
            nh = int(w * 9 / 16); y0 = (h - nh) // 2; bgr = bgr[y0:y0 + nh]
        else:
            nw = int(h * 16 / 9); x0 = (w - nw) // 2; bgr = bgr[:, x0:x0 + nw]
    if size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


@torch.inference_mode()
def ask(model, proc, imgs, max_new):
    content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": PROMPT}]
    inp = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(), \
        int(inp["input_ids"].shape[1])


def wrap(text, width):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    print("model", type(model).__name__, "processor size", proc.image_processor.size, flush=True)
    summ = open(os.path.join(a.out, "SUMMARY.txt"), "a")

    for vp in a.videos:
        clip = os.path.basename(vp).replace("_v2_h264.mp4", "").replace(".mp4", "")
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        g4, g10 = max(1, round(fps / 4)), max(1, round(fps / 10))
        buf = deque(maxlen=3 * g4 + 1)
        step = max(1, round(a.every * fps))
        vw = cv2.VideoWriter(os.path.join(a.out, f"{clip}_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                             (960, 600))
        rows, idx, cur = [], 0, {"A_train": "", "B_vehicle": ""}
        cf = csv.writer(open(os.path.join(a.out, f"{clip}_answers.csv"), "w", newline=""))
        cf.writerow(["clip", "time_s", "frame", "config", "answer", "decision_class", "words", "prompt_tokens",
                     "latency_s"])
        t_start = time.time()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            buf.append(fr)
            if idx >= 3 * g4 and idx % step == 0:
                win4 = [buf[-1 - k * g4] for k in (3, 2, 1, 0)]
                win10 = [buf[-1 - k * g10] for k in (3, 2, 1, 0)]
                cfgs = {"A_train": [pil(f) for f in win4],
                        "B_vehicle": [pil(f, (256, 144)) for f in win10[:3]] + [pil(win10[3], (512, 288))],
                        "C_train169": [pil(f, crop169=True) for f in win4]}
                for name, imgs in cfgs.items():
                    t0 = time.time()
                    ans, ntok = ask(model, proc, imgs, a.max_new_tokens)
                    lat = time.time() - t0
                    r = [clip, round(idx / fps, 2), idx, name, ans, decision_class(ans), len(ans.split()), ntok,
                         round(lat, 2)]
                    cf.writerow(r); rows.append(r)
                    if name in cur:
                        cur[name] = ans
                if len(rows) % 30 == 0:
                    print(f"{clip} t={idx / fps:.0f}s  A: {cur['A_train']}", flush=True)
            if idx % 2 == 0:  # 10 fps overlay video
                d = cv2.resize(fr, (960, 600))
                lines = [("t=%.1fs  72.2 model, training setup (4 frames @4 Hz):" % (idx / fps), (120, 220, 255))]
                lines += [(l, (255, 255, 255)) for l in wrap(cur["A_train"], 80)]
                lines += [("vehicle setup (10 Hz, 256x144 history):", (150, 255, 150))]
                lines += [(l, (210, 210, 210)) for l in wrap(cur["B_vehicle"], 80)]
                bh = 12 + 22 * len(lines)
                ov = d.copy(); cv2.rectangle(ov, (0, 600 - bh), (960, 600), (0, 0, 0), -1)
                d = cv2.addWeighted(ov, 0.6, d, 0.4, 0)
                for i, (t, c) in enumerate(lines):
                    cv2.putText(d, t, (10, 600 - bh + 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
                vw.write(d)
            idx += 1
        vw.release()

        out = [f"\n=== {clip}  ({idx / fps:.0f} s video, {fps:.0f} fps, decision every {a.every:.0f} s, "
               f"run time {time.time() - t_start:.0f} s) ==="]
        for name in ("A_train", "B_vehicle", "C_train169"):
            rs = [r for r in rows if r[3] == name]
            if not rs:
                continue
            cls = Counter(r[5] for r in rs)
            ans = Counter(r[4] for r in rs)
            first3 = Counter(" ".join(r[4].split()[:3]).lower() for r in rs)
            out.append(f"\n[{name}]  windows {len(rs)}   unique answers {len(ans)}   "
                       f"mean words {sum(r[6] for r in rs) / len(rs):.1f}   mean latency {sum(r[8] for r in rs) / len(rs):.2f} s"
                       f"   prompt tokens {rs[0][7]}")
            out.append("  decision classes: " + ", ".join(f"{k} {v} ({100 * v / len(rs):.0f}%)" for k, v in cls.most_common()))
            out.append("  top openings: " + "; ".join(f"'{k}' x{v}" for k, v in first3.most_common(8)))
            out.append("  most repeated full answers:")
            out += [f"    x{v}: {k}" for k, v in ans.most_common(5)]
            out.append("  sample answers:")
            out += [f"    t={r[1]:>6}s  {r[4]}" for r in rs[::max(1, len(rs) // 8)][:8]]
        txt = "\n".join(out)
        print(txt, flush=True); summ.write(txt + "\n"); summ.flush()


if __name__ == "__main__":
    main()
