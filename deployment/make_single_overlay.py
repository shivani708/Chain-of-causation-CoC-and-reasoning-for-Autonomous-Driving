#!/usr/bin/env python3
"""Re-draw a test video with ONE CoC answer only (vehicle setup, first sentence).

  python3 make_single_overlay.py <video.mp4> <answers.csv> <out.mp4> [--config B_vehicle]
"""
import argparse, bisect, csv, re
import cv2


def first_sentence(text):
    # keep the driving decision + reason; drop the repeated "... sign is observed ..." tail
    m = re.split(r"(?<=[a-z0-9])\.?\s+(?=A (mandatory|cautionary|informatory|regulatory)\b)|(?<=\.)\s", text, maxsplit=1)
    return m[0].strip().rstrip(".") + "."


def wrap(text, width):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines


ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("csv"); ap.add_argument("out")
ap.add_argument("--config", default="B_vehicle")
ap.add_argument("--width", type=int, default=1280)
a = ap.parse_args()

rows = sorted((float(r["time_s"]), first_sentence(r["answer"])) for r in csv.DictReader(open(a.csv))
              if r["config"] == a.config)
times = [t for t, _ in rows]
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 20
W = a.width; H = int(round(cap.get(4) * W / cap.get(3) / 2) * 2)
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
fs = W / 1280
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    t = i / fps
    k = bisect.bisect_right(times, t) - 1
    d = cv2.resize(fr, (W, H))
    if k >= 0:
        lines = wrap("CoC: " + rows[k][1], int(70 / fs * fs))
        lh = int(34 * fs); bh = int(16 * fs) + lh * len(lines)
        ov = d.copy(); cv2.rectangle(ov, (0, H - bh), (W, H), (0, 0, 0), -1)
        d = cv2.addWeighted(ov, 0.6, d, 0.4, 0)
        for j, l in enumerate(lines):
            cv2.putText(d, l, (int(14 * fs), H - bh + int(8 * fs) + lh * (j + 1) - int(8 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85 * fs, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(d)
    i += 1
vw.release()
print("wrote", a.out, i, "frames")
