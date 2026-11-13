#!/usr/bin/env python3
"""Real-time single-CoC display with the 73.2 model (merged_sbfix_ck1500, Qwen3-VL-2B).

Reads a camera / video / stream, keeps the last second of frames and, whenever the model is free, asks the CoC
question on a 4-frame window ending at "now". Only ONE line is shown on the video:  "CoC: <decision + reason>".
The model runs in a background thread, so the video never freezes. Every answer goes to a CSV log.

Frame setups (--setup):
  vehicle (default)  4 frames 0.1 s apart (10 Hz), frames 1-3 at 256x144, last frame at 512x288
  train              4 frames 0.25 s apart (4 Hz), full frames (as in training)

Keys (video window):  p = pause/resume   q / ESC = quit

ADVISORY ONLY - never use the output to steer, brake or accelerate the vehicle.

Examples
  python3 live_coc_single.py                                   # webcam 0
  python3 live_coc_single.py --source 1                        # second camera
  python3 live_coc_single.py --source Hyderabad1.mp4           # recorded video, played in real time
  python3 live_coc_single.py --source rtsp://192.168.1.10/stream
  python3 live_coc_single.py --source drive.mp4 --no_display --out_video drive_coc.mp4   # headless, save video
"""
import argparse
import csv
import os
import queue
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "model", "merged_sbfix_ck1500")

# exact CoC training prompt - do not edit
PROMPT = ("These are 4 front-view camera frames captured by the ego vehicle in the seconds leading up to a driving "
          "decision, in time order. The last frame, frame 4, is the decision moment (t=0.0s).\n\nYou are given ONLY "
          "these frames -- no speed, trajectory or meta-action channel. Read the ego vehicle's own motion out of how "
          "the scene moves across the frames.\n\nIn one sentence, state what the ego vehicle did at the decision "
          "moment and why, on Indian roads: the driving decision, followed by the reason for it. Cite only what is "
          "already visible in these frames.")
SETUPS = {"vehicle": 0.1, "train": 0.25}   # seconds between the 4 frames


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", default="0", help="camera index, video file or stream URL")
    ap.add_argument("--setup", default="vehicle", choices=list(SETUPS))
    ap.add_argument("--ask_every", type=float, default=0, help="min seconds between questions (0 = as fast as possible)")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--threads", type=int, default=16, help="CPU threads")
    ap.add_argument("--log", default=os.path.join(HERE, "live_coc_answers.csv"))
    ap.add_argument("--display_width", type=int, default=1280)
    ap.add_argument("--no_display", action="store_true", help="no window (headless / ssh)")
    ap.add_argument("--out_video", default=None, help="also save the shown video to this mp4")
    return ap.parse_args()


def load_model(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if device == "cpu":
        torch.set_num_threads(args.threads)
    print(f"[load] {args.model}  device={device}  dtype={dtype}")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"[load] ready in {time.time() - t0:.1f} s")
    return model, processor


@torch.inference_mode()
def ask_coc(model, processor, images, max_new_tokens):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": PROMPT}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                           add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(model.device)
    # greedy: the model's generation_config has sampling on, so it must be switched off here
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def first_sentence(text):
    """Decision + reason only; drops the repeated '... A mandatory/cautionary sign is observed ...' tail."""
    m = re.split(r"(?<=[a-z0-9])\.?\s+(?=A (mandatory|cautionary|informatory|regulatory)\b)|(?<=\.)\s", text,
                 maxsplit=1)
    return m[0].strip().rstrip(".") + "."


def to_pil(bgr, size=None):
    if size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pick_window(buf, gap):
    """4 frames ending at the newest one, `gap` seconds apart (nearest stored frame). None if not enough history."""
    t_now = buf[-1][0]
    if t_now - buf[0][0] < 3 * gap - 0.02:
        return None
    picks = []
    for k in (3, 2, 1, 0):
        target = t_now - k * gap
        picks.append(min(buf, key=lambda tf: abs(tf[0] - target))[1])
    return picks


def build_images(frames, setup):
    if setup == "vehicle":
        return [to_pil(f, (256, 144)) for f in frames[:3]] + [to_pil(frames[3], (512, 288))]
    return [to_pil(f) for f in frames]


def wrap(text, width):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines


def draw(frame, coc, status):
    h, w = frame.shape[:2]
    fs = w / 1280
    lines = wrap("CoC: " + coc, 70) if coc else []
    lh = int(34 * fs)
    bh = int(16 * fs) + lh * max(1, len(lines))
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - bh), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
    for j, l in enumerate(lines or [status]):
        cv2.putText(frame, l, (int(14 * fs), h - bh + lh * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * fs,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"ADVISORY ONLY   {status}", (int(12 * fs), int(30 * fs)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * fs, (0, 200, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    model, processor = load_model(args)
    gap = SETUPS[args.setup]

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open video source {args.source!r}")
    is_file = isinstance(src, str) and os.path.isfile(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    jobs, results = queue.Queue(maxsize=1), queue.Queue()

    def worker():
        while True:
            job = jobs.get()
            if job is None:
                return
            t_video, frames = job
            t0 = time.time()
            try:
                text = ask_coc(model, processor, build_images(frames, args.setup), args.max_new_tokens)
            except Exception as e:  # keep the live view running
                text = f"[error] {e}"
            results.put((t_video, text, time.time() - t0))

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    new_log = not os.path.exists(args.log)
    logf = open(args.log, "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["wall_time", "source", "video_time_s", "setup", "coc", "full_answer", "latency_s"])

    buf = deque()                     # (time_s, bgr frame), last ~1 s
    coc, status, busy, paused = "", "collecting frames...", False, False
    last_ask, n_frame, writer = -1e9, 0, None
    t_start = time.time()
    win = "73.2 model - live CoC"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:
                break
            time.sleep(0.02); continue
        # file: video clock, played at real speed; camera/stream: wall clock
        t = n_frame / fps if is_file else time.time() - t_start
        n_frame += 1
        if is_file:
            lag = t - (time.time() - t_start)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5 and n_frame % 2:   # running behind real time: drop frames like a live camera would
                continue
        buf.append((t, frame))
        while buf and t - buf[0][0] > 3 * gap + 0.3:
            buf.popleft()

        while not results.empty():
            t_video, text, lat = results.get()
            busy = False
            coc = first_sentence(text)
            status = f"{args.setup} setup, last answer {lat:.1f} s"
            log.writerow([datetime.now().isoformat(timespec="seconds"), args.source, f"{t_video:.2f}", args.setup,
                          coc, text, f"{lat:.2f}"])
            logf.flush()
            print(f"[t={t_video:7.1f}s  {lat:4.1f}s] {coc}", flush=True)

        if not busy and not paused and time.time() - last_ask >= args.ask_every:
            window = pick_window(buf, gap)
            if window is not None:
                jobs.put((t, [f.copy() for f in window]))
                busy, last_ask = True, time.time()

        disp = frame
        if frame.shape[1] != args.display_width:
            s = args.display_width / frame.shape[1]
            disp = cv2.resize(frame, (args.display_width, int(round(frame.shape[0] * s / 2) * 2)))
        draw(disp, coc, status + ("  [paused]" if paused else ""))
        if args.out_video:
            if writer is None:
                writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (disp.shape[1], disp.shape[0]))
            writer.write(disp)
        if not args.no_display:
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused

    try:                              # let the running answer finish, then stop the worker cleanly
        jobs.put(None, timeout=60)
        th.join(timeout=60)
    except queue.Full:
        pass
    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    logf.close()
    print(f"[live] answers logged to {args.log}")


if __name__ == "__main__":
    main()
