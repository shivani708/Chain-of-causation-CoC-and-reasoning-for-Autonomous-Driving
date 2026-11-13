#!/usr/bin/env python3
"""Live front-camera VQA with the final 73.2 LingoQA model (merged_sbfix_ck1500, Qwen3-VL-2B).

Shows the camera live, keeps the last N frames (sampled every --frame_interval s, resized to 512x288 like the
training data) and asks the model a question about them. Inference runs in a background thread, so the video never
freezes; the latest answer is drawn on the frame and every answer is logged to CSV.

Keys (video window):  SPACE = ask now   n = next preset question   p = pause/resume auto-asking   q / ESC = quit
Terminal: type any question + Enter to ask it on the current frames.

Works on CPU (Ryzen 7950X: bf16, 16 threads, roughly 5-15 s per answer) or an NVIDIA GPU (~1 s).
This is an advisory / explanation model - never use its output to control the vehicle.

Examples
  python live_camera_vqa.py                                  # webcam 0
  python live_camera_vqa.py --source 1                       # second camera
  python live_camera_vqa.py --source drive.mp4               # recorded video
  python live_camera_vqa.py --source rtsp://192.168.1.10/stream
  python live_camera_vqa.py --question "Is it safe to overtake now?" --ask_every 10
"""
import argparse
import csv
import os
import queue
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
PRESET_QUESTIONS = [
    "What are you paying attention to and why?",
    "What is the current action and its justification? Answer in the form \"action, justification\"",
    "Are there any pedestrians or two-wheelers you need to watch out for?",
    "Is it safe to accelerate in this situation? Why?",
    "What should the driver do next and why?",
    "Are there any traffic lights or traffic signs? What do they indicate?",
]
IMG_W, IMG_H = 512, 288  # training resolution


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", default="0", help="camera index, video file or stream URL")
    ap.add_argument("--num_frames", type=int, default=5, help="frames given to the model (LingoQA used 5)")
    ap.add_argument("--frame_interval", type=float, default=1.0, help="seconds between stored frames")
    ap.add_argument("--ask_every", type=float, default=0, help="auto-ask every N s (0 = after each answer)")
    ap.add_argument("--question", default=None, help="fixed question (default: rotate preset questions)")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="bf16 is fastest on Zen 4 / GPU")
    ap.add_argument("--threads", type=int, default=16, help="CPU threads (7950X: 16 physical cores)")
    ap.add_argument("--log", default=os.path.join(HERE, "live_answers.csv"))
    ap.add_argument("--save_frames", action="store_true", help="save the frames used for every answer")
    ap.add_argument("--display_width", type=int, default=1280)
    return ap.parse_args()


def load_model(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if device == "cpu":
        torch.set_num_threads(args.threads)
    print(f"[load] {args.model}  device={device}  dtype={dtype}  threads={torch.get_num_threads()}")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"[load] ready in {time.time() - t0:.1f} s")
    return model, processor, device


@torch.inference_mode()
def answer(model, processor, device, frames, question, max_new_tokens):
    content = [{"type": "image", "image": f} for f in frames] + [{"type": "text", "text": question}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return text.strip()


def to_model_frame(bgr):
    """Same preprocessing as training: 16:9 centre crop, then resize to 512x288."""
    h, w = bgr.shape[:2]
    if w / h > 16 / 9:
        nw = int(h * 16 / 9); x0 = (w - nw) // 2; bgr = bgr[:, x0:x0 + nw]
    else:
        nh = int(w * 9 / 16); y0 = (h - nh) // 2; bgr = bgr[y0:y0 + nh]
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines


def draw_overlay(frame, state):
    h, w = frame.shape[:2]
    lines = [("Q: " + state["question"], (120, 220, 255))] if state["question"] else []
    lines += [(l, (255, 255, 255)) for l in wrap("A: " + state["answer"], max(40, w // 13))] if state["answer"] else []
    status = state["status"] + (f"   last: {state['latency']:.1f} s" if state["latency"] else "")
    lines.append((status + ("   [auto-ask paused]" if state["paused"] else ""), (150, 255, 150)))
    box_h = 14 + 26 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (12, h - box_h + 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 1, cv2.LINE_AA)
    cv2.putText(frame, "ADVISORY ONLY - not a vehicle controller", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 200, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    model, processor, device = load_model(args)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open video source {args.source!r}")
    is_file = isinstance(src, str) and os.path.isfile(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    buf = deque(maxlen=args.num_frames)          # model frames (PIL, 512x288)
    jobs, results = queue.Queue(maxsize=1), queue.Queue()
    state = {"question": "", "answer": "", "status": "collecting frames...", "latency": 0.0, "paused": False,
             "busy": False}
    q_idx = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    new_log = not os.path.exists(args.log)
    logf = open(args.log, "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["time", "question", "answer", "latency_s", "frames_dir"])

    def worker():
        while True:
            job = jobs.get()
            if job is None:
                return
            frames, question = job
            t0 = time.time()
            try:
                text = answer(model, processor, device, frames, question, args.max_new_tokens)
            except Exception as e:  # keep the live view running
                text = f"[error] {e}"
            results.put((question, text, time.time() - t0, frames))

    def terminal_input():
        for line in sys.stdin:
            if line.strip():
                typed.put(line.strip())

    typed = queue.Queue()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=terminal_input, daemon=True).start()
    print("[live] SPACE ask now | n next question | p pause auto | q quit | or type a question in this terminal")

    def submit(question):
        if state["busy"] or len(buf) < args.num_frames:
            return False
        try:
            jobs.put_nowait((list(buf), question))
        except queue.Full:
            return False
        state.update(busy=True, question=question, status=f"thinking on {device}...")
        return True

    last_store, last_ask = 0.0, 0.0
    win = "73.2 model - live VQA"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:
                break
            time.sleep(0.05); continue
        now = time.time()
        if now - last_store >= args.frame_interval:
            buf.append(to_model_frame(frame)); last_store = now
            if len(buf) < args.num_frames:
                state["status"] = f"collecting frames {len(buf)}/{args.num_frames}..."

        while not results.empty():
            question, text, lat, frames = results.get()
            state.update(answer=text, latency=lat, busy=False, status="ready")
            fdir = ""
            if args.save_frames:
                fdir = os.path.join(HERE, "live_frames", datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(fdir, exist_ok=True)
                for i, f in enumerate(frames):
                    f.save(os.path.join(fdir, f"frame_{i}.jpg"))
            log.writerow([datetime.now().isoformat(timespec="seconds"), question, text, f"{lat:.2f}", fdir])
            logf.flush()
            print(f"\n[{lat:.1f}s] Q: {question}\n        A: {text}")
            last_ask = now

        if not typed.empty():
            submit(typed.get())
        elif not state["paused"] and not state["busy"] and now - last_ask >= args.ask_every:
            if submit(args.question or PRESET_QUESTIONS[q_idx % len(PRESET_QUESTIONS)]) and not args.question:
                q_idx += 1

        disp = frame
        if frame.shape[1] != args.display_width:
            s = args.display_width / frame.shape[1]
            disp = cv2.resize(frame, (args.display_width, int(frame.shape[0] * s)))
        draw_overlay(disp, state)
        cv2.imshow(win, disp)
        key = cv2.waitKey(max(1, int(1000 / fps)) if is_file else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            submit(args.question or PRESET_QUESTIONS[q_idx % len(PRESET_QUESTIONS)])
        elif key == ord("n"):
            q_idx += 1
            state["question"] = PRESET_QUESTIONS[q_idx % len(PRESET_QUESTIONS)] + "  (press SPACE)"
        elif key == ord("p"):
            state["paused"] = not state["paused"]

    try:
        jobs.put_nowait(None)
    except queue.Full:
        pass
    cap.release(); cv2.destroyAllWindows(); logf.close()
    print(f"[live] answers logged to {args.log}")


if __name__ == "__main__":
    main()
