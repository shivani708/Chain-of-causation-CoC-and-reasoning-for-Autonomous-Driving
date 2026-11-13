"""
Full fine-tune Qwen3-VL-2B-Instruct for general VQA capability (object/scene
understanding + temporal/motion reasoning), staged to avoid catastrophic
forgetting between the two data sources:

  Stage 1 (static):   --static_data_dir only. One QA example per image
                       (object/scene existence, hazards, etc.) from the
                       stage2_qa city JSON files.
  Stage 2 (temporal):  --static_data_dir (capped, replayed) + --temporal_data_dir
                       (method3 multi-frame motion/behavior QA), continuing
                       from stage 1's saved weights via --model <stage1_dir>.
                       Replaying a slice of static data alongside the new
                       temporal data (rather than pure sequential-only-new-data
                       fine-tuning) is what keeps stage 1's gains from being
                       overwritten -- a full fine-tune with no replay drifts
                       hard toward whatever it's shown last.

Both source formats are read directly (no intermediate per-example files):
  static:   {city}_vqa.json -> {"city", "records": [{"image_path", "qa_pairs":
            [{"question","answer"}]}]}  (1 image per record)
  temporal: <city>/stride*/templates/window_*.json -> {"city", "frames": [...],
            "qa_pairs": [{"question","answer"}]}  (12 frames per window)

Delhi is excluded from static training by default (project-wide policy: Delhi
is reserved for held-out evaluation only, never trained on, to keep the later
CoC-weights eval on Delhi leakage-free). kota_out (static) and Dasuya
(temporal) are the smallest cities in each source and are held out the same
way, purely to get a cheap in-distribution validation signal.
"""

import os
import json
import glob
import random
import re
import argparse
import time
from datetime import datetime

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 384 * 28 * 28


_IMG_EXT_FALLBACKS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _resolve_image_path(path):
    """Source images were converted png -> jpg on disk after our stage1/2
    JSONs were generated, so a recorded image_path's extension can be stale.
    Try the path as-is, then swap the extension across common variant s."""
    if os.path.exists(path):
        return path
    base, _ = os.path.splitext(path)
    for ext in _IMG_EXT_FALLBACKS:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return None


def load_static_records(json_dir, cities=None, exclude_cities=None,
                         max_samples=None, seed=42, split="train"):
    """Load stage2_qa-format city files (1 image + several qa_pairs per record)
    into flat {"frames": [image_path], "prompt", "answer"} examples.

    cities: if given, ONLY these cities are included (used to build a held-out
      val set). exclude_cities: if given, these are skipped (used to hold
      them out of training). The two are mutually exclusive per call.
    """
    rng = random.Random(seed)
    examples = []
    n_seen = 0
    missing_img = 0
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    seen_cities = []

    # Reservoir sampling when max_samples is set: bounds peak memory to
    # O(max_samples) instead of materializing the full corpus (which can be
    # tens of GB for the ~8.5M-example static set) before discarding most of
    # it. Each per-city JSON is still parsed whole (json.load has no partial
    # mode), but it's dropped once its records are folded into the reservoir,
    # so peak memory is the largest single city file, not the corpus sum.
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        city = d.get("city", os.path.basename(fp))
        if cities is not None and city not in cities:
            continue
        if exclude_cities is not None and city in exclude_cities:
            continue
        seen_cities.append(city)

        for rec in d.get("records", []):
            img = rec.get("image_path")
            img = _resolve_image_path(img) if img else None
            if not img:
                missing_img += 1
                continue
            for qa in rec.get("qa_pairs", []):
                q, a = qa.get("question"), qa.get("answer")
                if not q or not a:
                    continue
                item = {"frames": [img], "prompt": q, "answer": a}
                if max_samples:
                    n_seen += 1
                    if len(examples) < max_samples:
                        examples.append(item)
                    else:
                        j = rng.randint(0, n_seen - 1)
                        if j < max_samples:
                            examples[j] = item
                else:
                    examples.append(item)
        del d

    total = n_seen if max_samples else len(examples)
    print(f"  [{split}] static {json_dir}: cities={seen_cities} -> "
          f"{total} QA examples ({missing_img} missing images)")
    if max_samples and total > len(examples):
        print(f"  [{split}] reservoir-sampled to {len(examples)}")
    return examples


def load_temporal_windows(root_dir, cities=None, exclude_cities=None,
                           max_samples=None, seed=42, split="train"):
    """Load method3-format window files (multi-frame + qa_pairs per window)
    into flat {"frames": [...], "prompt", "answer"} examples."""
    rng = random.Random(seed)
    examples = []
    n_windows = 0
    files = sorted(glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True))
    seen_cities = set()

    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        city = d.get("city")
        if cities is not None and city not in cities:
            continue
        if exclude_cities is not None and city in exclude_cities:
            continue

        frames = d.get("frames", [])
        qa_pairs = d.get("qa_pairs")
        if not frames or not qa_pairs:
            continue
        resolved_frames = [_resolve_image_path(fr) for fr in frames]
        frames = [fr for fr in resolved_frames if fr is not None]
        if not frames:
            continue

        n_windows += 1
        seen_cities.add(city)
        for qa in qa_pairs:
            q, a = qa.get("question"), qa.get("answer")
            if not q or not a:
                continue
            examples.append({"frames": frames, "prompt": q, "answer": a})

    print(f"  [{split}] temporal {root_dir}: cities={sorted(seen_cities)} -> "
          f"{n_windows} windows, {len(examples)} QA examples")

    if max_samples and len(examples) > max_samples:
        rng.shuffle(examples)
        examples = examples[:max_samples]
        print(f"  [{split}] capped to {max_samples}")
    return examples


def load_temporal_manifest(manifest_path, cities=None, exclude_cities=None,
                           max_samples=None, seed=42, split="train", front_only=False):
    """Load a vqa_paper clean_*.json manifest -- {"windows": [{city, frames,
    qa_pairs}]} -- into the same flat example shape as load_temporal_windows.

    Prefer this over --temporal_data_dir: the manifests are the audit-clean
    subset (an audit found 13% of the raw method3 windows had 4 "consecutive"
    frames spanning two rosbags in reverse chronological order, so their QA
    text describes motion that never happened). Pointing at the raw templates
    directory silently trains on those again.
    """
    rng = random.Random(seed)
    with open(manifest_path) as f:
        man = json.load(f)

    examples, n_windows, seen_cities = [], 0, set()
    for w in man.get("windows", []):
        city = w.get("city")
        if cities is not None and city not in cities:
            continue
        if exclude_cities is not None and city in exclude_cities:
            continue
        _raw = w.get("frames", [])
        if front_only:
            _raw = _raw[:FRONT_ONLY_N_FRAMES]
        frames = [fr for fr in (_resolve_image_path(p) for p in _raw)
                  if fr is not None]
        qa_pairs = w.get("qa_pairs")
        if not frames or not qa_pairs:
            continue
        n_windows += 1
        seen_cities.add(city)
        for qa in qa_pairs:
            q, a = qa.get("question"), qa.get("answer")
            if q and a:
                examples.append({"frames": frames, "prompt": q, "answer": a})

    print(f"  [{split}] temporal-manifest {os.path.basename(manifest_path)}: "
          f"cities={sorted(seen_cities)} -> {n_windows} windows, "
          f"{len(examples)} QA examples")
    if max_samples and len(examples) > max_samples:
        rng.shuffle(examples)
        examples = examples[:max_samples]
        print(f"  [{split}] capped to {max_samples}")
    return examples


def _bags_in_split(split_dir):
    """(city, bag) pairs present in an indian_windows split."""
    out = set()
    for fp in glob.glob(os.path.join(split_dir, "window_*.json")):
        with open(fp) as f:
            d = json.load(f)
        out.add((d.get("city"), d.get("bag")))
    return out


def load_traj_replay(val_or_train_dir, max_samples=None, seed=42, split="train",
                     exclude_bags_from=None):
    """Trajectory-prediction replay, in the exact prompt/answer format the
    trajectory checkpoint was trained in (traj_only response mode, decimal
    waypoints, 12 images ordered forward/front_left/front_right).

    This stream exists because of a measured failure: traj_vqa_frozen_v1 was
    fine-tuned for 8,000 steps on static VQA whose targets contain no
    <traj_start>/<traj_end>, and the trajectory format degraded from 100% to
    24% strictly-parseable output (the geometry survived -- ADE moved only
    4-9% -- but the model stopped reliably emitting the tokens adjacent to the
    special-token boundaries). Keeping a slice of trajectory targets in the
    loss is what stops that recurring in later stages.

    Examples carry letterbox=True: the trajectory model reads 512x288
    aspect-preserving letterboxed frames (144 visual tokens/image), not the
    native-resolution frames the static/temporal streams use.
    """
    import sys as _sys
    _sys.path.insert(0, "/home/ai24mtech02003/vqa/coc")
    from traj_without_tres.data.message_builder import build_user_prompt
    from traj_without_tres.utils.trajectory_utils import format_future_trajectory
    from traj_without_tres.utils.response_utils import wrap_section

    rng = random.Random(seed)
    # titan_v2's train/val split follows /data/common_datasets/splits.yaml
    # (its val split contains exactly the four cities with val tracks there:
    # Agra, Faridabad, Gwalior, Indore_Aftnoon). The split is track-level, so a
    # handful of bags straddle the boundary and appear on both sides -- 44 of
    # 54,326 train windows (0.08%). Small, but it is still val data, so drop it.
    banned = _bags_in_split(exclude_bags_from) if exclude_bags_from else set()

    files = sorted(glob.glob(os.path.join(val_or_train_dir, "window_*.json")))
    if max_samples and len(files) > max_samples:
        # stride first, then trim: window_*.json is sequential within a bag, so
        # the first N files are near-consecutive frames from one drive.
        stride = max(1, len(files) // max_samples)
        files = files[::stride][:max_samples]

    examples, skipped, leaked = [], 0, 0
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        if banned and (d.get("city"), d.get("bag")) in banned:
            leaked += 1
            continue
        cams = d.get("frames_by_camera", {})
        try:
            frames = []
            for cam_key in ("forward", "front_left", "front_right"):
                for p in cams[cam_key]:
                    r = _resolve_image_path(p)
                    if r is None:
                        raise FileNotFoundError(p)
                    frames.append(r)
        except (KeyError, FileNotFoundError):
            skipped += 1
            continue
        if len(d.get("past_traj", [])) != 4 or len(d.get("future_traj", [])) != 12:
            skipped += 1
            continue
        sample = {
            "navigation_command": (d.get("navigation_command") or "GO STRAIGHT").upper(),
            "past_xyz": d["past_traj"],
            "future_xyz": d["future_traj"],
        }
        examples.append({
            "frames": frames,
            "prompt": build_user_prompt(sample, response_mode="traj_only",
                                        use_delta_tokens=False, inference=False),
            "answer": wrap_section("traj", format_future_trajectory(d["future_traj"])),
            "letterbox": True,
        })

    print(f"  [{split}] traj-replay {val_or_train_dir}: {len(examples)} windows "
          f"({skipped} skipped: missing images/bad traj shape"
          + (f"; {leaked} dropped: bag also in the val split)" if banned else ")"))
    return examples


# Every temporal window and every joint pair orders its 12 frames as
# camera3 x4, camera1 x4, camera4 x4 -- verified over all 2,928 manifest windows
# and all 11,810 joint records, 0 exceptions. camera3 is the forward camera, so
# frames[:4] is exactly "front camera, 4 consecutive frames (past -> present)".
# 43.6% of the joint QA names a side view outright ("the left camera view
# provides the earliest confirmation", "visible in the forward and left views").
# Those targets are unanswerable from camera3 alone, so a front-only run trained
# on them is scored partly on guessing -- which inflates the apparent value of the
# side cameras. --drop_side_ref_qa removes them so the ablation can also be read
# as "how good is a genuinely front-only model", not just "what breaks".
SIDE_REF_RE = re.compile(
    r"front[- ]?left|front[- ]?right|left[- ]?(camera|view)|right[- ]?(camera|view)"
    r"|side (camera|view)|peripheral", re.I)

FRONT_ONLY_N_FRAMES = 4
FRONT_ONLY_IMAGE_LINE = ("- 4 images from the front camera, "
                         "4 consecutive frames (past \u2192 present)")


def load_temporal_traj_joint(json_path, max_samples=None, seed=42,
                             split="train_joint", qa_cap=None, front_only=False,
                             drop_side_ref_qa=False):
    """Joint reasoning + trajectory stream: one target carries BOTH the
    temporal-reasoning answer and the 6 s future trajectory for the same
    window, so the waypoints are generated conditioned on the model's own
    reasoning rather than in a separate forward pass.

    Built by coc/build_temporal_traj_pairs.py, which joins each temporal
    window to the trajectory window nearest in time (|dt| <= 0.25 s) on
    (city, bag, timestamp). Frames come from the TEMPORAL window, because the
    reasoning answers reference them positionally ("in frame 1", "frames 5-8");
    substituting the trajectory window's frames would silently break every
    such reference.

    Geometry deliberately matches the temporal stream (native pixel budget,
    no letterbox), not the trajectory replay stream: the only intended
    difference from the plain temporal stage is the appended <traj_...>
    section, so that a LingoQA delta is attributable to the joint target and
    not to a change of image resolution.
    """
    import sys as _sys
    _sys.path.insert(0, "/home/ai24mtech02003/vqa/coc")
    from traj_without_tres.data import message_builder as _mb
    from traj_without_tres.utils.response_utils import wrap_section
    from traj_without_tres.utils.trajectory_utils import (
        format_past_trajectory, format_future_trajectory)

    with open(json_path) as f:
        records = json.load(f)

    n_side = 0
    if drop_side_ref_qa:
        kept = [r for r in records
                if not (SIDE_REF_RE.search(r["question"]) or SIDE_REF_RE.search(r["answer"]))]
        n_side = len(records) - len(kept)
        records = kept

    image_line = FRONT_ONLY_IMAGE_LINE if front_only else _mb._IMAGE_LINE
    fmt_block = (
        "Respond in the following structured format:\n\n"
        + wrap_section("coc", "Answer the question above, reasoning over the frames.") + "\n"
        + wrap_section("traj", _mb._TRAJ_DESC_TEXT) + "\n\n"
        + "Return both sections exactly once."
    )

    rng = random.Random(seed)
    if qa_cap:
        by_window = {}
        for r in records:
            by_window.setdefault((r["city"], r["bag"], r["window_idx"]), []).append(r)
        records = []
        for k in sorted(by_window):
            grp = by_window[k]
            rng.shuffle(grp)
            records += grp[:qa_cap]

    if max_samples and len(records) > max_samples:
        stride = max(1, len(records) // max_samples)
        records = records[::stride][:max_samples]

    examples, skipped = [], 0
    for r in records:
        raw = r.get("image_paths", [])
        if front_only:
            raw = raw[:FRONT_ONLY_N_FRAMES]
        frames = [_resolve_image_path(fr) for fr in raw]
        frames = [fr for fr in frames if fr is not None]
        if len(frames) != len(raw) or not frames:
            skipped += 1
            continue
        if len(r.get("past_xyz", [])) != 4 or len(r.get("future_xyz", [])) != 12:
            skipped += 1
            continue
        prompt = "\n".join([
            _mb._SYSTEM_PREAMBLE,
            "",
            "Input:",
            image_line,
            _mb._NAV_LINE_TEMPLATE.format(
                nav_command=(r.get("navigation_command") or "GO STRAIGHT").upper()),
            _mb._PAST_LINE_TEMPLATE.format(
                past_traj=format_past_trajectory(r["past_xyz"])),
            "",
            "Question: " + r["question"].strip(),
            "",
            fmt_block,
        ])
        answer = "\n".join([
            wrap_section("coc", r["answer"].strip()),
            wrap_section("traj", format_future_trajectory(r["future_xyz"])),
        ])
        examples.append({"frames": frames, "prompt": prompt, "answer": answer})

    n_win = len({(r["city"], r["bag"], r["window_idx"]) for r in records})
    print(f"  [{split}] joint reasoning+traj {json_path}: {len(examples)} QA "
          f"over {n_win} windows ({skipped} skipped: missing images/bad traj shape"
          + (f"; {n_side} dropped: QA references a side camera)" if drop_side_ref_qa else ")")
          + (" [FRONT CAMERA ONLY]" if front_only else ""))
    return examples


def letterbox_resize(img, target_w=512, target_h=288):
    """Aspect-preserving fit into (target_w, target_h) + black centre-pad.
    A naive resize distorts geometry and shifts the model's metric-distance
    regression, which is the one thing the trajectory stream is protecting."""
    ow, oh = img.size
    scale = min(target_w / ow, target_h / oh)
    nw, nh = max(1, round(ow * scale)), max(1, round(oh * scale))
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(img.resize((nw, nh), Image.LANCZOS),
                 ((target_w - nw) // 2, (target_h - nh) // 2))
    return canvas


class VQADataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _open_image_with_retry(path, retries=4, delay=2.0):
    """A transient NFS hiccup mid-training previously crashed a multi-day run
    (FileNotFoundError on a file that exists and is fine moments later).
    Retry with backoff before giving up on this specific image.

    Fixed after a live-run diagnosis: a bulk external png->jpg rename wave
    was making many files genuinely, permanently missing at their recorded
    path (not transient), and retrying those with escalating delay (up to
    12s/image, stacked across every missing frame in an example) tanked
    throughput 10-25x with no chance of success. Now: try the same
    extension-fallback resolution used at dataset-load time first (cheap,
    often recovers the file immediately instead of just dropping it faster);
    only genuine other OSErrors (not "not found") get the retry-with-delay
    treatment, since those can plausibly be transient.
    """
    resolved = _resolve_image_path(path)
    if resolved is not None:
        path = resolved
    last_exc = None
    for attempt in range(retries):
        try:
            return Image.open(path).convert("RGB")
        except FileNotFoundError as e:
            last_exc = e
            break  # not transient even after extension fallback -- don't wait on it
        except OSError as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"  [Collator] WARNING: dropping example, image unreadable after "
          f"{retries} attempts: {path} ({last_exc})")
    return None


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pad_id = self.processor.tokenizer.pad_token_id
        encoded = []

        for ex in batch:
            images = [_open_image_with_retry(p) for p in ex["frames"]]
            if any(img is None for img in images):
                continue  # unreadable after retries -- drop this example, don't crash the batch
            if ex.get("letterbox"):
                images = [letterbox_resize(img) for img in images]
            # 24_dataset_coc mix records: crop -> hflip -> resize (records are 16:9 after crop)
            if ex.get("crop"):
                images = [img.crop(tuple(ex["crop"])) for img in images]
            if ex.get("hflip"):
                images = [ImageOps.mirror(img) for img in images]
            if ex.get("resize"):
                images = [img.resize(tuple(ex["resize"]), Image.BICUBIC) for img in images]
            user_content = [{"type": "image", "image": img} for img in images]
            user_content.append({"type": "text", "text": ex["prompt"]})

            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
            ]

            full = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            prompt_only = self.processor.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True)

            full_inputs = self.processor(
                text=[full], images=images, return_tensors="pt", padding=False)
            prompt_ids = self.processor(
                text=[prompt_only], images=images,
                return_tensors="pt", padding=False)["input_ids"]
            n_prompt = prompt_ids.shape[1]

            input_ids = full_inputs["input_ids"][0]
            labels = input_ids.clone()
            labels[:n_prompt] = -100

            encoded.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": full_inputs["attention_mask"][0],
                "pixel_values": full_inputs.get("pixel_values"),
                "image_grid_thw": full_inputs.get("image_grid_thw"),
                "mm_token_type_ids": (
                    full_inputs["mm_token_type_ids"][0]
                    if "mm_token_type_ids" in full_inputs else None),
            })

        maxlen = max(e["input_ids"].shape[0] for e in encoded)

        def pad_to(t, length, value):
            if t.shape[0] >= length:
                return t[:length]
            return torch.cat([t, torch.full((length - t.shape[0],), value, dtype=t.dtype)])

        out = {
            "input_ids": torch.stack([pad_to(e["input_ids"], maxlen, pad_id) for e in encoded]),
            "labels": torch.stack([pad_to(e["labels"], maxlen, -100) for e in encoded]),
            "attention_mask": torch.stack([pad_to(e["attention_mask"], maxlen, 0) for e in encoded]),
        }
        if encoded[0]["mm_token_type_ids"] is not None:
            out["mm_token_type_ids"] = torch.stack(
                [pad_to(e["mm_token_type_ids"], maxlen, 0) for e in encoded])
        if encoded[0]["pixel_values"] is not None:
            out["pixel_values"] = torch.cat([e["pixel_values"] for e in encoded], dim=0)
            out["image_grid_thw"] = torch.cat([e["image_grid_thw"] for e in encoded], dim=0)
        return out


class MultiEvalTrainer(Trainer):
    """Trainer with:
    - Periodic eval (every eval_steps): on capped subsets
    - Epoch-end eval: on full val sets
    - Per-dataset loss breakdown both times
    - Best model tracking on periodic eval_loss
    """

    def __init__(self, *args,
                 quick_eval_datasets=None, quick_combined=None,
                 full_eval_datasets=None, full_combined=None,
                 best_model_dir=None, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quick_eval_datasets = quick_eval_datasets or {}
        self.quick_combined = quick_combined
        self.full_eval_datasets = full_eval_datasets or {}
        self.full_combined = full_combined
        self.best_model_dir = best_model_dir
        self.best_eval_loss = float("inf")
        self._processor = processor

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if not self.quick_eval_datasets:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        all_metrics = super().evaluate(
            eval_dataset=self.quick_combined,
            ignore_keys=ignore_keys,
            metric_key_prefix="eval",
        )
        print(f"  >> eval_overall: loss={all_metrics.get('eval_loss', '?'):.4f}  "
              f"({len(self.quick_combined)} samples)")

        for name, ds in self.quick_eval_datasets.items():
            sub = super().evaluate(
                eval_dataset=ds,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"eval_{name}",
            )
            all_metrics.update(sub)
            loss_key = f"eval_{name}_loss"
            if loss_key in sub:
                print(f"  >> eval_{name}: loss={sub[loss_key]:.4f}  ({len(ds)} samples)")

        eval_loss = all_metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_eval_loss and self.best_model_dir:
            self.best_eval_loss = eval_loss
            self.save_model(self.best_model_dir)
            if self._processor:
                self._processor.save_pretrained(self.best_model_dir)
            info = {"step": self.state.global_step,
                    "epoch": self.state.epoch,
                    "eval_loss": eval_loss}
            with open(os.path.join(self.best_model_dir, "best_info.json"), "w") as f:
                json.dump(info, f, indent=2)
            print(f"  >> New best model saved (eval_loss={eval_loss:.4f}, step={self.state.global_step})")

        self.log(all_metrics)
        return all_metrics

    def full_evaluate(self):
        if not self.full_eval_datasets:
            return {}

        print(f"\n{'='*40} FULL EPOCH EVAL {'='*40}")

        all_metrics = super().evaluate(
            eval_dataset=self.full_combined,
            metric_key_prefix="full_eval",
        )
        print(f"  >> full_eval_overall: loss={all_metrics.get('full_eval_loss', '?'):.4f}  "
              f"({len(self.full_combined)} samples)")

        for name, ds in self.full_eval_datasets.items():
            sub = super().evaluate(
                eval_dataset=ds,
                metric_key_prefix=f"full_eval_{name}",
            )
            all_metrics.update(sub)
            loss_key = f"full_eval_{name}_loss"
            if loss_key in sub:
                print(f"  >> full_eval_{name}: loss={sub[loss_key]:.4f}  ({len(ds)} samples)")

        print(f"{'='*97}\n")
        self.log(all_metrics)
        return all_metrics


class FileLogCallback(TrainerCallback):
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, "training_log.jsonl")
        self.txt_path = os.path.join(log_dir, "training_log.txt")
        header = f"# Training log started {datetime.now().isoformat()}\n"
        with open(self.jsonl_path, "w") as f:
            f.write(header)
        with open(self.txt_path, "w") as f:
            f.write(header)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        parts = [f"step={state.global_step}"]
        parts.append(f"epoch={state.epoch:.3f}" if state.epoch is not None else "epoch=-")
        for k, v in logs.items():
            parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
        with open(self.txt_path, "a") as f:
            f.write(" | ".join(parts) + "\n")


def load_coc_jsonl(path, split, max_samples=None, cities=None, exclude_cities=None):
    """Load CoC distillation examples written by coc/build_coc_examples.py.

    The file is already in this script's example format ({frames, prompt,
    answer}), so this only filters and caps. City filtering is by the `city`
    field rather than by path, because one JSONL holds every city.
    """
    examples, skipped = [], 0
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            city = rec.get("city", "")
            if cities and city not in cities:
                continue
            if exclude_cities and city in exclude_cities:
                continue
            if not rec.get("frames") or not rec.get("answer"):
                skipped += 1
                continue
            examples.append({"frames": rec["frames"], "prompt": rec["prompt"],
                             "answer": rec["answer"]})
    if max_samples and len(examples) > max_samples:
        random.Random(42).shuffle(examples)
        examples = examples[:max_samples]
    print(f"  [{split}] coc {path}: {len(examples)} windows"
          + (f" ({skipped} skipped: empty frames/answer)" if skipped else ""))
    return examples


def _mix_groups(path):
    """'open' or 'ground' per line of a mix file (source prefix 'ground_' marks grounding_v1)."""
    with open(path) as fh:
        return ["ground" if json.loads(l).get("source", "").startswith("ground_") else "open" for l in fh]


def load_mix_jsonl(path, split, sources=None, max_samples=None):
    """Load 24_dataset_coc mix records (open-VQA + grounding_v1), already in this
    script's example format plus image ops: {frames: [img], prompt, answer, crop,
    hflip, resize, source}. The collator applies crop -> hflip -> resize."""
    examples = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if sources and rec.get("source") not in sources:
                continue
            examples.append({"frames": rec["frames"], "prompt": rec["prompt"], "answer": rec["answer"],
                             "crop": rec.get("crop"), "hflip": bool(rec.get("hflip")),
                             "resize": rec.get("resize")})
    if max_samples and len(examples) > max_samples:
        random.Random(42).shuffle(examples)
        examples = examples[:max_samples]
    print(f"  [{split}] mix {path}" + (f" {sources}" if sources else "") + f": {len(examples)} examples")
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_data_dir", default=None,
                    help="Dir of {city}_vqa.json files (stage2_qa format).")
    ap.add_argument("--static_heldout_cities", nargs="+", default=["Delhi", "kota_out"],
                    help="Excluded from static training, used as static val sets.")
    ap.add_argument("--static_replay_cap", type=int, default=None,
                    help="Sample cap on static examples (use in stage 2 to replay "
                         "a slice of stage-1 data alongside new temporal data, "
                         "instead of the full ~530K, so the new task isn't drowned out).")
    ap.add_argument("--temporal_data_dir", default=None,
                    help="method3-format root dir (city/stride*/templates/window_*.json).")
    ap.add_argument("--temporal_heldout_cities", nargs="+", default=["Dasuya"],
                    help="Excluded from temporal training, used as temporal val set.")
    ap.add_argument("--temporal_manifest", default=None,
                    help="vqa_paper clean_*.json manifest (audit-clean windows, "
                         "embedded QA + resolved frame paths). Preferred over "
                         "--temporal_data_dir, which re-includes the 13%% of raw "
                         "windows whose frames span two rosbags out of order.")
    ap.add_argument("--temporal_cap", type=int, default=None,
                    help="Cap on temporal QA examples (after city filtering).")
    ap.add_argument("--add_traj_tokens", action="store_true",
                    help="Register <coc_start>/<coc_end>/<traj_start>/<traj_end> "
                         "as atomic special tokens before training. Without this "
                         "each tag is FOUR ordinary sub-word pieces ('<tr','aj',"
                         "'_start','>'), which is why traj_vqa_frozen_v1 decayed "
                         "from 100%% to 24%% strictly-parseable output. Use when "
                         "teaching the trajectory format to a checkpoint that has "
                         "never emitted it; do NOT use on a checkpoint that has "
                         "already learnt the 4-piece form (branch A).")
    ap.add_argument("--traj_replay_dir", default=None,
                    help="indian_windows split dir for trajectory replay "
                         "(traj_only targets, 512x288 letterboxed frames). "
                         "Keeps <traj_start>/<traj_end> in the loss so the "
                         "trajectory format does not decay again.")
    ap.add_argument("--traj_replay_cap", type=int, default=None,
                    help="Number of trajectory windows to replay (strided "
                         "across the split, not the first N).")
    ap.add_argument("--traj_val_dir", default=None,
                    help="indian_windows val split. Any (city, bag) appearing "
                         "here is dropped from --traj_replay_dir: the split is "
                         "track-level per splits.yaml, so a few bags straddle "
                         "the boundary and land in both directories.")
    ap.add_argument("--temporal_val_manifest", default=None,
                    help="Separate manifest for temporal validation (e.g. the "
                         "Delhi val manifest). Use this instead of carving a "
                         "val city out of the training manifest when that city "
                         "was trained on in an earlier stage -- holding out "
                         "Dasuya's temporal windows proves nothing when Dasuya "
                         "images were already seen during static VQA training.")
    ap.add_argument("--drop_side_ref_qa", action="store_true",
                    help="Drop joint QA whose question or answer names a side view "
                         "(43.6%% of the set). Pair with --front_camera_only for the "
                         "fair front-only model; omit it for the pure input ablation.")
    ap.add_argument("--front_camera_only", action="store_true",
                    help="Camera ablation: feed only camera3 (the forward camera), "
                         "4 consecutive frames, instead of all 12 (camera3/1/4 x4). "
                         "Applies to the joint and temporal-manifest streams -- the "
                         "static stream is single-image and the traj-replay stream is "
                         "left at 12 on purpose (see --traj_replay_dir).")
    ap.add_argument("--joint_traj_json", default=None,
                    help="temporal_traj_pairs/train.json -- joint reasoning+trajectory "
                         "stream (one target holds both the temporal answer and the "
                         "6s waypoints for the same window). Implies --add_traj_tokens.")
    ap.add_argument("--joint_traj_val_json", default=None,
                    help="Held-out split for the joint stream (heldout_trajval.json).")
    ap.add_argument("--joint_traj_cap", type=int, default=None,
                    help="Cap total joint QA examples.")
    ap.add_argument("--joint_qa_cap", type=int, default=None,
                    help="Cap QA per window before sampling. A window has ~14 QA but "
                         "exactly one trajectory, so without a cap the same 12 waypoints "
                         "are the target 14 times and the traj head overfits that window.")
    ap.add_argument("--synthetic_data_dir", default=None,
                    help="Synthetic method3-format windows (CARLA-style crash / "
                         "precrash / normal clips). Kept as its OWN stream and "
                         "its OWN eval split so its effect on real-data "
                         "performance is visible instead of averaged away.")
    ap.add_argument("--synthetic_val_dir", default=None,
                    help="Held-out synthetic windows -> reported as the "
                         "'synthetic' eval split.")
    ap.add_argument("--synthetic_cap", type=int, default=None,
                    help="Cap on synthetic QA examples used for training.")
    ap.add_argument("--coc_train_jsonl", default=None,
                    help="JSONL from coc/build_coc_examples.py (mass CoC distillation)")
    ap.add_argument("--coc_val_jsonl", default=None,
                    help="held-out-city JSONL from the same builder")
    ap.add_argument("--coc_heldout_cities", nargs="+", default=["delhi", "kota_out"],
                    help="never trained on; one eval split is built per city")
    ap.add_argument("--coc_cap", type=int, default=None)
    ap.add_argument("--mix_train_jsonl", default=None,
                    help="24_dataset_coc open-VQA + grounding mix (records with crop/hflip/resize).")
    ap.add_argument("--mix_val_jsonl", default=None)
    ap.add_argument("--lora", action="store_true",
                    help="Train a LoRA adapter on a frozen base instead of full fine-tuning (peft).")
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", default="llm", choices=["llm", "llm+merger"],
                    help="llm: q/k/v/o/gate/up/down of the 28 decoder layers; llm+merger also adapts the vision merger.")
    ap.add_argument("--save_epochs", action="store_true",
                    help="checkpoint once per epoch instead of every --save_steps")
    ap.add_argument("--save_only_model", action="store_true",
                    help="omit optimizer/scheduler state from checkpoints. A 2B full "
                         "fine-tune writes ~20GB per checkpoint with AdamW state "
                         "(4GB weights + 16GB fp32 moments) and ~4GB without, so this "
                         "is the difference between 3 epochs costing 60GB and 12GB. "
                         "Trade-off: cannot resume mid-run from such a checkpoint.")
    ap.add_argument("--final_eval", action="store_true",
                    help="run the uncapped end-of-training eval over every val "
                         "set. Off by default: it runs after best/ is already "
                         "saved and changes no artefact, so it is pure GPU cost.")
    ap.add_argument("--max_val_samples", type=int, default=500,
                    help="Per-val-set cap for PERIODIC eval. Full eval at epoch end uses all.")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct",
                    help="HF model id, or a local dir (e.g. a stage-1 checkpoint for stage 2).")
    ap.add_argument("--models_root", default="/data/common_datasets/ai24mtech02003/final_vqa/models")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=5000)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--save_total_limit", type=int, default=2,
                    help="How many step-checkpoints to keep. The default 2 is "
                         "enough when best/ is the only artefact that matters, but "
                         "best/ is chosen by EVAL LOSS, which ranks these models "
                         "backwards against LingoQA every time it has been checked. "
                         "Raise it to keep one checkpoint per epoch alive for "
                         "LingoQA-based selection instead.")
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="Cap total steps (for smoke tests); -1 = full epochs.")
    ap.add_argument("--dataloader_num_workers", type=int, default=4,
                    help="Dataloader worker processes (image decode/collate is CPU-bound; "
                         "raise this toward --cpus-per-task if GPU utilization is low).")
    ap.add_argument("--freeze_vision", action="store_true",
                    help="Freeze the vision tower (model.visual) entirely. Use when adapting "
                         "a checkpoint whose visual representations should be preserved.")
    ap.add_argument("--freeze_layers", type=int, default=0,
                    help="Freeze the first N language_model decoder layers (of 28) plus the "
                         "token embeddings. Use with a low --lr to adapt a specialized "
                         "checkpoint (e.g. trajectory-trained) to a new task without "
                         "overwriting its early/general representations.")
    args = ap.parse_args()
    if args.joint_traj_json and not args.add_traj_tokens:
        print("--joint_traj_json emits <coc_...>/<traj_...>; enabling --add_traj_tokens.")
        args.add_traj_tokens = True

    # The guard predates --temporal_manifest and --joint_traj_json. Both are
    # complete training streams on their own; every run until now happened to
    # also pass --static_data_dir, which is the only reason the narrow check
    # never fired on a valid configuration.
    if not (args.static_data_dir or args.temporal_data_dir
            or args.temporal_manifest or args.joint_traj_json
            or args.coc_train_jsonl or args.mix_train_jsonl):
        raise ValueError("Pass at least one training stream: --static_data_dir / "
                         "--temporal_data_dir / --temporal_manifest / "
                         "--joint_traj_json / --coc_train_jsonl")

    out_dir = os.path.join(args.models_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading processor + model...")
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    tok = getattr(processor, "tokenizer", processor)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
    print(f"Attention: {attn_impl}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=None, attn_implementation=attn_impl)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if args.add_traj_tokens:
        # The embedding matrix is [151936, 2048] while the tokenizer occupies
        # only 151669 ids, so all four tags fit in rows that are already
        # allocated -- resize_token_embeddings is a no-op and no checkpoint
        # shape changes. tie_word_embeddings=True (no separate lm_head), so
        # writing an input row also sets the output row.
        new_tokens = ["<coc_start>", "<coc_end>", "<traj_start>", "<traj_end>"]
        before = {t: tok.encode(t, add_special_tokens=False) for t in new_tokens}
        n_added = tok.add_special_tokens({"additional_special_tokens": new_tokens})
        rows = model.get_input_embeddings().weight.shape[0]
        if len(tok) > rows:
            model.resize_token_embeddings(len(tok))
        else:
            # Resizing DOWN to len(tok) would drop the spare rows and rewrite
            # both tied tensors for no benefit. The ids we just handed out are
            # already inside the matrix, so leave it exactly as the checkpoint
            # had it and keep config.vocab_size unchanged.
            print(f"Embedding matrix already has {rows} rows >= vocab "
                  f"{len(tok)}; skipping resize_token_embeddings.")
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            for t, pieces in before.items():
                tid = tok.convert_tokens_to_ids(t)
                if len(pieces) > 1:
                    # seed from the sub-word pieces the tokenizer used to spell
                    # the tag, so the new row starts somewhere meaningful
                    emb[tid] = emb[pieces].mean(0)
        for t in new_tokens:
            ids = tok.encode(t, add_special_tokens=False)
            assert len(ids) == 1, f"{t} still tokenizes to {len(ids)} pieces: {ids}"
        print(f"Special tokens: registered {n_added} new "
              f"({', '.join(new_tokens)}); vocab {len(tok)}, "
              f"embedding rows {emb.shape[0]}")

    if args.freeze_vision:
        for p in model.model.visual.parameters():
            p.requires_grad = False
    if args.freeze_layers > 0:
        if args.add_traj_tokens:
            # tie_word_embeddings=True means embed_tokens IS the output head.
            # Freezing it would leave the four fresh rows at their init value
            # and the model could never learn to EMIT the new tokens.
            print("Keeping embed_tokens trainable: --add_traj_tokens needs the "
                  "new rows (and the tied output head) in the optimiser.")
        else:
            for p in model.model.language_model.embed_tokens.parameters():
                p.requires_grad = False
        for i, layer in enumerate(model.model.language_model.layers):
            if i < args.freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
    if args.lora:
        from peft import LoraConfig, get_peft_model
        tgt = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
        if args.lora_targets == "llm+merger":
            tgt = rf"({tgt})|(model\.visual\.merger\.linear_fc[12])"
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                                 lora_dropout=args.lora_dropout, target_modules=tgt,
                                                 bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    if args.freeze_vision or args.freeze_layers > 0:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Frozen: vision={args.freeze_vision} first {args.freeze_layers} decoder layers + embeddings. "
              f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.1f}%)")

    # ---------- Train ----------
    train_examples = []
    if args.static_data_dir:
        train_examples += load_static_records(
            args.static_data_dir, exclude_cities=args.static_heldout_cities,
            max_samples=args.static_replay_cap, split="train_static")
    if args.temporal_data_dir:
        train_examples += load_temporal_windows(
            args.temporal_data_dir, exclude_cities=args.temporal_heldout_cities,
            split="train_temporal")
    if args.temporal_manifest:
        train_examples += load_temporal_manifest(
            args.temporal_manifest, exclude_cities=args.temporal_heldout_cities,
            max_samples=args.temporal_cap, split="train_temporal",
            front_only=args.front_camera_only)
    if args.joint_traj_json:
        train_examples += load_temporal_traj_joint(
            args.joint_traj_json, max_samples=args.joint_traj_cap,
            split="train_joint", qa_cap=args.joint_qa_cap,
            front_only=args.front_camera_only,
            drop_side_ref_qa=args.drop_side_ref_qa)
    if args.traj_replay_dir:
        train_examples += load_traj_replay(
            args.traj_replay_dir, max_samples=args.traj_replay_cap,
            split="train_traj_replay", exclude_bags_from=args.traj_val_dir)
    if args.synthetic_data_dir:
        train_examples += load_temporal_windows(
            args.synthetic_data_dir, max_samples=args.synthetic_cap,
            split="train_synthetic")
    if args.coc_train_jsonl:
        train_examples += load_coc_jsonl(
            args.coc_train_jsonl, split="train_coc", max_samples=args.coc_cap,
            exclude_cities=args.coc_heldout_cities)
    if args.mix_train_jsonl:
        train_examples += load_mix_jsonl(args.mix_train_jsonl, split="train_mix")
    random.Random(42).shuffle(train_examples)
    if not train_examples:
        raise RuntimeError("No usable training examples. Check paths / city filters.")
    train_dataset = VQADataset(train_examples)

    # ---------- Val (always held-out cities, never trained on) ----------
    quick_eval_datasets, full_eval_datasets = {}, {}

    if args.coc_val_jsonl:
        for city in (args.coc_heldout_cities or []):
            full_ex = load_coc_jsonl(args.coc_val_jsonl, cities=[city],
                                     split=f"val_full_coc_{city}")
            if not full_ex:
                continue
            full_eval_datasets[f"coc_{city}"] = VQADataset(full_ex)
            quick_eval_datasets[f"coc_{city}"] = VQADataset(
                full_ex[: args.max_val_samples])

    if args.static_data_dir and args.static_heldout_cities:
        for city in args.static_heldout_cities:
            full_ex = load_static_records(
                args.static_data_dir, cities=[city], split=f"val_full_static_{city}")
            if not full_ex:
                continue
            full_eval_datasets[f"static_{city}"] = VQADataset(full_ex)
            quick_ex = load_static_records(
                args.static_data_dir, cities=[city],
                max_samples=args.max_val_samples, split=f"val_quick_static_{city}")
            quick_eval_datasets[f"static_{city}"] = VQADataset(quick_ex)

    if args.temporal_val_manifest:
        full_ex = load_temporal_manifest(
            args.temporal_val_manifest, split="val_full_temporal",
            front_only=args.front_camera_only)
        if full_ex:
            full_eval_datasets["temporal"] = VQADataset(full_ex)
            quick_eval_datasets["temporal"] = VQADataset(
                load_temporal_manifest(
                    args.temporal_val_manifest, max_samples=args.max_val_samples,
                    split="val_quick_temporal",
                    front_only=args.front_camera_only))
    elif args.temporal_manifest and args.temporal_heldout_cities:
        for city in args.temporal_heldout_cities:
            full_ex = load_temporal_manifest(
                args.temporal_manifest, cities=[city],
                split=f"val_full_temporal_{city}",
                front_only=args.front_camera_only)
            if not full_ex:
                continue
            full_eval_datasets[f"temporal_{city}"] = VQADataset(full_ex)
            quick_eval_datasets[f"temporal_{city}"] = VQADataset(
                load_temporal_manifest(
                    args.temporal_manifest, cities=[city],
                    max_samples=args.max_val_samples,
                    split=f"val_quick_temporal_{city}",
                    front_only=args.front_camera_only))

    if args.temporal_data_dir and args.temporal_heldout_cities:
        for city in args.temporal_heldout_cities:
            full_ex = load_temporal_windows(
                args.temporal_data_dir, cities=[city], split=f"val_full_temporal_{city}")
            if not full_ex:
                continue
            full_eval_datasets[f"temporal_{city}"] = VQADataset(full_ex)
            quick_ex = load_temporal_windows(
                args.temporal_data_dir, cities=[city],
                max_samples=args.max_val_samples, split=f"val_quick_temporal_{city}")
            quick_eval_datasets[f"temporal_{city}"] = VQADataset(quick_ex)

    if args.synthetic_val_dir:
        full_ex = load_temporal_windows(
            args.synthetic_val_dir, split="val_full_synthetic")
        if full_ex:
            full_eval_datasets["synthetic"] = VQADataset(full_ex)
            quick_eval_datasets["synthetic"] = VQADataset(
                load_temporal_windows(
                    args.synthetic_val_dir, max_samples=args.max_val_samples,
                    split="val_quick_synthetic"))

    if args.joint_traj_val_json:
        full_ex = load_temporal_traj_joint(args.joint_traj_val_json, split="val_full_joint",
                                           front_only=args.front_camera_only,
                                           drop_side_ref_qa=args.drop_side_ref_qa)
        if full_ex:
            full_eval_datasets["joint"] = VQADataset(full_ex)
            quick_eval_datasets["joint"] = VQADataset(
                load_temporal_traj_joint(args.joint_traj_val_json,
                                         max_samples=args.max_val_samples,
                                         split="val_quick_joint",
                                         front_only=args.front_camera_only,
                                         drop_side_ref_qa=args.drop_side_ref_qa))

    if args.traj_val_dir:
        full_ex = load_traj_replay(args.traj_val_dir, split="val_full_traj")
        if full_ex:
            full_eval_datasets["traj"] = VQADataset(full_ex)
            quick_eval_datasets["traj"] = VQADataset(
                load_traj_replay(args.traj_val_dir,
                                 max_samples=args.max_val_samples,
                                 split="val_quick_traj"))

    if args.mix_val_jsonl:
        mix_val, groups = load_mix_jsonl(args.mix_val_jsonl, split="val_full_mix"), _mix_groups(args.mix_val_jsonl)
        for grp in ("open", "ground"):
            full_ex = [e for e, r in zip(mix_val, groups) if r == grp]
            if not full_ex:
                continue
            full_eval_datasets[f"mix_{grp}"] = VQADataset(full_ex)
            quick_eval_datasets[f"mix_{grp}"] = VQADataset(full_ex[: args.max_val_samples])
    quick_combined = (VQADataset([e for ds in quick_eval_datasets.values() for e in ds.examples])
                       if quick_eval_datasets else None)
    full_combined = (VQADataset([e for ds in full_eval_datasets.values() for e in ds.examples])
                      if full_eval_datasets else None)

    collator = Collator(processor)
    has_val = bool(quick_eval_datasets)
    best_dir = os.path.join(out_dir, "best")

    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch" if args.save_epochs else "steps",
        save_steps=args.save_steps,
        save_total_limit=None if args.save_epochs else args.save_total_limit,
        save_only_model=args.save_only_model,
        eval_strategy="steps" if has_val else "no",
        eval_steps=args.eval_steps if has_val else None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        # MUST stay False. transformers caches the eval dataloader under the
        # single key "eval" for any Dataset object (trainer.py get_eval_dataloader),
        # so with persistent workers every per-split and full eval silently reused
        # the FIRST dataset built (quick_combined). That made every per-split loss
        # identical to the combined loss in stage1_static_vqa_v2_bs32,
        # traj_vqa_frozen_v1 and temporal_replay_v1.
        dataloader_persistent_workers=False,
        run_name=args.run_name,
    )

    trainer = MultiEvalTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=quick_combined,
        data_collator=collator,
        quick_eval_datasets=quick_eval_datasets,
        quick_combined=quick_combined,
        full_eval_datasets=full_eval_datasets,
        full_combined=full_combined,
        best_model_dir=best_dir,
        processor=processor,
    )
    trainer.add_callback(FileLogCallback(out_dir))

    print(f"\n{'='*60}")
    print(f"Run:            {args.run_name}")
    print(f"Base model:     {args.model}")
    print(f"Train:          {len(train_dataset)} examples")
    if quick_eval_datasets:
        print(f"--- Periodic eval (every {args.eval_steps} steps, capped {args.max_val_samples}/set) ---")
        for n, ds in quick_eval_datasets.items():
            print(f"  Val [{n}]:     {len(ds)} examples")
        if args.final_eval:
            print(f"--- End-of-training full eval ---")
            for n, ds in full_eval_datasets.items():
                print(f"  Val [{n}]:     {len(ds)} examples")
    print(f"Effective BS:   {args.bs} x {args.grad_accum} = {args.bs * args.grad_accum}")
    print(f"Save every:     {args.save_steps} steps (keep last {targs.save_total_limit})")
    print(f"Best model:     {best_dir}/")
    print(f"Logs:           {out_dir}/training_log.txt")
    print(f"{'='*60}\n")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Off by default: this re-evaluates every val set uncapped AFTER best/ is
    # already written, so on a large val set it burns hours of GPU without
    # changing a single artefact. Pass --final_eval when the uncapped numbers
    # are actually wanted.
    if args.final_eval:
        trainer.full_evaluate()

    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    print(f"Done.")
    print(f"  Last weights  -> {out_dir}")
    print(f"  Best weights  -> {best_dir}")
    if os.path.exists(os.path.join(best_dir, "best_info.json")):
        with open(os.path.join(best_dir, "best_info.json")) as f:
            info = json.load(f)
        print(f"  Best eval_loss: {info['eval_loss']:.4f} at step {info['step']}")


if __name__ == "__main__":
    main()
