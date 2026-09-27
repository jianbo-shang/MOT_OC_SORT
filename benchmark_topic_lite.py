"""Measure uncached video throughput; optionally include annotated MP4 encoding."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2

from mot_core import FrameAnalyzer, ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "weights" / "轻量级模型.pt")
    parser.add_argument("--reid-model", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "0", "auto"))
    parser.add_argument("--detection-interval", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-video", type=Path,
                        help="Save annotated MP4 and include video encoding in the timed run")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "topic_lite_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file() or not args.model.is_file():
        raise FileNotFoundError("Input video and detector model are required")
    if args.reid_model is not None and not args.reid_model.is_file():
        raise FileNotFoundError(f"ReID model not found: {args.reid_model}")
    video_path = args.save_video.resolve() if args.save_video else None
    if video_path is not None:
        if video_path.suffix.lower() != ".mp4":
            raise ValueError("--save-video requires an .mp4 output path")
        if video_path in {args.input.resolve(), args.output.resolve()}:
            raise ValueError("Saved video must differ from input and JSON report")
        if video_path.exists():
            raise FileExistsError(f"Refusing to overwrite saved video: {video_path}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer = FrameAnalyzer(
        args.model, device=args.device, detection_interval=args.detection_interval,
        reid_model_path=args.reid_model,
    )
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input: {args.input}")
    ok, warmup = capture.read()
    if not ok:
        raise RuntimeError("Input contains no readable frame")
    analyzer.process(warmup)
    analyzer.reset()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = 0
    detector_frames = 0
    detector_ms = tracker_ms = reid_ms = 0.0
    video_write_ms = 0.0
    writer = None
    started = time.perf_counter()
    try:
        if video_path is not None:
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = 25.0
            height, width = warmup.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {video_path}")
        while args.max_frames <= 0 or frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            result = analyzer.process(frame)
            if writer is not None:
                write_started = time.perf_counter()
                writer.write(result.frame)
                video_write_ms += (time.perf_counter() - write_started) * 1000.0
            frames += 1
            detector_frames += int(result.detector_ran)
            detector_ms += result.detector_ms
            tracker_ms += result.tracker_ms
            reid_ms += result.reid_ms
    finally:
        if writer is not None:
            writer.release()
        capture.release()
    elapsed = time.perf_counter() - started
    if video_path is not None and (not video_path.is_file() or video_path.stat().st_size == 0):
        raise RuntimeError(f"Output video is empty: {video_path}")
    report = {
        "input": str(args.input.resolve()),
        "model": str(args.model.resolve()),
        "reid_model": str(args.reid_model.resolve()) if args.reid_model else None,
        "device": args.device,
        "detection_interval": args.detection_interval,
        "frames": frames,
        "detector_frames": detector_frames,
        "elapsed_seconds": elapsed,
        "end_to_end_fps": frames / max(elapsed, 1e-9),
        "mean_detector_ms_per_detector_frame": detector_ms / max(detector_frames, 1),
        "mean_tracker_ms_per_video_frame": tracker_ms / max(frames, 1),
        "mean_reid_ms_per_video_frame": reid_ms / max(frames, 1),
        "mean_video_write_ms_per_frame": video_write_ms / max(frames, 1),
        "saved_video": str(video_path) if video_path is not None else None,
        "saved_video_bytes": video_path.stat().st_size if video_path is not None else None,
        "tracker_summary": analyzer.tracker.summary if hasattr(analyzer.tracker, "summary") else {},
        "scope": "uncached decode + detection/prediction + tracking + overlay"
                 + (" + MP4 encoding; no GUI" if video_path is not None
                    else "; no GUI or video encoding"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
