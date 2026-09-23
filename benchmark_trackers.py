"""Run a fixed-detection A/B comparison between the baseline tracker and OC-SORT."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from mot_core import ROOT, MultiObjectTracker, OcSortTracker, YoloDetector


DEFAULT_MODEL = ROOT / "weights" / "轻量级模型.pt"


def summarize(track_rows: Counter[int], next_id: int, maximum_active: int) -> dict:
    lengths = list(track_rows.values())
    return {
        "created_tracks": next_id - 1,
        "visible_track_ids": len(lengths),
        "visible_track_rows": sum(lengths),
        "maximum_active_tracks": maximum_active,
        "tracks_visible_at_most_5_frames": sum(length <= 5 for length in lengths),
        "mean_visible_frames_per_visible_track": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "0"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--low-confidence", type=float, default=0.10)
    parser.add_argument("--track-iou", type=float, default=0.25)
    parser.add_argument("--max-age", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    detector = YoloDetector(DEFAULT_MODEL, device=args.device)
    baseline = MultiObjectTracker(args.track_iou, args.max_age)
    ocsort = OcSortTracker(
        args.track_iou,
        args.max_age,
        high_confidence=args.confidence,
        low_confidence=args.low_confidence,
        second_stage_iou=min(0.10, args.track_iou),
    )
    baseline_rows: Counter[int] = Counter()
    ocsort_rows: Counter[int] = Counter()
    baseline_maximum = 0
    ocsort_maximum = 0
    baseline_tracking_ms = 0.0
    ocsort_tracking_ms = 0.0
    detector_ms = 0.0
    frames = 0
    started = time.perf_counter()
    while args.limit <= 0 or frames < args.limit:
        ok, frame = capture.read()
        if not ok:
            break
        detect_started = time.perf_counter()
        detections = detector.infer(frame, args.low_confidence, 0.55, True)
        detector_ms += (time.perf_counter() - detect_started) * 1000

        baseline_started = time.perf_counter()
        baseline_visible = baseline.update(
            [detection for detection in detections if detection.score >= args.confidence]
        )
        baseline_tracking_ms += (time.perf_counter() - baseline_started) * 1000

        ocsort_started = time.perf_counter()
        ocsort_visible = ocsort.update(detections)
        ocsort_tracking_ms += (time.perf_counter() - ocsort_started) * 1000

        baseline_rows.update(track.track_id for track in baseline_visible)
        ocsort_rows.update(track.track_id for track in ocsort_visible)
        baseline_maximum = max(baseline_maximum, len(baseline_visible))
        ocsort_maximum = max(ocsort_maximum, len(ocsort_visible))
        frames += 1
    capture.release()

    report = {
        "input": str(Path(args.input).resolve()),
        "frames": frames,
        "device": str(detector.device),
        "confidence": args.confidence,
        "low_confidence": args.low_confidence,
        "track_iou": args.track_iou,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mean_detector_ms": round(detector_ms / frames, 3) if frames else 0.0,
        "baseline_mean_tracking_ms": round(baseline_tracking_ms / frames, 5) if frames else 0.0,
        "ocsort_mean_tracking_ms": round(ocsort_tracking_ms / frames, 5) if frames else 0.0,
        "baseline": summarize(baseline_rows, baseline.next_id, baseline_maximum),
        "ocsort": summarize(ocsort_rows, ocsort.next_id, ocsort_maximum),
        "ocsort_stages": ocsort.summary,
        "scope": "Fixed detector stream without ground truth; this is continuity/runtime evidence, not IDF1 or IDSW.",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
