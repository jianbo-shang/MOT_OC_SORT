"""Evaluate the baseline and OC-SORT on the seven unique MOT17 train videos.

The runner caches YOLO detections once, replays the exact same cache through
both trackers, exports MOTChallenge text files, and optionally invokes the
official TrackEval implementation.  It is intentionally separate from the GUI
so playback throttling and rendering do not contaminate metric timings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mot_core import (Detection, MultiObjectTracker, OcSortTracker,
                      OnnxPersonReIdEncoder, ROOT, YoloDetector, read_image)


SEQUENCES = (
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
)
DEFAULT_DATASET_ROOT = ROOT.parent / "opencv_project" / "data" / "MOT17" / "train"
DEFAULT_TRACKEVAL_ROOT = ROOT.parent / "opencv_project" / "third_party" / "TrackEval"
DEFAULT_MODEL = ROOT / "weights" / "轻量级模型.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--trackeval-root", type=Path, default=DEFAULT_TRACKEVAL_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "0"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--low-confidence", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--track-iou", type=float, default=0.25)
    parser.add_argument("--max-age", type=int, default=20)
    parser.add_argument("--ocsort-min-hits", type=int, default=2,
                        help="Number of detection observations needed to confirm an OC-SORT track")
    parser.add_argument("--detection-interval", type=int, default=1,
                        help="Run cached detector output every N frames and predict in between")
    parser.add_argument("--reid-model", type=Path,
                        help="Optional person ReID ONNX model for conditional TOPIC-Lite association")
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--limit", type=int, default=0, help="Limit frames per sequence for a smoke test")
    parser.add_argument("--resume", action="store_true", help="Reuse complete detection caches")
    parser.add_argument("--skip-trackeval", action="store_true")
    return parser.parse_args()


def read_sequence_info(path: Path) -> dict[str, str | int]:
    values: dict[str, str | int] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if key in {"seqLength", "frameRate", "imWidth", "imHeight"}:
            values[key] = int(value)
        elif key in {"imDir", "imExt", "name"}:
            values[key] = value
    required = {"seqLength", "frameRate", "imWidth", "imHeight", "imDir", "imExt"}
    if not required.issubset(values):
        raise ValueError(f"Incomplete seqinfo.ini: {path}")
    return values


def validate_inputs(args: argparse.Namespace) -> dict[str, dict[str, str | int]]:
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not 0.0 <= args.low_confidence < args.confidence <= 1.0:
        raise ValueError("Require 0 <= low-confidence < confidence <= 1")
    if args.limit < 0 or args.max_age < 1 or args.detection_interval < 1 or args.ocsort_min_hits < 1:
        raise ValueError("limit must be >= 0; max-age and detection-interval must be >= 1")
    if args.reid_model is not None and not args.reid_model.is_file():
        raise FileNotFoundError(f"ReID model not found: {args.reid_model}")
    infos: dict[str, dict[str, str | int]] = {}
    for sequence in args.sequences:
        sequence_root = args.dataset_root / sequence
        info_path = sequence_root / "seqinfo.ini"
        gt_path = sequence_root / "gt" / "gt.txt"
        if not info_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(f"Incomplete MOT17 sequence: {sequence_root}")
        info = read_sequence_info(info_path)
        first_image = sequence_root / str(info["imDir"]) / f"000001{info['imExt']}"
        if not first_image.is_file():
            raise FileNotFoundError(f"First image not found: {first_image}")
        infos[sequence] = info
    if not args.skip_trackeval and not (args.trackeval_root / "trackeval" / "__init__.py").is_file():
        raise FileNotFoundError(f"TrackEval not found: {args.trackeval_root}")
    return infos


def cache_paths(output: Path, sequence: str) -> tuple[Path, Path]:
    cache_dir = output / "detection_cache"
    return cache_dir / f"{sequence}.csv", cache_dir / f"{sequence}.json"


def cache_is_complete(manifest_path: Path, expected_frames: int, settings: dict[str, Any]) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return manifest.get("status") == "complete" and manifest.get("frames") == expected_frames \
        and manifest.get("settings") == settings


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_detection_cache(args: argparse.Namespace, output: Path,
                          infos: dict[str, dict[str, str | int]]) -> dict[str, Any]:
    settings = {
        "model": str(args.model.resolve()),
        "model_sha256": file_sha256(args.model),
        "low_confidence": args.low_confidence,
        "nms_iou": args.nms_iou,
        "class_id": 0,
        "device": args.device,
    }
    expected_by_sequence = {
        sequence: min(int(info["seqLength"]), args.limit) if args.limit else int(info["seqLength"])
        for sequence, info in infos.items()
    }
    pending = []
    for sequence, expected in expected_by_sequence.items():
        csv_path, manifest_path = cache_paths(output, sequence)
        if not (args.resume and csv_path.is_file() and cache_is_complete(manifest_path, expected, settings)):
            pending.append(sequence)
    if not pending:
        print("Detection caches are complete; reusing them.", flush=True)
        return settings

    detector = YoloDetector(args.model, device=args.device)
    for sequence in pending:
        info = infos[sequence]
        expected_frames = expected_by_sequence[sequence]
        csv_path, manifest_path = cache_paths(output, sequence)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = csv_path.with_suffix(".tmp")
        detections_count = 0
        elapsed_ms = 0.0
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "class_id", "confidence", "x1", "y1", "x2", "y2"])
            for frame_index in range(1, expected_frames + 1):
                image_path = args.dataset_root / sequence / str(info["imDir"]) / f"{frame_index:06d}{info['imExt']}"
                frame = read_image(image_path)
                if frame is None:
                    raise RuntimeError(f"Cannot read MOT17 image: {image_path}")
                started = time.perf_counter()
                detections = detector.infer(frame, args.low_confidence, args.nms_iou, traffic_only=False)
                elapsed_ms += (time.perf_counter() - started) * 1000.0
                for detection in detections:
                    if detection.class_id != 0:
                        continue
                    x1, y1, x2, y2 = detection.xyxy.tolist()
                    writer.writerow([
                        frame_index, detection.class_id, f"{detection.score:.8f}",
                        f"{x1:.5f}", f"{y1:.5f}", f"{x2:.5f}", f"{y2:.5f}",
                    ])
                    detections_count += 1
                if frame_index % 100 == 0 or frame_index == expected_frames:
                    print(f"CACHE {sequence}: {frame_index}/{expected_frames}", flush=True)
        temporary.replace(csv_path)
        manifest = {
            "status": "complete",
            "sequence": sequence,
            "frames": expected_frames,
            "detections": detections_count,
            "mean_detector_ms": elapsed_ms / max(1, expected_frames),
            "resolved_device": str(detector.device),
            "settings": settings,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return settings


def read_detection_cache(path: Path, expected_frames: int) -> list[list[Detection]]:
    frames: list[list[Detection]] = [[] for _ in range(expected_frames)]
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frame", "class_id", "confidence", "x1", "y1", "x2", "y2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected detection cache columns: {path}")
        for row in reader:
            frame_index = int(row["frame"])
            if not 1 <= frame_index <= expected_frames:
                raise ValueError(f"Frame index outside cache range in {path}: {frame_index}")
            frames[frame_index - 1].append(Detection(
                np.array([float(row["x1"]), float(row["y1"]),
                          float(row["x2"]), float(row["y2"])], dtype=np.float32),
                float(row["confidence"]),
                int(row["class_id"]),
            ))
    return frames


def motchallenge_row(frame_index: int, track: Any, frame_width: int,
                     frame_height: int) -> tuple[Any, ...] | None:
    x1, y1, x2, y2 = map(float, track.xyxy)
    x1 = max(0.0, min(x1, frame_width - 1.0))
    y1 = max(0.0, min(y1, frame_height - 1.0))
    x2 = max(0.0, min(x2, float(frame_width)))
    y2 = max(0.0, min(y2, float(frame_height)))
    width = x2 - x1
    height = y2 - y1
    if width < 1.0 or height < 1.0:
        return None
    return frame_index, track.track_id, x1, y1, width, height, track.score, -1, -1, -1


def run_trackers(args: argparse.Namespace, output: Path,
                 infos: dict[str, dict[str, str | int]]) -> dict[str, Any]:
    tracker_root = output / "trackers" / "MOT17-train"
    aggregate: dict[str, dict[str, Any]] = {
        "baseline": {"frames": 0, "rows": 0, "ids": 0, "tracking_ms": 0.0},
        "ocsort": {"frames": 0, "rows": 0, "ids": 0, "tracking_ms": 0.0},
    }
    appearance_encoder = OnnxPersonReIdEncoder(args.reid_model) if args.reid_model else None
    for sequence, info in infos.items():
        expected_frames = min(int(info["seqLength"]), args.limit) if args.limit else int(info["seqLength"])
        detections_by_frame = read_detection_cache(cache_paths(output, sequence)[0], expected_frames)
        variants = {
            "baseline": MultiObjectTracker(args.track_iou, args.max_age),
            "ocsort": OcSortTracker(
                args.track_iou,
                args.max_age,
                min_hits=args.ocsort_min_hits,
                high_confidence=args.confidence,
                low_confidence=args.low_confidence,
                second_stage_iou=min(0.10, args.track_iou),
                appearance_encoder=appearance_encoder,
            ),
        }
        handles: dict[str, Any] = {}
        writers: dict[str, Any] = {}
        counters: dict[str, Counter[int]] = {key: Counter() for key in variants}
        timings = {key: 0.0 for key in variants}
        try:
            for key in variants:
                target = tracker_root / key / "data" / f"{sequence}.txt"
                target.parent.mkdir(parents=True, exist_ok=True)
                handles[key] = target.open("w", encoding="utf-8", newline="")
                writers[key] = csv.writer(handles[key], lineterminator="\n")
            for frame_index, detections in enumerate(detections_by_frame, start=1):
                detector_ran = (frame_index - 1) % args.detection_interval == 0
                frame = None
                if detector_ran and appearance_encoder is not None:
                    image_path = (args.dataset_root / sequence / str(info["imDir"])
                                  / f"{frame_index:06d}{info['imExt']}")
                    frame = read_image(image_path)
                    if frame is None:
                        raise RuntimeError(f"Cannot read MOT17 image for ReID: {image_path}")
                for key, tracker in variants.items():
                    tracker_input: Iterable[Detection]
                    if key == "baseline":
                        tracker_input = [item for item in detections if item.score >= args.confidence]
                    else:
                        tracker_input = detections
                    started = time.perf_counter()
                    if detector_ran:
                        visible = tracker.update(list(tracker_input), frame=frame) if key == "ocsort" \
                            else tracker.update(list(tracker_input))
                    else:
                        visible = tracker.predict_only()
                    timings[key] += (time.perf_counter() - started) * 1000.0
                    for track in visible:
                        row = motchallenge_row(
                            frame_index, track, int(info["imWidth"]), int(info["imHeight"])
                        )
                        if row is not None:
                            writers[key].writerow((
                                row[0], row[1], f"{row[2]:.5f}", f"{row[3]:.5f}",
                                f"{row[4]:.5f}", f"{row[5]:.5f}", f"{row[6]:.8f}",
                                row[7], row[8], row[9],
                            ))
                            counters[key][track.track_id] += 1
        finally:
            for handle in handles.values():
                handle.close()

        for key, tracker in variants.items():
            summary = aggregate[key]
            summary["frames"] += expected_frames
            summary["rows"] += sum(counters[key].values())
            summary["ids"] += len(counters[key])
            summary["tracking_ms"] += timings[key]
            if key == "ocsort":
                stages = summary.setdefault("stages", defaultdict(int))
                for stage, value in tracker.summary.items():
                    stages[stage] += value
        print(f"TRACK {sequence}: {expected_frames} frames complete", flush=True)

    for summary in aggregate.values():
        summary["mean_tracking_ms"] = summary.pop("tracking_ms") / max(1, summary["frames"])
        if isinstance(summary.get("stages"), defaultdict):
            summary["stages"] = dict(summary["stages"])
    (output / "tracker_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return aggregate


def prepare_ground_truth(args: argparse.Namespace, output: Path) -> Path:
    gt_root = output / "ground_truth"
    for sequence in args.sequences:
        target = gt_root / "MOT17-train" / sequence / "gt" / "gt.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        source = args.dataset_root / sequence / "gt" / "gt.txt"
        if args.limit <= 0:
            shutil.copy2(source, target)
        else:
            with source.open("r", encoding="utf-8", newline="") as input_stream, target.open(
                "w", encoding="utf-8", newline=""
            ) as output_stream:
                writer = csv.writer(output_stream, lineterminator="\n")
                for row in csv.reader(input_stream):
                    if row and int(float(row[0])) <= args.limit:
                        writer.writerow(row)
    return gt_root


def import_trackeval(root: Path) -> Any:
    # The pinned TrackEval checkout predates NumPy 1.24 aliases.
    for name, value in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    sys.path.insert(0, str(root.resolve()))
    import trackeval  # type: ignore

    return trackeval


def metric_row(variant: str, scope: str, values: dict[str, Any]) -> dict[str, Any]:
    hota = values["HOTA"]
    clear = values["CLEAR"]
    identity = values["Identity"]
    return {
        "variant": variant,
        "scope": scope,
        "hota": float(np.mean(hota["HOTA"])),
        "deta": float(np.mean(hota["DetA"])),
        "assa": float(np.mean(hota["AssA"])),
        "mota": float(clear["MOTA"]),
        "idf1": float(identity["IDF1"]),
        "idsw": int(clear["IDSW"]),
        "frag": int(clear["Frag"]),
        "fn": int(clear["CLR_FN"]),
        "fp": int(clear["CLR_FP"]),
    }


def run_trackeval(args: argparse.Namespace, output: Path,
                  infos: dict[str, dict[str, str | int]]) -> list[dict[str, Any]]:
    trackeval = import_trackeval(args.trackeval_root)
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "BREAK_ON_ERROR": True,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
        "LOG_ON_ERROR": None,
    })
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(prepare_ground_truth(args, output)),
        "TRACKERS_FOLDER": str(output / "trackers"),
        "OUTPUT_FOLDER": str(output / "trackeval_native"),
        "TRACKERS_TO_EVAL": ["baseline", "ocsort"],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,
        "DO_PREPROC": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "SEQ_INFO": {
            sequence: min(int(infos[sequence]["seqLength"]), args.limit)
            if args.limit else int(infos[sequence]["seqLength"])
            for sequence in args.sequences
        },
        "SKIP_SPLIT_FOL": False,
    })
    metric_config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(metric_config),
        trackeval.metrics.CLEAR(metric_config),
        trackeval.metrics.Identity(metric_config),
    ]
    results, messages = evaluator.evaluate([dataset], metrics)
    dataset_key = "MotChallenge2DBox"
    rows: list[dict[str, Any]] = []
    for variant in ("baseline", "ocsort"):
        if messages[dataset_key][variant] != "Success":
            raise RuntimeError(f"TrackEval failed for {variant}: {messages[dataset_key][variant]}")
        values = results[dataset_key][variant]
        rows.append(metric_row(variant, "COMBINED", values["COMBINED_SEQ"]["pedestrian"]))
        for sequence in args.sequences:
            rows.append(metric_row(variant, sequence, values[sequence]["pedestrian"]))
    with (output / "trackeval_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_report(output: Path, rows: list[dict[str, Any]], tracker_summary: dict[str, Any],
                 args: argparse.Namespace) -> None:
    combined = {row["variant"]: row for row in rows if row["scope"] == "COMBINED"}
    baseline = combined["baseline"]
    improved = combined["ocsort"]
    report = output / "report.md"
    improved_label = "TOPIC-Lite" if args.reid_model else \
        f"OC-SORT + Detect/{args.detection_interval}" if args.detection_interval > 1 else "OC-SORT"
    with report.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# YOLOv8 + {improved_label} MOT17 TrackEval 报告\n\n")
        stream.write(
            f"评测范围：MOT17 训练集 {len(args.sequences)} 个不重复 FRCNN 序列。"
            "两个跟踪器消费同一份 YOLOv8 检测缓存；这是本地训练集回归评测，"
            "不是 MOTChallenge 隐藏测试集排行榜成绩。"
            f"检测间隔为 {args.detection_interval} 帧；"
            f"条件 ReID {'已启用' if args.reid_model else '未启用'}。\n\n"
        )
        stream.write("|方案|HOTA|DetA|AssA|MOTA|IDF1|IDSW|Frag|FN|FP|跟踪均值 ms|\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for key, label in (("baseline", "Kalman-Hungarian"), ("ocsort", improved_label)):
            row = combined[key]
            timing = tracker_summary[key]["mean_tracking_ms"]
            stream.write(
                f"|{label}|{row['hota'] * 100:.3f}|{row['deta'] * 100:.3f}|"
                f"{row['assa'] * 100:.3f}|{row['mota'] * 100:.3f}|{row['idf1'] * 100:.3f}|"
                f"{row['idsw']}|{row['frag']}|{row['fn']}|{row['fp']}|{timing:.4f}|\n"
            )
        stream.write(f"\n## {improved_label} 相对基线\n\n")
        stream.write(f"- HOTA：{(improved['hota'] - baseline['hota']) * 100:+.3f} 个百分点。\n")
        stream.write(f"- AssA：{(improved['assa'] - baseline['assa']) * 100:+.3f} 个百分点。\n")
        stream.write(f"- IDF1：{(improved['idf1'] - baseline['idf1']) * 100:+.3f} 个百分点。\n")
        stream.write(f"- IDSW：{improved['idsw'] - baseline['idsw']:+d}。\n")
        stream.write(f"- Frag：{improved['frag'] - baseline['frag']:+d}。\n")
        stream.write("\n逐序列结果见 `trackeval_metrics.csv`，阶段统计见 `tracker_summary.json`。\n")


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.trackeval_root = args.trackeval_root.resolve()
    args.model = args.model.resolve()
    if args.reid_model is not None:
        args.reid_model = args.reid_model.resolve()
    if args.output is None:
        args.output = ROOT / "outputs" / datetime.now().strftime("mot17_eval_%Y%m%d_%H%M%S")
    else:
        args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output is not empty; use --resume or choose another path: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    infos = validate_inputs(args)
    settings = build_detection_cache(args, args.output, infos)
    tracker_summary = run_trackers(args, args.output, infos)
    run_manifest = {
        "status": "trackers_complete",
        "dataset_root": str(args.dataset_root),
        "trackeval_root": str(args.trackeval_root),
        "sequences": args.sequences,
        "limit": args.limit,
        "confidence": args.confidence,
        "low_confidence": args.low_confidence,
        "track_iou": args.track_iou,
        "max_age": args.max_age,
        "ocsort_min_hits": args.ocsort_min_hits,
        "detection_interval": args.detection_interval,
        "reid_model": str(args.reid_model) if args.reid_model else None,
        "detector": settings,
        "tracker_summary": tracker_summary,
    }
    if not args.skip_trackeval:
        rows = run_trackeval(args, args.output, infos)
        write_report(args.output, rows, tracker_summary, args)
        run_manifest["status"] = "complete"
        run_manifest["combined_metrics"] = [row for row in rows if row["scope"] == "COMBINED"]
    (args.output / "summary.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": run_manifest["status"], "output": str(args.output)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
