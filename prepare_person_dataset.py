"""Prepare a one-class YOLO dataset from MOT17 and CrowdHuman.

MOT17 validation is split by complete sequences to avoid evaluating on frames
from a sequence used for fine-tuning.  Images are hard-linked by default so the
prepared dataset does not duplicate tens of gigabytes on the same drive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_mot17 import SEQUENCES, read_sequence_info
from mot_core import ROOT


DEFAULT_MOT17 = ROOT.parent / "opencv_project" / "data" / "MOT17" / "train"
DEFAULT_CROWDHUMAN = ROOT.parent / "CrowdHuman"
DEFAULT_OUTPUT = ROOT / "data" / "person_mot17_crowdhuman"
DEFAULT_VAL_SEQUENCES = ("MOT17-11-FRCNN", "MOT17-13-FRCNN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot17-root", type=Path, default=DEFAULT_MOT17)
    parser.add_argument("--crowdhuman-root", type=Path, default=DEFAULT_CROWDHUMAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-sequences", nargs="+", default=list(DEFAULT_VAL_SEQUENCES))
    parser.add_argument("--min-visibility", type=float, default=0.10)
    parser.add_argument("--copy-images", action="store_true", help="Copy instead of hard-linking images")
    parser.add_argument("--allow-mot17-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def place_image(source: Path, target: Path, copy_images: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def yolo_row(box: list[float], width: int, height: int) -> str | None:
    x, y, box_width, box_height = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    normalized_width = (x2 - x1) / width
    normalized_height = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {normalized_width:.8f} {normalized_height:.8f}"


def prepare_mot17(args: argparse.Namespace, counts: dict[str, int]) -> None:
    validation = set(args.val_sequences)
    for sequence in SEQUENCES:
        sequence_root = args.mot17_root / sequence
        info = read_sequence_info(sequence_root / "seqinfo.ini")
        width, height = int(info["imWidth"]), int(info["imHeight"])
        annotations: dict[int, list[str]] = defaultdict(list)
        with (sequence_root / "gt" / "gt.txt").open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) < 9:
                    continue
                frame_index = int(float(row[0]))
                marked = int(float(row[6])) == 1
                class_id = int(float(row[7]))
                visibility = float(row[8])
                if not marked or class_id != 1 or visibility < args.min_visibility:
                    continue
                label = yolo_row([float(value) for value in row[2:6]], width, height)
                if label is not None:
                    annotations[frame_index].append(label)

        split = "val" if sequence in validation else "train"
        image_dir = sequence_root / str(info["imDir"])
        for frame_index in range(1, int(info["seqLength"]) + 1):
            source = image_dir / f"{frame_index:06d}{info['imExt']}"
            stem = f"mot17_{sequence}_{frame_index:06d}"
            target_image = args.output / "images" / split / f"{stem}{source.suffix.lower()}"
            target_label = args.output / "labels" / split / f"{stem}.txt"
            place_image(source, target_image, args.copy_images)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(annotations.get(frame_index, [])) + "\n", encoding="utf-8")
            counts[f"mot17_{split}_images"] += 1
            counts[f"mot17_{split}_boxes"] += len(annotations.get(frame_index, []))


def find_crowdhuman_image(root: Path, image_id: str) -> Path:
    for directory in (root / "Images", root / "images", root):
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = directory / f"{image_id}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"CrowdHuman image not found: {image_id}")


def prepare_crowdhuman_split(args: argparse.Namespace, split: str,
                             annotation_path: Path, counts: dict[str, int]) -> None:
    import cv2

    with annotation_path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            record = json.loads(raw)
            image_id = str(record["ID"])
            source = find_crowdhuman_image(args.crowdhuman_root, image_id)
            image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read CrowdHuman image: {source}")
            height, width = image.shape[:2]
            labels = []
            for item in record.get("gtboxes", []):
                if item.get("tag") != "person" or item.get("extra", {}).get("ignore", 0):
                    continue
                label = yolo_row([float(value) for value in item["fbox"]], width, height)
                if label is not None:
                    labels.append(label)
            stem = f"crowdhuman_{image_id}"
            target_image = args.output / "images" / split / f"{stem}{source.suffix.lower()}"
            target_label = args.output / "labels" / split / f"{stem}.txt"
            place_image(source, target_image, args.copy_images)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
            counts[f"crowdhuman_{split}_images"] += 1
            counts[f"crowdhuman_{split}_boxes"] += len(labels)


def prepare_crowdhuman(args: argparse.Namespace, counts: dict[str, int]) -> bool:
    train_annotation = args.crowdhuman_root / "annotation_train.odgt"
    val_annotation = args.crowdhuman_root / "annotation_val.odgt"
    if not train_annotation.is_file() or not val_annotation.is_file():
        if args.allow_mot17_only:
            return False
        raise FileNotFoundError(
            "CrowdHuman is incomplete. Expected annotation_train.odgt and annotation_val.odgt under "
            f"{args.crowdhuman_root}"
        )
    prepare_crowdhuman_split(args, "train", train_annotation, counts)
    prepare_crowdhuman_split(args, "val", val_annotation, counts)
    return True


def main() -> int:
    args = parse_args()
    args.mot17_root = args.mot17_root.resolve()
    args.crowdhuman_root = args.crowdhuman_root.resolve()
    args.output = args.output.resolve()
    invalid_validation = set(args.val_sequences) - set(SEQUENCES)
    if invalid_validation or not args.val_sequences or len(args.val_sequences) == len(SEQUENCES):
        raise ValueError(f"Validation must be a nonempty, proper subset of MOT17 sequences: {sorted(invalid_validation)}")
    missing_sequences = [sequence for sequence in SEQUENCES
                         if not (args.mot17_root / sequence / "seqinfo.ini").is_file()]
    if missing_sequences:
        raise FileNotFoundError(f"Missing MOT17 sequences under {args.mot17_root}: {missing_sequences}")
    crowdhuman_annotations = (
        args.crowdhuman_root / "annotation_train.odgt",
        args.crowdhuman_root / "annotation_val.odgt",
    )
    if not args.allow_mot17_only and not all(path.is_file() for path in crowdhuman_annotations):
        raise FileNotFoundError(
            f"CrowdHuman annotations missing under {args.crowdhuman_root}; "
            "provide --crowdhuman-root or use --allow-mot17-only for a baseline"
        )
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output is not empty; use --overwrite: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    prepare_mot17(args, counts)
    crowdhuman_included = prepare_crowdhuman(args, counts)
    yaml_path = args.output / "person.yaml"
    yaml_path.write_text(
        f"path: {args.output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    mot17_holdout = args.output / "mot17_holdout.txt"
    mot17_holdout.write_text(
        "\n".join(path.as_posix() for path in sorted((args.output / "images" / "val").glob("mot17_*"))) + "\n",
        encoding="utf-8",
    )
    mot17_holdout_yaml = args.output / "person_mot17_holdout.yaml"
    mot17_holdout_yaml.write_text(
        f"path: {args.output.as_posix()}\ntrain: images/train\nval: mot17_holdout.txt\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    summary = {
        "status": "complete",
        "mot17_root": str(args.mot17_root),
        "crowdhuman_root": str(args.crowdhuman_root),
        "crowdhuman_included": crowdhuman_included,
        "mot17_validation_sequences": args.val_sequences,
        "counts": dict(counts),
        "dataset_yaml": str(yaml_path),
        "mot17_holdout_yaml": str(mot17_holdout_yaml),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
