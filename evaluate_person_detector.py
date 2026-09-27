"""Report person detector Precision, Recall, mAP50 and mAP50-95."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from mot_core import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "person_mot17_crowdhuman" / "person_mot17_holdout.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "weights" / "person_mot17_crowdhuman.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.is_file() or not args.model.is_file():
        raise FileNotFoundError("Dataset yaml and trained model are both required")
    output = args.output or ROOT / "outputs" / datetime.now().strftime("person_eval_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()), device=args.device, imgsz=args.imgsz,
        batch=args.batch, conf=args.conf, plots=True, project=str(output.parent), name=output.name,
        exist_ok=True,
    )
    values = metrics.results_dict
    report = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "confidence": args.conf,
        "precision": float(values["metrics/precision(B)"]),
        "recall": float(values["metrics/recall(B)"]),
        "map50": float(values["metrics/mAP50(B)"]),
        "map50_95": float(values["metrics/mAP50-95(B)"]),
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
    }
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
