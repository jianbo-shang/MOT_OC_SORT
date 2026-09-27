"""Fine-tune the bundled YOLO detector for one-class person detection."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mot_core import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "person_mot17_crowdhuman" / "person.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "weights" / "轻量级模型.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Conservative initial learning rate for fine-tuning")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Freeze early backbone layers to reduce forgetting")
    parser.add_argument("--patience", type=int, default=10,
                        help="Stop after this many validation epochs without improvement")
    parser.add_argument("--project", type=Path, default=ROOT / "outputs" / "person_finetune")
    parser.add_argument("--name", default="mot17_crowdhuman")
    parser.add_argument("--export-best", type=Path, default=ROOT / "weights" / "person_mot17_crowdhuman.pt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Initial model not found: {args.model}")
    if args.epochs < 1 or args.batch < 1 or args.lr0 <= 0 or args.freeze < 0 or args.patience < 1:
        raise ValueError("epochs, batch, lr0 and patience must be positive; freeze must be nonnegative")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    if args.freeze:
        def freeze_early_layers(trainer) -> None:
            for layer in list(trainer.model.model)[:args.freeze]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)

        model.add_callback("on_pretrain_routine_end", freeze_early_layers)
    model.train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        project=str(args.project.resolve()), name=args.name, exist_ok=False,
        pretrained=True, single_cls=True, lr0=args.lr0,
        patience=args.patience,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError(f"Training completed but best checkpoint is missing: {best}")
    args.export_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.export_best)
    result = {
        "status": "complete", "best": str(best), "exported": str(args.export_best.resolve()),
        "epochs": args.epochs, "lr0": args.lr0, "freeze": args.freeze,
        "patience": args.patience,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
