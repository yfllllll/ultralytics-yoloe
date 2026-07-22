# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train YOLOE with multiple aggregate multi-dataset configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch
from ultralytics.utils import YAML


def load_multidataset(config: str | Path) -> dict[str, list[str]]:
    """Load one aggregate config and resolve its child YAML paths relative to the aggregate config."""
    config = Path(config).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Multi-dataset configuration not found: '{config}'")
    data = YAML.load(config)
    resolved = {}
    for split in ("train", "val"):
        section = data.get(split)
        if not isinstance(section, dict) or not section.get("yolo_data"):
            raise ValueError(f"'{config}' must define a non-empty {split}.yolo_data list.")
        sources = section["yolo_data"]
        sources = sources if isinstance(sources, (list, tuple)) else [sources]
        resolved[split] = []
        for source in sources:
            source = Path(source).expanduser()
            source = source.resolve() if source.is_absolute() else (config.parent / source).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Child dataset configuration referenced by '{config}' not found: '{source}'")
            resolved[split].append(str(source))
    return resolved


def combine_multidatasets(configs: list[str | Path]) -> dict:
    """Concatenate all child dataset sources in order without deduplicating them."""
    train_sources, val_sources = [], []
    for config in configs:
        data = load_multidataset(config)
        train_sources.extend(data["train"])
        val_sources.extend(data["val"])
    return {"train": {"yolo_data": train_sources}, "val": {"yolo_data": val_sources}}


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOE with two or more multi-dataset YAML files without deduplicating their sources."
    )
    parser.add_argument("configs", nargs="+", type=Path, help="Aggregate yoloe-multidataset.yaml files to combine.")
    parser.add_argument("--model", default="yoloe-26m.yaml", help="YOLOE model YAML or checkpoint.")
    parser.add_argument("--pretrained", default="yoloe-26m-seg.pt", help="Pretrained checkpoint used for fine-tuning.")
    parser.add_argument("--text-model", help="Frozen encoder, e.g. chineseclip:/path/to/chinese-clip.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--nbs", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name", default="merged-multidataset")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Combine the requested datasets and start YOLOE fine-tuning with conservative SGD settings."""
    args = parse_args()
    data = combine_multidatasets(args.configs)
    print(
        f"Loaded {len(args.configs)} aggregate configuration(s): "
        f"{len(data['train']['yolo_data'])} train and {len(data['val']['yolo_data'])} validation dataset entries."
    )

    model = YOLOE(args.model)
    if args.text_model:
        model.set_text_model(args.text_model)
    model.train(
        data=data,
        trainer=YOLOETrainerFromScratch,
        pretrained=args.pretrained,
        optimizer="SGD",
        lr0=0.002,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,
        epochs=args.epochs,
        batch=args.batch,
        nbs=args.nbs,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
