"""Summarize class, bounding-box, and image counts across YOLOE multi-dataset configurations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from ultralytics.data.generate_yoloe_multidataset import _class_names
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import YAML


def _resolve_path(path: str | Path, parent: Path) -> Path:
    """Resolve a user or YAML path relative to its owning configuration."""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def load_child_configs(configs: list[str | Path], count_duplicates: bool = False) -> list[Path]:
    """Resolve child dataset YAML files from aggregate or direct dataset configurations."""
    children, seen = [], set()
    for config in configs:
        config = Path(config).expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(f"Dataset configuration not found: '{config}'")
        data = YAML.load(config)
        sources = []
        if data.get("names") is not None:
            sources.append(config)
        else:
            for split in ("train", "val"):
                section = data.get(split)
                if isinstance(section, dict) and section.get("yolo_data"):
                    values = section["yolo_data"]
                    sources.extend(values if isinstance(values, (list, tuple)) else [values])
            sources = [_resolve_path(source, config.parent) for source in sources]
        if not sources:
            raise ValueError(
                f"'{config}' is neither a YOLO dataset YAML nor an aggregate config with train/val.yolo_data."
            )
        for source in sources:
            source = Path(source).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Child dataset configuration referenced by '{config}' not found: '{source}'")
            if count_duplicates or source not in seen:
                children.append(source)
                seen.add(source)
    return children


def _image_files(source: str | Path, root: Path) -> list[Path]:
    """Resolve image files from a YOLO directory, image, or text-file source."""
    source = _resolve_path(source, root)
    if source.is_dir():
        files = source.rglob("*")
    elif source.is_file() and source.suffix[1:].lower() in IMG_FORMATS:
        files = [source]
    elif source.is_file():
        files = []
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                files.append(_resolve_path(line, source.parent))
    else:
        raise FileNotFoundError(f"Image source does not exist: '{source}'")
    return sorted(path.resolve() for path in files if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS)


def _dataset_images(data: dict, yaml_path: Path, splits: tuple[str, ...]) -> tuple[list[Path], int]:
    """Collect unique images for selected splits and count repeated split references."""
    root = _resolve_path(data.get("path", "."), yaml_path.parent)
    images, references = {}, 0
    for split in splits:
        sources = data.get(split)
        if not sources:
            continue
        for source in sources if isinstance(sources, (list, tuple)) else [sources]:
            for image in _image_files(source, root):
                references += 1
                images[image] = None
    return list(images), references - len(images)


def _read_label(label_path: Path, class_count: int) -> tuple[list[int], int]:
    """Return valid class IDs and invalid row count from one YOLO label file."""
    if not label_path.is_file():
        return [], 0
    classes, invalid = [], 0
    for row in label_path.read_text(encoding="utf-8").splitlines():
        values = row.split()
        try:
            cls_value, x, y, width, height = map(float, values[:5])
            cls_id = int(cls_value)
            valid = (
                len(values) >= 5
                and cls_value == cls_id
                and 0 <= cls_id < class_count
                and all(math.isfinite(value) for value in (x, y, width, height))
                and width > 0
                and height > 0
            )
        except (TypeError, ValueError):
            valid = False
        if valid:
            classes.append(cls_id)
        elif row.strip():
            invalid += 1
    return classes, invalid


def summarize_dataset(yaml_path: str | Path, splits: tuple[str, ...] = ("train", "val")) -> dict:
    """Summarize one YOLO dataset without reading image pixels."""
    yaml_path = Path(yaml_path).resolve()
    data = YAML.load(yaml_path)
    names = _class_names(data, yaml_path)
    images, duplicate_split_images = _dataset_images(data, yaml_path, splits)
    labels = [Path(path) for path in img2label_paths([str(image) for image in images])]
    bbox_counts = [0] * len(names)
    image_counts = [0] * len(names)
    labeled_images = missing_labels = invalid_rows = 0
    for label in labels:
        class_ids, invalid = _read_label(label, len(names))
        invalid_rows += invalid
        missing_labels += int(not label.is_file())
        if class_ids:
            labeled_images += 1
            for cls_id in class_ids:
                bbox_counts[cls_id] += 1
            for cls_id in set(class_ids):
                image_counts[cls_id] += 1

    return {
        "name": data.get("name") or yaml_path.parent.name,
        "config": str(yaml_path),
        "images": len(images),
        "labeled_images": labeled_images,
        "background_images": len(images) - labeled_images,
        "missing_labels": missing_labels,
        "invalid_label_rows": invalid_rows,
        "duplicate_split_images": duplicate_split_images,
        "bboxes": sum(bbox_counts),
        "classes": [
            {"id": cls_id, "name": name, "bboxes": bbox_counts[cls_id], "images": image_counts[cls_id]}
            for cls_id, name in enumerate(names)
        ],
    }


def summarize_configs(
    configs: list[str | Path],
    splits: tuple[str, ...] = ("train", "val"),
    count_duplicates: bool = False,
) -> dict:
    """Aggregate exact class names across all referenced child datasets."""
    child_configs = load_child_configs(configs, count_duplicates=count_duplicates)
    datasets = [summarize_dataset(config, splits) for config in child_configs]
    classes = defaultdict(
        lambda: {
            "bboxes": 0,
            "images": 0,
            "declared_datasets": [],
            "annotated_datasets": [],
        }
    )
    for dataset in datasets:
        for item in dataset["classes"]:
            stats = classes[item["name"]]
            stats["bboxes"] += item["bboxes"]
            stats["images"] += item["images"]
            stats["declared_datasets"].append(dataset["config"])
            if item["bboxes"]:
                stats["annotated_datasets"].append(dataset["config"])

    class_rows = [
        {
            "name": name,
            "bboxes": stats["bboxes"],
            "images": stats["images"],
            "declared_dataset_count": len(stats["declared_datasets"]),
            "annotated_dataset_count": len(stats["annotated_datasets"]),
            "declared_datasets": stats["declared_datasets"],
            "annotated_datasets": stats["annotated_datasets"],
        }
        for name, stats in classes.items()
    ]
    class_rows.sort(key=lambda item: (-item["bboxes"], -item["images"], item["name"]))
    return {
        "configs": [str(Path(config).expanduser().resolve()) for config in configs],
        "splits": list(splits),
        "dataset_count": len(datasets),
        "images": sum(dataset["images"] for dataset in datasets),
        "bboxes": sum(dataset["bboxes"] for dataset in datasets),
        "class_count": len(class_rows),
        "datasets": datasets,
        "classes": class_rows,
    }


def save_summary(summary: dict, output_dir: str | Path, min_bboxes: int = 1, min_images: int = 1, top: int = 0) -> dict:
    """Write JSON, CSV, and service-ready prompt files and return prompt selection metadata."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        item["name"]
        for item in summary["classes"]
        if item["bboxes"] >= min_bboxes and item["images"] >= min_images
    ]
    if top:
        prompts = prompts[:top]
    summary = {**summary, "recommended_prompts": prompts}
    (output_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "prompts.txt").write_text("".join(f"{name}\n" for name in prompts), encoding="utf-8")

    with (output_dir / "classes.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class_name",
                "bbox_count",
                "image_count",
                "annotated_dataset_count",
                "declared_dataset_count",
                "annotated_datasets",
            ]
        )
        for item in summary["classes"]:
            writer.writerow(
                [
                    item["name"],
                    item["bboxes"],
                    item["images"],
                    item["annotated_dataset_count"],
                    item["declared_dataset_count"],
                    " | ".join(item["annotated_datasets"]),
                ]
            )

    with (output_dir / "datasets.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dataset",
                "config",
                "images",
                "labeled_images",
                "background_images",
                "bbox_count",
                "missing_labels",
                "invalid_label_rows",
                "duplicate_split_images",
            ]
        )
        for dataset in summary["datasets"]:
            writer.writerow(
                [
                    dataset["name"],
                    dataset["config"],
                    dataset["images"],
                    dataset["labeled_images"],
                    dataset["background_images"],
                    dataset["bboxes"],
                    dataset["missing_labels"],
                    dataset["invalid_label_rows"],
                    dataset["duplicate_split_images"],
                ]
            )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Count classes, bounding boxes, and images across YOLO datasets.")
    parser.add_argument("configs", nargs="+", type=Path, help="Aggregate multi-dataset or direct child YAML files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test", "minival"),
        default=("train", "val"),
        help="Child dataset splits to scan (default: train val).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset-stats"), help="Output directory.")
    parser.add_argument("--min-bboxes", type=int, default=1, help="Minimum bbox count for prompts.txt.")
    parser.add_argument("--min-images", type=int, default=1, help="Minimum image count for prompts.txt.")
    parser.add_argument("--top", type=int, default=80, help="Keep the top N prompt classes (default: 80); 0 keeps all.")
    parser.add_argument(
        "--count-duplicate-configs",
        action="store_true",
        help="Count repeated references to the exact same child YAML instead of scanning it once.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset statistics and print the most frequent classes."""
    args = parse_args()
    if args.min_bboxes < 0 or args.min_images < 0 or args.top < 0:
        raise ValueError("--min-bboxes, --min-images, and --top must be non-negative.")
    summary = summarize_configs(args.configs, tuple(args.splits), args.count_duplicate_configs)
    summary = save_summary(summary, args.output_dir, args.min_bboxes, args.min_images, args.top)
    print(
        f"Scanned {summary['dataset_count']} dataset(s), {summary['images']} image entries, "
        f"{summary['bboxes']} bboxes, and {summary['class_count']} exact class names."
    )
    print(f"{'bbox':>10} {'images':>10}  class")
    for item in summary["classes"][: min(30, len(summary["classes"]))]:
        print(f"{item['bboxes']:>10} {item['images']:>10}  {item['name']}")
    print(f"Saved {len(summary['recommended_prompts'])} recommended prompts to '{args.output_dir.resolve()}'.")


if __name__ == "__main__":
    main()
