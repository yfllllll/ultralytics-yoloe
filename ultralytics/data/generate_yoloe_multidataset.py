# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Discover nested YOLO datasets and generate a YOLOE multi-dataset configuration."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import YAML

DATA_YAML_NAMES = ("data.yaml", "dataset.yaml")


def _class_names(data: dict, yaml_path: Path) -> list[str]:
    """Validate and return class names ordered by class ID."""
    names = data.get("names")
    if isinstance(names, list):
        ordered = names
    elif isinstance(names, dict):
        try:
            indexed = {int(k): v for k, v in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Class IDs in '{yaml_path}' must be integers.") from error
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise ValueError(f"Class IDs in '{yaml_path}' must be contiguous from 0, but found {sorted(indexed)}.")
        ordered = [indexed[i] for i in expected]
    else:
        raise ValueError(f"'{yaml_path}' must contain a non-empty 'names' list or mapping.")
    if not ordered or any(not isinstance(name, str) or not name.strip() for name in ordered):
        raise ValueError(f"Class names in '{yaml_path}' must be non-empty strings.")
    return ordered


def _has_chinese(text: str) -> bool:
    """Return whether text contains a CJK Unified Ideograph."""
    return any("\u3400" <= character <= "\u9fff" for character in text)


def discover_yolo_datasets(root: str | Path, require_chinese_names: bool = False) -> list[dict]:
    """Recursively find valid YOLO dataset roots below a directory.

    A dataset root must contain ``images/``, ``labels/``, and either ``data.yaml`` or ``dataset.yaml``. If both YAML
    files exist, ``data.yaml`` takes precedence.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset search root does not exist or is not a directory: '{root}'")

    candidates = {}
    for filename in reversed(DATA_YAML_NAMES):
        for yaml_path in root.rglob(filename):
            dataset_root = yaml_path.parent
            if (dataset_root / "images").is_dir() and (dataset_root / "labels").is_dir():
                candidates[dataset_root.resolve()] = yaml_path.resolve()

    datasets = []
    for dataset_root, yaml_path in sorted(candidates.items(), key=lambda item: str(item[0])):
        data = YAML.load(yaml_path)
        names = _class_names(data, yaml_path)
        non_chinese = [name for name in names if not _has_chinese(name)]
        if require_chinese_names and non_chinese:
            raise ValueError(f"'{yaml_path}' contains non-Chinese class names: {non_chinese}")
        datasets.append(
            {
                "root": dataset_root,
                "yaml": yaml_path,
                "names": names,
                "non_chinese_names": non_chinese,
            }
        )

    if not datasets:
        raise FileNotFoundError(
            f"No YOLO datasets found below '{root}'. Each dataset must contain images/, labels/, and "
            "data.yaml or dataset.yaml."
        )
    return datasets


def _split_images(dataset: dict, val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Return deterministic train and validation image splits for one dataset."""
    images_dir = dataset["root"] / "images"
    images = sorted(
        path.resolve() for path in images_dir.rglob("*") if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS
    )
    if len(images) < 2:
        raise ValueError(f"'{images_dir}' must contain at least 2 images to create separate train and val splits.")

    random.Random(seed).shuffle(images)
    val_count = min(len(images) - 1, max(1, int(len(images) * val_ratio + 0.5)))
    return sorted(images[val_count:]), sorted(images[:val_count])


def _write_image_list(path: Path, images: list[Path]) -> None:
    """Write absolute image paths in Ultralytics dataset-list format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image}\n" for image in images), encoding="utf-8")


def generate_multidataset_yaml(
    root: str | Path,
    output: str | Path,
    require_chinese_names: bool = False,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[Path, list[dict]]:
    """Discover flat YOLO datasets, split their images, and write complete per-dataset and aggregate configurations."""
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, but received {val_ratio}.")

    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    datasets = discover_yolo_datasets(root, require_chinese_names=require_chinese_names)
    generated_root = output.parent / f"{output.stem}_datasets"
    sources = []
    for dataset in datasets:
        relative_root = dataset["root"].relative_to(root)
        generated_dir = generated_root / (relative_root if relative_root.parts else Path(dataset["root"].name))
        train_images, val_images = _split_images(dataset, val_ratio, seed)
        train_list, val_list = generated_dir / "train.txt", generated_dir / "val.txt"
        generated_yaml = generated_dir / "data.yaml"
        _write_image_list(train_list, train_images)
        _write_image_list(val_list, val_images)
        YAML.save(
            generated_yaml,
            {
                "path": str(dataset["root"]),
                "train": str(train_list.resolve()),
                "val": str(val_list.resolve()),
                "names": dict(enumerate(dataset["names"])),
            },
            header=f"# Generated from {dataset['yaml']} without moving image or label files.\n",
        )
        dataset.update(
            {
                "generated_yaml": generated_yaml.resolve(),
                "train_images": train_images,
                "val_images": val_images,
            }
        )
        sources.append(str(generated_yaml.resolve()))

    YAML.save(
        output,
        {"train": {"yolo_data": sources}, "val": {"yolo_data": sources.copy()}},
        header="# YOLOE multi-dataset configuration generated automatically.\n",
    )
    return output, datasets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Recursively discover YOLO datasets and generate a YOLOE multi-dataset YAML file."
    )
    parser.add_argument("root", type=Path, help="Directory containing nested YOLO dataset folders.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("yoloe-multidataset.yaml"),
        help="Output YAML path (default: ./yoloe-multidataset.yaml).",
    )
    parser.add_argument(
        "--require-chinese-names",
        action="store_true",
        help="Fail if any discovered class name does not contain Chinese characters.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Fraction of each dataset assigned to validation (default: 0.2)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random split seed (default: 0).")
    return parser.parse_args()


def main() -> None:
    """Generate the configuration and print a concise discovery summary."""
    args = parse_args()
    output, datasets = generate_multidataset_yaml(
        args.root, args.output, args.require_chinese_names, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"Generated '{output}' with {len(datasets)} dataset(s):")
    for dataset in datasets:
        warning = f"; {len(dataset['non_chinese_names'])} non-Chinese name(s)" if dataset["non_chinese_names"] else ""
        print(
            f"- {dataset['generated_yaml']} ({len(dataset['names'])} classes, "
            f"{len(dataset['train_images'])} train, {len(dataset['val_images'])} val{warning})"
        )


if __name__ == "__main__":
    main()
