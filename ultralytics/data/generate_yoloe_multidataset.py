# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Discover nested YOLO datasets and generate a YOLOE multi-dataset configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

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
        missing_splits = [split for split in ("train", "val") if not data.get(split)]
        if missing_splits:
            raise ValueError(f"'{yaml_path}' is missing required split(s): {', '.join(missing_splits)}")
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


def generate_multidataset_yaml(
    root: str | Path, output: str | Path, require_chinese_names: bool = False
) -> tuple[Path, list[dict]]:
    """Discover YOLO datasets and write a YOLOE train/validation configuration."""
    datasets = discover_yolo_datasets(root, require_chinese_names=require_chinese_names)
    sources = [str(dataset["yaml"]) for dataset in datasets]
    output = Path(output).expanduser().resolve()
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
    return parser.parse_args()


def main() -> None:
    """Generate the configuration and print a concise discovery summary."""
    args = parse_args()
    output, datasets = generate_multidataset_yaml(args.root, args.output, args.require_chinese_names)
    print(f"Generated '{output}' with {len(datasets)} dataset(s):")
    for dataset in datasets:
        warning = f"; {len(dataset['non_chinese_names'])} non-Chinese name(s)" if dataset["non_chinese_names"] else ""
        print(f"- {dataset['yaml']} ({len(dataset['names'])} classes{warning})")


if __name__ == "__main__":
    main()
