import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from joblib import dump, load

from marine.logger import getLogger


logger: logging.Logger | None = None


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge marine yaml corpus directories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("base_dir", type=Path, help="Base corpus directory")
    parser.add_argument("extra_dir", type=Path, help="Additional corpus directory")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--fixed_eval_id_source_dir",
        type=Path,
        default=None,
        help="Optional source directory that contains val/test ids.pkl files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level",
    )
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return dict(yaml.safe_load(file))


def merge_yaml_dict(
    base_dict: dict[str, Any],
    extra_dict: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    overlap_keys = sorted(set(base_dict.keys()) & set(extra_dict.keys()))
    if len(overlap_keys) > 0:
        raise ValueError(
            f"Duplicated keys found while merging {label}: "
            f"{overlap_keys[:5]} (total: {len(overlap_keys)})"
        )

    merged = dict(base_dict)
    merged.update(extra_dict)
    return merged


def write_yaml(path: Path, content: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            content,
            file,
            allow_unicode=True,
            sort_keys=True,
            default_flow_style=False,
        )


def copy_fixed_eval_ids(source_dir: Path, out_dir: Path) -> None:
    fixed_eval_dir = out_dir / "fixed_eval_ids"
    fixed_eval_dir.mkdir(parents=True, exist_ok=True)

    for phase in ["val", "test"]:
        source_path = source_dir / phase / "ids.pkl"
        if source_path.exists() is False:
            raise FileNotFoundError(f"ids.pkl not found for {phase}: {source_path}")

        target_phase_dir = fixed_eval_dir / phase
        target_phase_dir.mkdir(parents=True, exist_ok=True)
        dump(load(source_path), target_phase_dir / "ids.pkl", compress=True)


def entry(argv: list[str] = sys.argv) -> None:
    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(f"Loaded parameters: {args}")
    assert logger is not None

    base_text = load_yaml(args.base_dir / "text.yaml")
    base_annotation = load_yaml(args.base_dir / "annotation.yaml")
    extra_text = load_yaml(args.extra_dir / "text.yaml")
    extra_annotation = load_yaml(args.extra_dir / "annotation.yaml")

    merged_text = merge_yaml_dict(base_text, extra_text, "text.yaml")
    merged_annotation = merge_yaml_dict(
        base_annotation,
        extra_annotation,
        "annotation.yaml",
    )

    if set(merged_text.keys()) != set(merged_annotation.keys()):
        raise ValueError("Merged text and annotation keys do not match")

    if args.out_dir.exists() is False:
        args.out_dir.mkdir(parents=True)

    write_yaml(args.out_dir / "text.yaml", merged_text)
    write_yaml(args.out_dir / "annotation.yaml", merged_annotation)

    if args.fixed_eval_id_source_dir is not None:
        copy_fixed_eval_ids(args.fixed_eval_id_source_dir, args.out_dir)

    logger.info(
        f"Merged yaml corpora. base: {len(base_text)}, extra: {len(extra_text)}, "
        f"merged: {len(merged_text)}"
    )


if __name__ == "__main__":
    sys.exit(entry())
