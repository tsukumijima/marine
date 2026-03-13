import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import yaml

from marine.bin.make_raw_corpus import parse_jsut_annotation
from marine.types import AccentRepresentMode
from marine.utils.jsut_annotation import get_annotation_structure_errors
from marine.utils.jsut_source import strip_jsut_annotation_symbols


SUSPICIOUS_TERMINAL_MARKER_PATTERN = re.compile(r"[#_]\\?\$$")


def get_parser() -> argparse.ArgumentParser:
    """
    YAML corpus を検証する CLI の引数を生成する。

    Returns:
        argparse.ArgumentParser: 引数パーサー
    """

    parser = argparse.ArgumentParser(
        description="Validate marine yaml corpus files and report suspicious annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Directory that contains corpus.yaml (or legacy text.yaml + annotation.yaml).",
    )
    parser.add_argument(
        "--accent_status_seq_level",
        "-s",
        type=str,
        choices=["ap", "mora"],
        default="mora",
        help="Sequence level used for parser validation.",
    )
    parser.add_argument(
        "--accent_status_represent_mode",
        "-m",
        type=str,
        choices=["binary", "high_low"],
        default="binary",
        help="Accent representation mode used for parser validation.",
    )
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    """
    YAML ファイルを辞書として読み込む。

    Args:
        path (Path): YAML パス

    Returns:
        dict[str, Any]: 読み込んだ辞書
    """

    with open(path, encoding="utf-8") as file:
        loaded_yaml = yaml.safe_load(file)

    if loaded_yaml is None:
        return {}

    return dict(loaded_yaml)


def validate_parseable_annotations(
    annotations: dict[str, str],
    accent_status_seq_level: str,
    accent_status_represent_mode: AccentRepresentMode,
) -> list[str]:
    """
    annotation を既存 parser で検証し、異常な script id を返す。

    Args:
        annotations (dict[str, str]): script id ごとの annotation
        accent_status_seq_level (str): アクセント系列粒度
        accent_status_represent_mode (str): アクセント表現

    Returns:
        list[str]: parse できなかった script id 一覧
    """

    invalid_script_ids: list[str] = []

    for script_id, annotation in annotations.items():
        try:
            parse_jsut_annotation(
                annotation,
                accent_status_seq_level,
                accent_status_represent_mode,
            )
        except Exception:
            invalid_script_ids.append(script_id)

    return invalid_script_ids


def validate_structural_annotations(
    annotations: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """
    annotation の prosody 構文を検証し、異常な script id と理由を返す。

    Args:
        annotations (dict[str, str]): script id ごとの annotation

    Returns:
        list[tuple[str, list[str]]]: script id と構文エラー一覧
    """

    invalid_annotations: list[tuple[str, list[str]]] = []

    for script_id, annotation in annotations.items():
        structural_errors = get_annotation_structure_errors(annotation)
        if len(structural_errors) > 0:
            invalid_annotations.append((script_id, structural_errors))

    return invalid_annotations


def detect_suspicious_terminal_markers(
    annotations: dict[str, str],
) -> list[str]:
    """
    終端直前に不自然な prosody marker を持つ annotation を検出する。

    Args:
        annotations (dict[str, str]): script id ごとの annotation

    Returns:
        list[str]: 疑わしい script id 一覧
    """

    suspicious_script_ids: list[str] = []

    for script_id, annotation in annotations.items():
        normalized_annotation = annotation.replace("?$", "$")
        if SUSPICIOUS_TERMINAL_MARKER_PATTERN.search(normalized_annotation) is not None:
            suspicious_script_ids.append(script_id)

    return suspicious_script_ids


def detect_exact_duplicate_annotations(
    texts: dict[str, dict[str, Any]],
    annotations: dict[str, str],
) -> list[tuple[str, str]]:
    """
    異なる本文に対して同一 annotation が使われている組を検出する。

    Args:
        texts (dict[str, dict[str, Any]]): `text.yaml`
        annotations (dict[str, str]): `annotation.yaml`

    Returns:
        list[tuple[str, str]]: 問題のある script id ペア
    """

    annotation_to_script_ids: dict[str, list[str]] = defaultdict(list)
    suspicious_pairs: list[tuple[str, str]] = []

    for script_id, annotation in annotations.items():
        annotation_to_script_ids[annotation].append(script_id)

    for script_ids in annotation_to_script_ids.values():
        if len(script_ids) < 2:
            continue

        base_script_id = script_ids[0]
        base_surface_signature = get_utterance_signature(texts[base_script_id])

        for compared_script_id in script_ids[1:]:
            compared_surface_signature = get_utterance_signature(
                texts[compared_script_id]
            )
            if base_surface_signature != compared_surface_signature:
                suspicious_pairs.append((base_script_id, compared_script_id))

    return suspicious_pairs


def detect_inconsistent_duplicate_surfaces(
    texts: dict[str, dict[str, Any]],
    annotations: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """
    同じ本文に対して異なる annotation が付いている script 群を検出する。

    Args:
        texts (dict[str, dict[str, Any]]): `text.yaml`
        annotations (dict[str, str]): `annotation.yaml`

    Returns:
        list[tuple[str, list[str]]]: 本文と対応する script id 群
    """

    surface_to_script_ids: dict[str, list[str]] = defaultdict(list)
    suspicious_surfaces: list[tuple[str, list[str]]] = []

    for script_id, text_item in texts.items():
        surface_to_script_ids[get_utterance_signature(text_item)].append(script_id)

    for surface, script_ids in surface_to_script_ids.items():
        if len(script_ids) < 2:
            continue

        annotation_set = {annotations[script_id] for script_id in script_ids}
        if len(annotation_set) > 1:
            suspicious_surfaces.append((surface, script_ids))

    return suspicious_surfaces


def get_utterance_signature(text_item: dict[str, Any]) -> str:
    """
    annotation と対応づける実発話ベースの本文シグネチャを返す。

    同一テキストでも読みが異なる場合（例: 「苗代」→ なわしろ / いなしろ）は
    異なるシグネチャとなるよう、annotation からプロソディ記号を除去した読みも
    シグネチャに含める。

    Args:
        text_item (dict[str, Any]): corpus エントリの各項目

    Returns:
        str: duplicate 判定用のシグネチャ
    """

    text_field = str(text_item.get("text", "")).strip()
    annotation = str(text_item.get("annotation", "")).strip()
    # annotation からプロソディ記号を除去し、純粋な読み (カタカナ) を取得する
    pronunciation = strip_jsut_annotation_symbols(annotation) if len(annotation) > 0 else ""

    return f"{text_field}\t{pronunciation}"


def main(argv: list[str] | None = None) -> int:
    """
    CLI のエントリーポイント。

    Args:
        argv (list[str] | None): コマンドライン引数

    Returns:
        int: 終了コード
    """

    parser = get_parser()
    args = parser.parse_args(argv)

    corpus = load_yaml(args.corpus_dir / "corpus.yaml")
    texts = {
        script_id: entry
        for script_id, entry in corpus.items()
    }
    annotations = {
        script_id: str(entry["annotation"])
        for script_id, entry in corpus.items()
    }

    invalid_script_ids = validate_parseable_annotations(
        annotations,
        args.accent_status_seq_level,
        cast(AccentRepresentMode, args.accent_status_represent_mode),
    )
    structurally_invalid_annotations = validate_structural_annotations(annotations)
    suspicious_terminal_script_ids = detect_suspicious_terminal_markers(annotations)
    duplicate_annotation_pairs = detect_exact_duplicate_annotations(texts, annotations)
    inconsistent_duplicate_surfaces = detect_inconsistent_duplicate_surfaces(
        texts,
        annotations,
    )

    if len(invalid_script_ids) > 0:
        print(
            "ERROR Invalid annotations were found. "
            f"count: {len(invalid_script_ids)}, examples: {invalid_script_ids[:5]}"
        )

    if len(structurally_invalid_annotations) > 0:
        print(
            "ERROR Structurally invalid annotations were found. "
            f"count: {len(structurally_invalid_annotations)}, "
            f"examples: {structurally_invalid_annotations[:5]}"
        )

    if len(duplicate_annotation_pairs) > 0:
        print(
            "WARN Exact duplicate annotations across different surfaces were found. "
            f"count: {len(duplicate_annotation_pairs)}, "
            f"examples: {duplicate_annotation_pairs[:5]}"
        )

    if len(inconsistent_duplicate_surfaces) > 0:
        print(
            "WARN Inconsistent annotations across duplicate surfaces were found. "
            f"count: {len(inconsistent_duplicate_surfaces)}, "
            f"examples: {inconsistent_duplicate_surfaces[:5]}"
        )

    if len(suspicious_terminal_script_ids) > 0:
        print(
            "WARN Suspicious terminal markers were found. "
            f"count: {len(suspicious_terminal_script_ids)}, "
            f"examples: {suspicious_terminal_script_ids[:5]}"
        )

    if (
        len(invalid_script_ids) == 0
        and len(structurally_invalid_annotations) == 0
        and len(suspicious_terminal_script_ids) == 0
        and len(duplicate_annotation_pairs) == 0
        and len(inconsistent_duplicate_surfaces) == 0
    ):
        print("OK No suspicious annotations were found.")
        return 0

    return 1 if len(invalid_script_ids) > 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
