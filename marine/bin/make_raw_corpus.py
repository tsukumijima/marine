import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from tqdm import tqdm

from marine.logger import getLogger
from marine.types import AccentRepresentMode
from marine.utils.g2p_util import pron2mora
from marine.utils.openjtalk_util import trans_hyphen2katakana


logger: logging.Logger | None = None


UNUSED_SYMBOL_REMOVER = str.maketrans("", "", "^$[!?")
ACCENT_NUCLEUS_SYMBOL = "]"
ACCENT_PHRASE_BOUNDARY_SYMBOL = "#"
INTONATION_PHRASE_BOUNDARY_SYMBOL = "_"
INTONATION_PHRASE_BOUNDARY_PUNCTUATION = ","


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Special format txt format data to json file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "in_path", default="./data", type=Path, help="Input path or directory"
    )
    parser.add_argument("out_dir", default="./raw", type=Path, help="Output directory")

    parser.add_argument(
        "--corpus_f_name",
        type=str,
        default="corpus.yaml",
        help="Corpus yaml file name ({text, annotation} per entry).",
    )
    parser.add_argument(
        "--accent_status_seq_level",
        "-s",
        type=str,
        choices=["ap", "mora"],
        default="mora",
        help="Sequence level for accent status label",
    )
    parser.add_argument(
        "--accent_status_represent_mode",
        "-m",
        type=str,
        choices=["binary", "high_low"],
        default="binary",
        help="""Representation mode for accent status label
        (this option will be ignored when --accent-status-seq-level is chosen as 'ap')""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level",
    )
    return parser


def alignment_feature(
    features: NDArray[np.str_],
    target_feature: str,
    ignore_features: str,
) -> NDArray[np.bool_]:
    # filtering
    for ignore_feature in ignore_features:
        features = features[features != ignore_feature]

    # detect
    _features = features == target_feature

    # move & pad
    mask = ~(np.concatenate([_features[1:], [False]]))
    _features = _features[mask]

    if target_feature in [
        ACCENT_PHRASE_BOUNDARY_SYMBOL,
        INTONATION_PHRASE_BOUNDARY_SYMBOL,
    ]:
        _features = np.concatenate([[False], _features[:-1]])

    return _features


def convert_mask_seq_to_int_seq(feature: NDArray[np.bool_]) -> NDArray[np.int_]:
    return np.where(feature, 1, 0)


def merge_ip_ap_boundary(
    accent_phrase_boundaries: NDArray[np.int_],
    intonation_phrase_boundaries: NDArray[np.int_],
) -> NDArray[np.int_]:
    assert len(accent_phrase_boundaries) == len(intonation_phrase_boundaries)

    return accent_phrase_boundaries + intonation_phrase_boundaries


def binary_accent_to_ap_accent(
    binary_accents: NDArray[np.int_],
    accent_phrase_boundaries: NDArray[np.int_],
    accent_label: int = 1,
    accent_phrase_label: int = 1,
) -> list[int]:
    assert len(binary_accents) == len(accent_phrase_boundaries)

    accent_phrase_boundary_indexes = np.where(
        accent_phrase_boundaries == accent_phrase_label
    )[0]
    splitted_binary_accents = np.split(binary_accents, accent_phrase_boundary_indexes)

    ap_accent_labbels = []

    for binary_accent in splitted_binary_accents:
        accent_indexs = np.where(binary_accent == accent_label)[0]

        if len(accent_indexs) >= 1:
            acc = accent_indexs[0] + 1  # zero pad
        else:
            acc = 0

        ap_accent_labbels.append(acc)

    return ap_accent_labbels


def binary_accent_to_high_low_accent(
    moras: NDArray[np.str_],
    binary_accents: NDArray[np.int_],
    accent_phrase_boundaries: NDArray[np.int_],
    accent_label: int = 1,
    accent_phrase_label: int = 1,
    accent_status_represent_mode: AccentRepresentMode = "high_low",
) -> list[int]:
    assert len(moras) == len(binary_accents) == len(accent_phrase_boundaries)

    accent_phrase_boundary_indexes = np.where(
        accent_phrase_boundaries == accent_phrase_label
    )[0]
    splitted_moras = np.split(moras, accent_phrase_boundary_indexes)
    splitted_binary_accents = np.split(binary_accents, accent_phrase_boundary_indexes)

    high_low_accent_labels = []

    for mora, binary_accent in zip(splitted_moras, splitted_binary_accents):
        accent_indexs = np.where(binary_accent == accent_label)[0]

        if len(accent_indexs) >= 1:
            _acc = accent_indexs[0] + 1  # zero pad
            result = pron2mora(mora, int(_acc), accent_status_represent_mode)
            assert isinstance(result, tuple)
            _, acc = result
            if (
                accent_status_represent_mode == "high_low"
                and int(_acc) < len(mora)
                and mora[int(_acc)] == "ー"
            ):
                acc[int(_acc)] = 0
        else:
            result = pron2mora(mora, 0, accent_status_represent_mode)
            _, acc = result

        high_low_accent_labels += acc

    return high_low_accent_labels


def convert_to_srt_feature(
    features: NDArray[Any] | list[int] | list[str],
    splitter: str = ",",
) -> str:
    if isinstance(features, np.ndarray):
        features = features.tolist()

    return splitter.join([str(value) for value in features])


def parse_jsut_annotation(
    annotation: str,
    accent_status_seq_level: str,
    accent_status_represent_mode: AccentRepresentMode,
) -> dict[str, str]:
    features: dict[str, str] = {}

    # preprocessing: remove unused symbols
    annotation = annotation.translate(UNUSED_SYMBOL_REMOVER)
    # preprocessing: replace ヲ -> オ
    annotation = annotation.replace("ヲ", "オ")

    annotation = trans_hyphen2katakana(annotation)
    # preprocessing: parse as sequence
    mora_based_annotation = np.array(cast(list[str], pron2mora(annotation)))

    # filtering symbol
    moras = mora_based_annotation[
        (mora_based_annotation != ACCENT_NUCLEUS_SYMBOL)
        & (mora_based_annotation != ACCENT_PHRASE_BOUNDARY_SYMBOL)
        & (mora_based_annotation != INTONATION_PHRASE_BOUNDARY_SYMBOL)
    ]

    assert len(moras) > 0, "Empty annotation"

    binary_accents = alignment_feature(
        mora_based_annotation,
        ACCENT_NUCLEUS_SYMBOL,
        ignore_features=ACCENT_PHRASE_BOUNDARY_SYMBOL
        + INTONATION_PHRASE_BOUNDARY_SYMBOL,
    )
    accent_phrase_boundaries = alignment_feature(
        mora_based_annotation,
        ACCENT_PHRASE_BOUNDARY_SYMBOL,
        ignore_features=ACCENT_NUCLEUS_SYMBOL + INTONATION_PHRASE_BOUNDARY_SYMBOL,
    )
    intonation_phrase_boundaries = alignment_feature(
        mora_based_annotation,
        INTONATION_PHRASE_BOUNDARY_SYMBOL,
        ignore_features=ACCENT_NUCLEUS_SYMBOL + ACCENT_PHRASE_BOUNDARY_SYMBOL,
    )

    assert (
        len(moras)
        == len(binary_accents)
        == len(accent_phrase_boundaries)
        == len(intonation_phrase_boundaries)
    ), (
        f"{len(moras)} != {len(binary_accents)} != {len(accent_phrase_boundaries)} != {len(intonation_phrase_boundaries)},{annotation}"
    )

    accents = convert_mask_seq_to_int_seq(binary_accents)
    accent_phrase_boundaries = convert_mask_seq_to_int_seq(accent_phrase_boundaries)
    intonation_phrase_boundaries = convert_mask_seq_to_int_seq(
        intonation_phrase_boundaries
    )

    accent_phrase_boundaries = merge_ip_ap_boundary(
        accent_phrase_boundaries, intonation_phrase_boundaries
    )

    if accent_status_seq_level == "ap":
        accents = binary_accent_to_ap_accent(accents, accent_phrase_boundaries)
    elif accent_status_seq_level == "mora" and accent_status_represent_mode != "binary":
        accents = binary_accent_to_high_low_accent(
            moras,
            accents,
            accent_phrase_boundaries,
            accent_status_represent_mode=accent_status_represent_mode,
        )

    features["pron"] = convert_to_srt_feature(moras, splitter="")
    features["accent_status"] = convert_to_srt_feature(accents)
    features["accent_phrase_boundary"] = convert_to_srt_feature(
        accent_phrase_boundaries
    )
    features["intonation_phrase_boundary"] = convert_to_srt_feature(
        intonation_phrase_boundaries
    )

    return features


def load_yaml_corpus(
    yaml_corpus_dir: Path,
    accent_status_seq_level: str,
    accent_status_represent_mode: AccentRepresentMode,
    corpus_file_name: str,
    logger: logging.Logger,
) -> list[dict[str, str]]:
    """
    YAML corpus をロードして raw corpus エントリ群を返す。

    各エントリは `{text, annotation}` の 2 フィールドを持つ。

    Args:
        yaml_corpus_dir (Path): YAML ファイルが格納されたディレクトリ
        accent_status_seq_level (str): アクセント系列粒度
        accent_status_represent_mode (AccentRepresentMode): アクセント表現モード
        corpus_file_name (str): corpus YAML ファイル名
        logger (logging.Logger): ロガー

    Returns:
        list[dict[str, str]]: raw corpus エントリ群
    """

    corpus_yaml_path = yaml_corpus_dir / corpus_file_name
    scripts: list[dict[str, str]] = []

    with open(corpus_yaml_path, encoding="utf-8") as file:
        corpus = yaml.safe_load(file)

    if corpus is None or not isinstance(corpus, dict):
        raise ValueError(
            f"Invalid corpus yaml (expected a dict, got {type(corpus).__name__}): "
            f"{corpus_yaml_path}"
        )

    for script_id in tqdm(corpus.keys(), "Parse annotations"):
        entry = corpus[script_id]
        surface = str(entry["text"])
        annotation = str(entry["annotation"])
        feature = parse_jsut_annotation(
            annotation, accent_status_seq_level, accent_status_represent_mode
        )

        features: dict[str, str] = {}
        features["script_id"] = script_id
        features["surface"] = surface
        features.update(feature)

        scripts.append(features)

    logger.info(f"Loaded {len(scripts)} scripts")

    return scripts


def entry(argv: list[str] = sys.argv) -> None:
    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(f"Loaded parameters: {args}")

    scripts = load_yaml_corpus(
        args.in_path,
        args.accent_status_seq_level,
        args.accent_status_represent_mode,
        args.corpus_f_name,
        logger,
    )

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    with open(args.out_dir / "raw_corpus.json", "w", encoding="utf-8") as file:
        json.dump(scripts, file, ensure_ascii=False, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    sys.exit(entry())
