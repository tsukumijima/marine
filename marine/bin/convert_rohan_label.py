import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from marine.logger import getLogger


logger: logging.Logger | None = None
ROHAN_TEXT_PATTERN = re.compile(r"^(ROHAN4600_\d{4}):(.*)$")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ROHAN label resources to marine yaml corpus files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Path to Zundamon_ROHAN_label repository",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output directory for text.yaml and annotation.yaml",
    )
    parser.add_argument(
        "--prosody_yaml_path",
        type=Path,
        default=None,
        help=(
            "Optional helper yaml path that already stores JSUT-compatible "
            "prosodic labels"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level",
    )
    return parser


def _normalize_script_id(script_id: str) -> str:
    return script_id.upper()


def load_rohan_texts_from_vvproj(source_dir: Path) -> dict[str, str]:
    """
    ROHAN の vvproj 群から script id ごとの本文を抽出する。

    Args:
        source_dir (Path): `Zundamon_ROHAN_label` リポジトリのルートディレクトリ

    Returns:
        dict[str, str]: `ROHAN4600_XXXX` をキーとする本文辞書
    """

    vvproj_dir = source_dir / "vvproj_data"
    if vvproj_dir.exists() is False:
        raise FileNotFoundError(f"vvproj_data directory not found: {vvproj_dir}")

    texts: dict[str, str] = {}

    for vvproj_path in sorted(vvproj_dir.glob("project*.vvproj")):
        with open(vvproj_path, encoding="utf-8") as file:
            vvproj = json.load(file)

        audio_items = cast(dict[str, dict[str, Any]], vvproj["audioItems"])
        audio_keys = cast(list[str], vvproj["audioKeys"])

        for audio_key in audio_keys:
            audio_item = audio_items[audio_key]
            raw_text = str(audio_item.get("text", "")).strip()
            if raw_text == "":
                continue

            match = ROHAN_TEXT_PATTERN.match(raw_text)
            if match is None:
                continue

            script_id = _normalize_script_id(match.group(1))
            surface = match.group(2).strip()

            if script_id in texts and texts[script_id] != surface:
                raise ValueError(
                    f"Duplicated script id with different text: {script_id}"
                )

            texts[script_id] = surface

    return texts


def load_rohan_phone_level3(source_dir: Path) -> dict[str, str]:
    """
    ROHAN の phone level3 相当表記を読み込む。

    Args:
        source_dir (Path): `Zundamon_ROHAN_label` リポジトリのルートディレクトリ

    Returns:
        dict[str, str]: `ROHAN4600_XXXX` をキーとする音素列辞書
    """

    phoneme_csv_path = source_dir / "csv_data" / "phoneme.csv"
    if phoneme_csv_path.exists() is False:
        raise FileNotFoundError(f"phoneme.csv not found: {phoneme_csv_path}")

    phone_level3_map: dict[str, str] = {}
    with open(phoneme_csv_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line == "":
                continue

            script_id, phoneme_text = line.split(",", 1)
            normalized_script_id = _normalize_script_id(script_id)
            phone_level3_map[normalized_script_id] = phoneme_text.replace(" ", "-")

    return phone_level3_map


def load_rohan_texts_from_helper_yaml(prosody_yaml_path: Path) -> dict[str, str]:
    """
    Helper YAML から ROHAN 本文を抽出する。

    Args:
        prosody_yaml_path (Path): JSUT 互換の prosody helper YAML パス

    Returns:
        dict[str, str]: `ROHAN4600_XXXX` をキーとする本文辞書
    """

    if prosody_yaml_path.exists() is False:
        raise FileNotFoundError(f"Prosody yaml not found: {prosody_yaml_path}")

    with open(prosody_yaml_path, encoding="utf-8") as file:
        helper_yaml = cast(dict[str, dict[str, Any]], yaml.safe_load(file))

    texts: dict[str, str] = {}
    for helper_key, helper_value in helper_yaml.items():
        script_id = _normalize_script_id(
            helper_key.replace("zundamon-normal_", "").replace(
                "rohan4600_", "ROHAN4600_"
            )
        )
        texts[script_id] = str(helper_value["text"])

    return texts


def normalize_helper_prosodic_label(surface: str, prosodic_label: str) -> str:
    """
    Helper YAML の prosodic_label を JSUT 互換形式へ正規化する。

    Args:
        surface (str): 対応する本文
        prosodic_label (str): helper YAML に記録された prosodic_label

    Returns:
        str: `annotation.yaml` に書き込める正規化済みラベル
    """

    normalized = prosodic_label.strip()

    if normalized.startswith("^") is True:
        normalized = normalized[1:]
    if normalized.endswith("$") is True:
        normalized = normalized[:-1]

    normalized = normalized.replace("、", "_")
    normalized = normalized.replace("，", "_")
    normalized = normalized.replace("。", "")
    normalized = normalized.replace("．", "")
    normalized = normalized.replace("?", "")
    normalized = normalized.replace("？", "")

    suffix = "?$" if surface.endswith(("?", "？")) is True else "$"
    return f"^{normalized}{suffix}"


def load_rohan_annotations_from_helper_yaml(
    texts: dict[str, str],
    prosody_yaml_path: Path,
) -> dict[str, str]:
    """
    Helper YAML から ROHAN 用 annotation を構築する。

    Args:
        texts (dict[str, str]): script id ごとの本文辞書
        prosody_yaml_path (Path): JSUT 互換の prosody helper YAML パス

    Returns:
        dict[str, str]: `annotation.yaml` 相当の辞書
    """

    if prosody_yaml_path.exists() is False:
        raise FileNotFoundError(f"Prosody yaml not found: {prosody_yaml_path}")

    with open(prosody_yaml_path, encoding="utf-8") as file:
        helper_yaml = cast(dict[str, dict[str, Any]], yaml.safe_load(file))

    annotations: dict[str, str] = {}
    assert logger is not None

    for helper_key, helper_value in helper_yaml.items():
        script_id = _normalize_script_id(
            helper_key.replace("zundamon-normal_", "").replace(
                "rohan4600_", "ROHAN4600_"
            )
        )
        if script_id not in texts:
            continue

        helper_surface = str(helper_value["text"])
        if helper_surface != texts[script_id]:
            logger.warning(
                f"Text mismatch was resolved by preferring helper yaml text. "
                f"script_id: {script_id}, helper: {helper_surface}, vvproj: {texts[script_id]}"
            )
            texts[script_id] = helper_surface

        annotations[script_id] = normalize_helper_prosodic_label(
            texts[script_id],
            str(helper_value["prosodic_label"]),
        )

    missing_script_ids = sorted(set(texts.keys()) - set(annotations.keys()))
    if len(missing_script_ids) > 0:
        raise ValueError(
            f"Missing helper prosody labels: {missing_script_ids[:5]} "
            f"(total: {len(missing_script_ids)})"
        )

    return annotations


def extract_kana_from_annotation(annotation: str) -> str:
    """
    Annotation から kana 表記を復元する。

    Args:
        annotation (str): JSUT 互換 annotation

    Returns:
        str: `kana_level*` に格納するカナ表記
    """

    stripped = annotation.strip()
    if stripped.startswith("^") is True:
        stripped = stripped[1:]
    if stripped.endswith("$") is True:
        stripped = stripped[:-1]

    for marker in ["[", "]", "#", "?"]:
        stripped = stripped.replace(marker, "")

    stripped = stripped.replace("_", "、")
    return stripped


def build_text_entries(
    texts: dict[str, str],
    annotations: dict[str, str],
    phone_level3_map: dict[str, str],
) -> dict[str, dict[str, str]]:
    """
    ROHAN corpus の `text.yaml` エントリを構築する。

    Args:
        texts (dict[str, str]): script id ごとの本文辞書
        annotations (dict[str, str]): script id ごとの annotation 辞書
        phone_level3_map (dict[str, str]): script id ごとの音素列辞書

    Returns:
        dict[str, dict[str, str]]: `text.yaml` の内容
    """

    text_entries: dict[str, dict[str, str]] = {}

    for script_id in sorted(texts.keys()):
        surface = texts[script_id]
        annotation = annotations[script_id]
        kana = extract_kana_from_annotation(annotation)
        phone_level3 = phone_level3_map.get(script_id, "")

        text_entries[script_id] = {
            "text_level0": surface,
            "kana_level0": kana,
            "text_level1": surface,
            "text_level2": surface,
            "kana_level2": kana,
            "kana_level3": kana,
            "phone_level3": phone_level3,
        }

    return text_entries


def entry(argv: list[str] = sys.argv) -> None:
    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(f"Loaded parameters: {args}")
    assert logger is not None

    texts = load_rohan_texts_from_vvproj(args.source_dir)
    phone_level3_map = load_rohan_phone_level3(args.source_dir)

    if args.prosody_yaml_path is None:
        raise ValueError(
            "prosody_yaml_path is required for now because exact JSUT-compatible "
            "prosodic labels cannot be recovered losslessly from raw vvproj labels"
        )

    if len(texts) != 4600:
        helper_texts = load_rohan_texts_from_helper_yaml(args.prosody_yaml_path)
        missing_script_ids = sorted(set(helper_texts.keys()) - set(texts.keys()))

        for script_id in missing_script_ids:
            texts[script_id] = helper_texts[script_id]

        logger.warning(
            f"ROHAN texts were supplemented from helper yaml. "
            f"vvproj_count: {len(texts) - len(missing_script_ids)}, "
            f"supplemented_count: {len(missing_script_ids)}, total: {len(texts)}"
        )

    if len(texts) != 4600:
        raise ValueError(f"Unexpected ROHAN script count: {len(texts)}")

    annotations = load_rohan_annotations_from_helper_yaml(
        texts,
        args.prosody_yaml_path,
    )
    text_entries = build_text_entries(texts, annotations, phone_level3_map)

    if args.out_dir.exists() is False:
        args.out_dir.mkdir(parents=True)

    text_yaml_path = args.out_dir / "text.yaml"
    annotation_yaml_path = args.out_dir / "annotation.yaml"

    with open(text_yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            text_entries,
            file,
            allow_unicode=True,
            sort_keys=True,
            default_flow_style=False,
        )

    with open(annotation_yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            annotations,
            file,
            allow_unicode=True,
            sort_keys=True,
            default_flow_style=False,
        )

    logger.info(
        f"Converted ROHAN labels. scripts: {len(text_entries)}, output: {args.out_dir}"
    )


if __name__ == "__main__":
    sys.exit(entry())
