import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from marine.bin.make_raw_corpus import parse_jsut_annotation
from marine.logger import getLogger
from marine.types import AccentRepresentMode
from marine.utils.g2p_util.util import CANONICAL_MORA_BY_PHONEMES, PHON_TABLE
from marine.utils.jsut_annotation import get_annotation_structure_errors
from marine.utils.jsut_source import reorder_text_entry_fields
from marine.utils.util import normalize_punctuation_characters


logger: logging.Logger | None = None

NATURAL_SORT_PATTERN = re.compile(r"(\d+)")
ROHAN_TEXT_PREFIX_PATTERN = re.compile(r"^(ROHAN4600_\d{4}):(.*)$")


def get_parser() -> argparse.ArgumentParser:
    """
    `vvproj` / `aisp` から marine 用 YAML corpus を生成する CLI の引数を生成する。

    Returns:
        argparse.ArgumentParser: 引数パーサー
    """

    parser = argparse.ArgumentParser(
        description="Convert vvproj or aisp files to marine yaml corpus files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Input directory that contains vvproj or aisp files.",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output directory for text.yaml and annotation.yaml.",
    )
    parser.add_argument(
        "--script-id-prefix",
        type=str,
        default=None,
        help="Prefix for generated script ids. Defaults to the source directory name.",
    )
    parser.add_argument(
        "--script-id-padding",
        type=int,
        default=4,
        help="Zero padding width for generated script ids.",
    )
    parser.add_argument(
        "--accent_status_seq_level",
        "-s",
        type=str,
        choices=["ap", "mora"],
        default="mora",
        help="Sequence level for validation.",
    )
    parser.add_argument(
        "--accent_status_represent_mode",
        "-m",
        type=str,
        choices=["binary", "high_low"],
        default="binary",
        help="Accent representation mode for validation.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level.",
    )
    parser.add_argument(
        "--strip-text-prefix-regex",
        type=str,
        default=None,
        help="Optional regex to strip a synthetic leading prefix from audio item text.",
    )
    return parser


def natural_sort_key(path: Path) -> list[str | int]:
    """
    パス名を自然順ソートするためのキーへ変換する。

    Args:
        path (Path): ソート対象のパス

    Returns:
        list[str | int]: 自然順ソート用のキー
    """

    split_name = NATURAL_SORT_PATTERN.split(path.name)
    natural_key: list[str | int] = []

    for split_item in split_name:
        if split_item.isdigit() is True:
            natural_key.append(int(split_item))
        else:
            natural_key.append(split_item)

    return natural_key


def discover_project_paths(source_dir: Path) -> list[Path]:
    """
    入力ディレクトリから `vvproj` / `aisp` ファイルを自然順で列挙する。

    Args:
        source_dir (Path): 入力ディレクトリ

    Returns:
        list[Path]: 発見したプロジェクトファイル一覧

    Raises:
        FileNotFoundError: 対象ファイルが見つからない場合
    """

    project_paths = sorted(
        [
            *source_dir.glob("*.vvproj"),
            *source_dir.glob("*.aisp"),
        ],
        key=natural_sort_key,
    )

    if len(project_paths) == 0:
        raise FileNotFoundError(
            f"No vvproj or aisp files were found in the directory: {source_dir}"
        )

    return project_paths


def sanitize_surface_text(
    raw_text: str,
    strip_text_prefix_pattern: re.Pattern[str] | None = None,
) -> str:
    """
    `vvproj` の `text` フィールドから学習用本文を抽出する。

    Args:
        raw_text (str): `audioItems[*].text` の生文字列
        strip_text_prefix_pattern (re.Pattern[str] | None): 明示指定された prefix 除去用正規表現

    Returns:
        str: 学習用本文
    """

    stripped_text = raw_text.strip()

    matched_rohan_prefix = ROHAN_TEXT_PREFIX_PATTERN.match(stripped_text)
    if matched_rohan_prefix is not None:
        return normalize_punctuation_characters(matched_rohan_prefix.group(2).strip())

    if strip_text_prefix_pattern is not None:
        matched_prefix = strip_text_prefix_pattern.match(stripped_text)

        if matched_prefix is not None:
            if matched_prefix.lastindex is None or matched_prefix.lastindex < 1:
                raise ValueError(
                    "strip_text_prefix_pattern must contain a capture group for the remaining text."
                )

            return normalize_punctuation_characters(matched_prefix.group(1).strip())

    return normalize_punctuation_characters(stripped_text)


def extract_project_audio_data(
    project_path: Path,
    project_data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    `vvproj` / `aisp` から audioItems と audioKeys を取り出す。

    Args:
        project_path (Path): 読み込み元プロジェクトファイル
        project_data (dict[str, Any]): JSON として読み込んだプロジェクトデータ

    Returns:
        tuple[dict[str, Any], list[str]]: `audioItems` と `audioKeys`

    Raises:
        KeyError: 既知のプロジェクト構造から音声アイテムを取り出せない場合
    """

    if "audioItems" in project_data and "audioKeys" in project_data:
        audio_items = project_data["audioItems"]
        audio_keys = project_data["audioKeys"]
        return dict(audio_items), list(audio_keys)

    talk_data = project_data.get("talk")
    if isinstance(talk_data, dict):
        if "audioItems" in talk_data and "audioKeys" in talk_data:
            audio_items = talk_data["audioItems"]
            audio_keys = talk_data["audioKeys"]
            return dict(audio_items), list(audio_keys)

    raise KeyError(
        f"Failed to find audioItems/audioKeys in project file: {project_path}"
    )


def normalize_mora_vowel(raw_vowel: str) -> str:
    """
    `vvproj` 由来の母音表記を canonical mora 変換用へ正規化する。

    Args:
        raw_vowel (str): `vvproj` に格納された母音

    Returns:
        str: 正規化後の母音
    """

    if raw_vowel in {"N", "cl", "pau"}:
        return raw_vowel

    lowered_vowel = raw_vowel.lower()

    if lowered_vowel == "pau":
        return "pau"

    return lowered_vowel


def get_mora_phonemes(mora: dict[str, Any]) -> tuple[str | None, str]:
    """
    `vvproj` mora から音素列タプルを生成する。

    Args:
        mora (dict[str, Any]): `vvproj` mora

    Returns:
        tuple[str | None, str]: mora の音素列
    """

    consonant = mora.get("consonant")
    vowel = normalize_mora_vowel(str(mora["vowel"]))

    return (str(consonant) if consonant is not None else None, vowel)


def get_canonical_mora_text(mora: dict[str, Any]) -> str:
    """
    `vvproj` の mora 情報を、音素列に整合する canonical カタカナへ変換する。
    VOICEVOX / AivisSpeech 側で rare な外来音が簡約化されている場合でも、
    ここでは surface 表記には寄せず、与えられた phone sequence と矛盾しない
    mora 表記を優先する。

    Args:
        mora (dict[str, Any]): `query.accentPhrases[*].moras[*]` の要素

    Returns:
        str: canonical mora 表記

    Raises:
        ValueError: 対応する mora 表記が見つからない場合
    """

    phonemes = get_mora_phonemes(mora)
    expected_phonemes = tuple(phoneme for phoneme in phonemes if phoneme is not None)
    raw_text = str(mora["text"])
    raw_phonemes = tuple(PHON_TABLE.get(raw_text, []))
    if raw_phonemes == expected_phonemes:
        return raw_text

    canonical_mora = CANONICAL_MORA_BY_PHONEMES.get(expected_phonemes)

    if canonical_mora is not None:
        return canonical_mora

    raise ValueError(
        "Failed to resolve canonical mora text from phonemes. "
        f"mora: {mora}, normalized_phonemes: {phonemes}"
    )


def extract_kana_from_annotation(annotation: str) -> str:
    """
    JSUT 互換 annotation から `kana_level*` 向けカタカナを復元する。

    Args:
        annotation (str): JSUT 互換 annotation

    Returns:
        str: `kana_level*` 向けのカタカナ列
    """

    stripped_annotation = annotation.strip()

    if stripped_annotation.startswith("^") is True:
        stripped_annotation = stripped_annotation[1:]

    if stripped_annotation.endswith("$") is True:
        stripped_annotation = stripped_annotation[:-1]

    for marker in ["[", "]", "#", "?"]:
        stripped_annotation = stripped_annotation.replace(marker, "")

    return stripped_annotation.replace("_", ",")


def build_annotation_from_accent_phrase(accent_phrase: dict[str, Any]) -> str:
    """
    1 つの accent phrase から JSUT 互換のアノテーション断片を復元する。

    Args:
        accent_phrase (dict[str, Any]): `query.accentPhrases[*]` の要素

    Returns:
        str: accent phrase に対応する annotation 断片
    """

    moras = accent_phrase["moras"]
    phrase_annotation_parts: list[str] = []
    accent_position = int(accent_phrase["accent"])

    for mora_index, mora in enumerate(moras):
        if mora_index >= 1 and mora_index == accent_position:
            phrase_annotation_parts.append("]")
        elif mora_index == 1 and accent_position != 1:
            phrase_annotation_parts.append("[")

        phrase_annotation_parts.append(get_canonical_mora_text(mora))

    return "".join(phrase_annotation_parts)


def build_annotation_from_accent_phrases(
    accent_phrases: list[dict[str, Any]],
    surface: str,
) -> str:
    """
    accent phrase 群から JSUT 互換 annotation を構築する。

    Args:
        accent_phrases (list[dict[str, Any]]): `query.accentPhrases`
        surface (str): 対応する本文

    Returns:
        str: JSUT 互換 annotation
    """

    annotation_parts = ["^"]

    for phrase_index, accent_phrase in enumerate(accent_phrases):
        annotation_parts.append(build_annotation_from_accent_phrase(accent_phrase))

        if phrase_index == len(accent_phrases) - 1:
            continue

        if accent_phrase.get("pauseMora") is not None:
            annotation_parts.append("_")
        else:
            annotation_parts.append("#")

    if surface.endswith(("?", "？")) is True:
        annotation_parts.append("?$")
    else:
        annotation_parts.append("$")

    return "".join(annotation_parts)


def build_phone_level3_from_accent_phrases(
    accent_phrases: list[dict[str, Any]],
) -> str:
    """
    accent phrase 群から `phone_level3` 用の音素列を構築する。

    Args:
        accent_phrases (list[dict[str, Any]]): `query.accentPhrases`

    Returns:
        str: `-` 区切りの音素列
    """

    phonemes: list[str] = []

    for accent_phrase in accent_phrases:
        for mora in accent_phrase["moras"]:
            consonant = mora.get("consonant")
            if consonant is not None:
                phonemes.append(str(consonant))

            phonemes.append(str(mora["vowel"]))

        pause_mora = accent_phrase.get("pauseMora")
        if pause_mora is not None:
            phonemes.append(str(pause_mora["vowel"]))

    return "-".join(phonemes)


def validate_annotation(
    script_id: str,
    annotation: str,
    accent_status_seq_level: str,
    accent_status_represent_mode: AccentRepresentMode,
) -> None:
    """
    生成した annotation が既存パイプラインで解釈可能かを検証する。

    Args:
        script_id (str): 対象 script id
        annotation (str): 検証対象 annotation
        accent_status_seq_level (str): 検証時のアクセント系列粒度
        accent_status_represent_mode (str): 検証時のアクセント表現

    Raises:
        ValueError: annotation を既存パイプラインが解釈できない場合
    """

    try:
        parse_jsut_annotation(
            annotation,
            accent_status_seq_level,
            accent_status_represent_mode,
        )
    except Exception as ex:
        raise ValueError(
            f"Failed to validate generated annotation. script_id: {script_id}"
        ) from ex

    structural_errors = get_annotation_structure_errors(annotation)
    if len(structural_errors) > 0:
        raise ValueError(
            "Generated annotation has structural prosody errors. "
            f"script_id: {script_id}, errors: {structural_errors}"
        )


def build_script_id(
    script_id_prefix: str,
    index: int,
    script_id_padding: int,
) -> str:
    """
    出力用 script id を生成する。

    Args:
        script_id_prefix (str): script id の接頭辞
        index (int): 1 始まりの連番
        script_id_padding (int): ゼロ埋め幅

    Returns:
        str: 生成した script id
    """

    return f"{script_id_prefix}_{index:0{script_id_padding}d}"


def build_corpus_entries(
    source_dir: Path,
    script_id_prefix: str,
    script_id_padding: int,
    accent_status_seq_level: str,
    accent_status_represent_mode: AccentRepresentMode,
    strip_text_prefix_pattern: re.Pattern[str] | None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """
    入力ディレクトリから `text.yaml` / `annotation.yaml` の内容を構築する。

    Args:
        source_dir (Path): `vvproj` / `aisp` を含む入力ディレクトリ
        script_id_prefix (str): script id の接頭辞
        script_id_padding (int): ゼロ埋め幅
        accent_status_seq_level (str): annotation 検証時のアクセント系列粒度
        accent_status_represent_mode (str): annotation 検証時のアクセント表現
        strip_text_prefix_pattern (re.Pattern[str] | None): 明示指定された prefix 除去用正規表現

    Returns:
        tuple[dict[str, dict[str, str]], dict[str, str]]: `text.yaml` と `annotation.yaml`
    """

    text_entries: dict[str, dict[str, str]] = {}
    annotations: dict[str, str] = {}
    current_index = 1

    for project_path in discover_project_paths(source_dir):
        with open(project_path, encoding="utf-8") as file:
            project_data = json.load(file)

        audio_items, audio_keys = extract_project_audio_data(project_path, project_data)

        for audio_key in audio_keys:
            audio_item = audio_items[audio_key]
            raw_text = str(audio_item.get("text", "")).strip()

            if raw_text == "":
                continue

            surface = sanitize_surface_text(raw_text, strip_text_prefix_pattern)
            accent_phrases = audio_item["query"]["accentPhrases"]
            if len(accent_phrases) == 0:
                continue

            script_id = build_script_id(
                script_id_prefix=script_id_prefix,
                index=current_index,
                script_id_padding=script_id_padding,
            )
            annotation = build_annotation_from_accent_phrases(accent_phrases, surface)
            validate_annotation(
                script_id=script_id,
                annotation=annotation,
                accent_status_seq_level=accent_status_seq_level,
                accent_status_represent_mode=accent_status_represent_mode,
            )
            kana_text = extract_kana_from_annotation(annotation)
            phone_level3 = build_phone_level3_from_accent_phrases(accent_phrases)

            annotations[script_id] = annotation
            text_entries[script_id] = reorder_text_entry_fields(
                {
                    "text_level0": surface,
                    "kana_level0": kana_text,
                    "text_level1": surface,
                    "text_level2": surface,
                    "kana_level2": kana_text,
                    "kana_level3": kana_text,
                    "phone_level3": phone_level3,
                }
            )
            current_index += 1

    return text_entries, annotations


def write_yaml(path: Path, content: dict[str, Any]) -> None:
    """
    辞書を YAML として保存する。

    Args:
        path (Path): 出力先パス
        content (dict[str, Any]): 保存する辞書
    """

    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            content,
            file,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def entry(argv: list[str] = sys.argv) -> None:
    """
    CLI のエントリーポイント。

    Args:
        argv (list[str]): コマンドライン引数
    """

    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(f"Loaded parameters: {args}")

    if args.source_dir.exists() is False:
        raise FileNotFoundError(f"Source directory was not found: {args.source_dir}")

    script_id_prefix = args.script_id_prefix
    if script_id_prefix is None:
        script_id_prefix = args.source_dir.name

    strip_text_prefix_pattern = None
    if args.strip_text_prefix_regex is not None:
        strip_text_prefix_pattern = re.compile(args.strip_text_prefix_regex)

    text_entries, annotations = build_corpus_entries(
        source_dir=args.source_dir,
        script_id_prefix=script_id_prefix,
        script_id_padding=args.script_id_padding,
        accent_status_seq_level=args.accent_status_seq_level,
        accent_status_represent_mode=cast(
            AccentRepresentMode,
            args.accent_status_represent_mode,
        ),
        strip_text_prefix_pattern=strip_text_prefix_pattern,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(args.out_dir / "text.yaml", text_entries)
    write_yaml(args.out_dir / "annotation.yaml", annotations)

    assert logger is not None
    logger.info(
        f"Converted vvproj corpus. scripts: {len(text_entries)}, output: {args.out_dir}"
    )


if __name__ == "__main__":
    sys.exit(entry())
