import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, cast

import numpy as np
from joblib import dump, load
from tqdm import tqdm

from marine.data.feature.feature_set import FeatureSet
from marine.logger import getLogger
from marine.types import BatchFeature, MarineFeature
from marine.utils.g2p_util import pron2mora
from marine.utils.nbest_reading import get_nbest_pronunciation_candidates
from marine.utils.util import load_json_corpus, split_corpus


logger: logging.Logger | None = None

AP_BOUNDARY_LABEL = 1
SAFE_MORA_SUBSTITUTIONS: set[tuple[str, str]] = {
    ("ア", "ー"),
    ("イ", "ー"),
    ("ウ", "ー"),
    ("エ", "ー"),
    ("オ", "ー"),
    ("ー", "ア"),
    ("ー", "イ"),
    ("ー", "ウ"),
    ("ー", "エ"),
    ("ー", "オ"),
    ("ヲ", "オ"),
    ("オ", "ヲ"),
    ("ヅ", "ズ"),
    ("ズ", "ヅ"),
    ("ヂ", "ジ"),
    ("ジ", "ヂ"),
    ("ワ", "ハ"),
    ("ハ", "ワ"),
}
KANJI_SURFACE_PATTERN = re.compile(r"^[一-龯々]+$")

VOICE_NORMALIZATION_TABLE = str.maketrans(
    {
        "ガ": "カ",
        "ギ": "キ",
        "グ": "ク",
        "ゲ": "ケ",
        "ゴ": "コ",
        "ザ": "サ",
        "ジ": "シ",
        "ズ": "ス",
        "ゼ": "セ",
        "ゾ": "ソ",
        "ダ": "タ",
        "ヂ": "チ",
        "ヅ": "ツ",
        "デ": "テ",
        "ド": "ト",
        "バ": "ハ",
        "ビ": "ヒ",
        "ブ": "フ",
        "ベ": "ヘ",
        "ボ": "ホ",
        "パ": "ハ",
        "ピ": "ヒ",
        "プ": "フ",
        "ペ": "ヘ",
        "ポ": "ホ",
        "ヴ": "ウ",
    }
)
MAX_ALIGNMENT_NODE_SPAN = 2

SURFACE_PRONUNCIATION_VARIANTS: dict[str, set[str]] = {
    # 「いう / ゆう」は話し言葉の揺れとして非常に頻出する。
    "いう": {"イウ", "ユウ", "ユー"},
    "言う": {"イウ", "ユウ", "ユー"},
    "云う": {"イウ", "ユウ", "ユー"},
    "言うまでもない": {"イウマデモナイ", "ユウマデモナイ", "ユーマデモナイ"},
    # ここから下は、同じ表層・同じ形態素境界で頻出する代表的な読み分かれのみを許容する。
    "う": {"ウ", "オ"},
    "一": {"イチ", "イッ"},
    "十": {"ジュウ", "ジュッ"},
    "何": {"ナニ", "ナン"},
    "人": {"ニン", "ジン"},
    "分": {"プン", "ブン"},
    "前": {"マエ", "ゼン"},
    "後": {"アト", "ノチ"},
    "大": {"ダイ", "オオ", "オー"},
    "御": {"ゴ", "オ"},
    "方": {"ホオ", "ホー", "カタ"},
    "良い": {"ヨイ", "イイ", "イー"},
    "よい": {"ヨイ", "イイ", "イー"},
    "家": {"カ", "ケ"},
    "行く": {"イク", "ユク"},
    "著作権": {"チョサクケン", "チョサッケン"},
    "出生": {"シュッショオ", "シュッショー", "シュッセエ", "シュッセー"},
}
LabelArray = np.ndarray[Any, np.dtype[np.uint8]]
PackedCorpusItem = tuple[str, BatchFeature, dict[str, LabelArray]]
PackedCorpusProcessResult = tuple[PackedCorpusItem | None, str | None]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Special format txt format data to json file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("corpus_path", type=Path, help="Path or directory for corpus")
    parser.add_argument(
        "feature_path", type=Path, help="Path or directory for feature file"
    )
    parser.add_argument("vocab_path", type=Path, help="Vocab file path")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of jobs")
    parser.add_argument(
        "--feature_table_key",
        "-f",
        type=str,
        choices=["unidic-csj", "open-jtalk"],
        default="open-jtalk",
        help="Sequence level for accent status label",
    )
    parser.add_argument(
        "--max_size",
        "-m",
        type=int,
        default=-1,
        help="""Maximum number of scripts to convert feature
        (-m < 0 = use all scripts)""",
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
        "--test_size",
        "-t",
        type=int,
        default=-1,
        help="""Specific size of samples for val and test
        (-t < 0 = 5%% and / 5%% for val and test respectively)""",
    )
    parser.add_argument(
        "--skip_corpus_split",
        action="store_true",
        help="Whether skip corpus splitting",
    )
    parser.add_argument(
        "--single_corpus_key",
        "-k",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="""Dataset key to export
        (this option will be ignored if --skip_corpus_split is not activated)""",
    )
    parser.add_argument(
        "--target_id_dir",
        type=Path,
        default=None,
        help="Directory of id files to reproduce specific dataset",
    )
    parser.add_argument(
        "--random_seed",
        "-r",
        type=int,
        default=12345,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level",
    )
    return parser


def insert_punctuation_by_extracted_features(
    mora_seq_w_puncts: np.ndarray[Any, Any],
    mora_seq_wo_puncts: np.ndarray[Any, Any],
    labels: dict[str, list[int]],
    punctuation_ids: list[int],
    accent_status_seq_level: str,
) -> dict[str, list[int]]:
    """
    記号を含むモーラ列に合わせて、記号を除いたモーラ列とラベルに記号位置を挿入する。

    Args:
        mora_seq_w_puncts (np.ndarray[Any, Any]): 記号を含むモーラ ID 列
        mora_seq_wo_puncts (np.ndarray[Any, Any]): 記号を除いたモーラ ID 列（in-place で更新）
        labels (dict[str, list[int]]): ラベル名をキー、ラベル列を値とする辞書（in-place で更新）
        punctuation_ids (list[int]): 記号として扱うモーラ ID のリスト
        accent_status_seq_level (str): アクセントラベルのシーケンスレベル（"ap" または "mora"）

    Returns:
        dict[str, list[int]]: 記号位置を挿入したラベル辞書
    """
    indexs = np.where(np.isin(mora_seq_w_puncts, punctuation_ids))[0]

    for index in indexs:
        mora_seq_wo_puncts = np.insert(
            mora_seq_wo_puncts, index, int(mora_seq_w_puncts[index])
        )
        for label_key in labels.keys():
            if label_key == "accent_status" and accent_status_seq_level == "ap":
                # the AP-based AN label represents the position of AN in a AP
                # i.e., Don't need consider the global indexå
                # Also, the accent nucleus never located at punctuation
                continue

            # the label where punctuation are must be 0
            labels[label_key].insert(index, 0)
            assert len(mora_seq_wo_puncts) == len(labels[label_key]), (
                f"Length of inserted result is not matched: {len(mora_seq_wo_puncts)} != {len(labels[label_key])} ({label_key})"
            )

    assert len(mora_seq_w_puncts) == len(mora_seq_wo_puncts), (
        f"The inserted mora seq length is not matched: {len(mora_seq_w_puncts)} != {len(mora_seq_wo_puncts)}"
    )
    assert np.array_equal(
        np.isin(mora_seq_w_puncts, punctuation_ids),
        np.isin(mora_seq_wo_puncts, punctuation_ids),
    ), "The inserted mora seq does not preserve punctuation positions"

    return labels


def is_softmatch_mora_sequence(
    extracted_moras: list[str],
    expected_moras: list[str],
) -> bool:
    """
    長さを変えない安全な表記揺れだけで構成される差分かを判定する。

    Args:
        extracted_moras (list[str]): OpenJTalk から抽出されたモーラ列
        expected_moras (list[str]): アノテーション由来の期待モーラ列

    Returns:
        bool: 長音表記揺れなどの安全な差分のみで一致すると見なせる場合は True
    """

    if len(extracted_moras) != len(expected_moras):
        return False

    if extracted_moras == expected_moras:
        return True

    return all(
        extracted_mora == expected_mora
        or (extracted_mora, expected_mora) in SAFE_MORA_SUBSTITUTIONS
        for extracted_mora, expected_mora in zip(extracted_moras, expected_moras)
    )


def _normalize_vowel_sequences_for_alignment(text: str) -> str:
    """
    アライメント比較用に、連続母音を長音記号へ正規化する。

    Args:
        text (str): カタカナ発音列

    Returns:
        str: 長音表記揺れを正規化したカタカナ発音列
    """

    if "ー" in text:
        return text

    for vowel in ["ア", "イ", "ウ", "エ", "オ"]:
        text = re.sub(
            f"(?<!{vowel})({vowel})({vowel})({vowel})(?!{vowel})",
            "\\1ー\\3",
            text,
        )

    if "ー" in text:
        return text

    for vowel in ["ア", "イ", "ウ", "エ", "オ"]:
        text = re.sub(f"({vowel})({vowel})(?!{vowel})", r"\1ー", text)

    return text


def _normalize_pronunciation_for_alignment(text: str) -> str:
    """
    話し言葉の代表的な長音揺れを吸収しやすい形へ発音列を正規化する。

    Args:
        text (str): カタカナ発音列

    Returns:
        str: 長音・四つ仮名の揺れを吸収した比較用の発音列
    """

    normalized_text = text
    normalized_text = re.sub(
        r"([カガサザタダナハバパマヤラワ])([ア])(?!ア{2})(?![ーイウエオッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([キギシジチヂニヒビピミリ])([イ])(?!イ{2})(?![ーアウエオッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([クグスズツヅヌフブプムユル])([ウ])(?!ウ{2})(?![ーィェォアイエオッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([エケゲセゼテデネヘベペメレ])([イ])(?![ーアイウオッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([ケゲセゼテデネヘベペメレ])([エ])(?!エ{2})(?![ーアイウオッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"(?<![オ])([オコゴソゾトドノホボポモヨロ])([ウ])(?![ーィェォアイウエッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([コゴソゾトドノホボポモヨロ])([オ])(?!オ{2})(?![ーアイウエッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([キシチニヒミリギジビピ][ャ])([ア])(?!ア{2})(?![ーッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([キシチニヒミリギジビピ][ュ])([ウ])(?!ウ{2})(?![ーィェォッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([キシチニヒミリギジビピ][ョ])([ウ])(?!ウ{2})(?![ーィェォッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = re.sub(
        r"([キシチニヒミリギジビピ][ョ])([オ])(?!オ{2})(?![ーッ])",
        r"\1ー",
        normalized_text,
    )
    normalized_text = _normalize_vowel_sequences_for_alignment(normalized_text)
    normalized_text = (
        normalized_text.replace("ヲ", "オ").replace("ヅ", "ズ").replace("ヂ", "ジ")
    )
    return normalized_text


def _normalize_voicing_for_alignment(text: str) -> str:
    """
    濁点・半濁点の差分を吸収しやすい比較用文字列へ正規化する。

    Args:
        text (str): カタカナ発音列

    Returns:
        str: 濁点差分を除去した比較用の発音列
    """

    return text.translate(VOICE_NORMALIZATION_TABLE)


def _is_softmatch_node_pronunciation(
    surface: str,
    candidate_moras: list[str],
    expected_moras: list[str],
) -> bool:
    """
    単一形態素の発音差分が、安全に吸収できる揺れかを判定する。

    Args:
        surface (str): 形態素の表層形
        candidate_moras (list[str]): surface から導ける候補モーラ列
        expected_moras (list[str]): アノテーション側のモーラ列

    Returns:
        bool: 品詞対応を維持したまま安全に吸収できる揺れなら True
    """

    if candidate_moras == expected_moras:
        return True

    candidate_text = "".join(candidate_moras)
    expected_text = "".join(expected_moras)
    allowed_variants = SURFACE_PRONUNCIATION_VARIANTS.get(surface)
    if (
        allowed_variants is not None
        and candidate_text in allowed_variants
        and expected_text in allowed_variants
    ):
        return True

    if len(candidate_moras) == len(expected_moras) and is_softmatch_mora_sequence(
        candidate_moras,
        expected_moras,
    ):
        return True

    normalized_candidate_text = _normalize_pronunciation_for_alignment(candidate_text)
    normalized_expected_text = _normalize_pronunciation_for_alignment(expected_text)
    if normalized_candidate_text == normalized_expected_text:
        return True

    if _normalize_voicing_for_alignment(
        normalized_candidate_text
    ) == _normalize_voicing_for_alignment(normalized_expected_text):
        return True

    for candidate_pronunciation in get_nbest_pronunciation_candidates(surface):
        if (
            candidate_pronunciation == expected_text
            or _normalize_pronunciation_for_alignment(candidate_pronunciation)
            == normalized_expected_text
            or _normalize_voicing_for_alignment(
                _normalize_pronunciation_for_alignment(candidate_pronunciation)
            )
            == _normalize_voicing_for_alignment(normalized_expected_text)
        ):
            return True

    return False


def _can_merge_alignment_nodes(surface_parts: list[str]) -> bool:
    """
    n-best 補助探索のために隣接 node を結合してよいかを判定する。

    Args:
        surface_parts (list[str]): 結合候補の surface 一覧

    Returns:
        bool: 漢字複合語として結合探索する価値がある場合は True
    """

    return all(
        KANJI_SURFACE_PATTERN.fullmatch(surface_part) is not None
        for surface_part in surface_parts
    )


def _get_surface_candidate_moras(
    surface: str, extracted_pronunciation: str
) -> tuple[list[str], ...]:
    """
    単一 surface に対して照合に使う pronunciation 候補のモーラ列を返す。

    Args:
        surface (str): 形態素の表層形
        extracted_pronunciation (str): OpenJTalk が返した pronunciation

    Returns:
        tuple[list[str], ...]: 重複を除いた pronunciation 候補のモーラ列タプル
    """

    candidate_pronunciations: list[str] = [extracted_pronunciation]
    allowed_variants = SURFACE_PRONUNCIATION_VARIANTS.get(surface)
    if allowed_variants is not None:
        candidate_pronunciations.extend(sorted(allowed_variants))
    candidate_pronunciations.extend(get_nbest_pronunciation_candidates(surface))

    unique_candidate_moras: list[list[str]] = []
    seen_candidate_texts: set[str] = set()
    for candidate_pronunciation in candidate_pronunciations:
        if candidate_pronunciation in seen_candidate_texts:
            continue
        candidate_moras = cast(list[str], pron2mora(candidate_pronunciation))
        candidate_text = "".join(candidate_moras)
        if candidate_text in seen_candidate_texts:
            continue
        seen_candidate_texts.add(candidate_pronunciation)
        seen_candidate_texts.add(candidate_text)
        unique_candidate_moras.append(candidate_moras)

    return tuple(unique_candidate_moras)


def _count_mora_mismatches(left_moras: list[str], right_moras: list[str]) -> int:
    """
    同長モーラ列どうしの差分数を数える。

    Args:
        left_moras (list[str]): 比較元モーラ列
        right_moras (list[str]): 比較先モーラ列

    Returns:
        int: 差分モーラ数。長さが異なる場合は大きな値を返す。
    """

    if len(left_moras) != len(right_moras):
        return max(len(left_moras), len(right_moras))

    return sum(
        left_mora != right_mora
        for left_mora, right_mora in zip(left_moras, right_moras)
    )


def _is_close_dataset_typo_mora_sequence(
    candidate_moras: list[str],
    expected_moras: list[str],
) -> bool:
    """
    dataset 側 typo と見なせる近距離差分かを判定する。

    Args:
        candidate_moras (list[str]): surface 候補由来のモーラ列
        expected_moras (list[str]): dataset 側のモーラ列

    Returns:
        bool: 1 モーラ程度の近距離 typo と見なせる場合は True
    """

    if len(candidate_moras) != len(expected_moras):
        return False

    if _count_mora_mismatches(candidate_moras, expected_moras) > 1:
        return False

    normalized_candidate_moras = cast(
        list[str],
        pron2mora(_normalize_pronunciation_for_alignment("".join(candidate_moras))),
    )
    normalized_expected_moras = cast(
        list[str],
        pron2mora(_normalize_pronunciation_for_alignment("".join(expected_moras))),
    )
    if (
        _count_mora_mismatches(normalized_candidate_moras, normalized_expected_moras)
        <= 1
    ):
        return True

    devoiced_candidate_moras = cast(
        list[str],
        pron2mora(_normalize_voicing_for_alignment("".join(candidate_moras))),
    )
    devoiced_expected_moras = cast(
        list[str],
        pron2mora(_normalize_voicing_for_alignment("".join(expected_moras))),
    )
    return (
        _count_mora_mismatches(devoiced_candidate_moras, devoiced_expected_moras) <= 1
    )


def _is_supported_by_surface_candidates(
    surface: str,
    extracted_pronunciation: str,
    target_moras: list[str],
) -> bool:
    """
    target の読みが surface 候補として説明可能かを判定する。

    Args:
        surface (str): 形態素の表層形
        extracted_pronunciation (str): OpenJTalk が返した pronunciation
        target_moras (list[str]): 判定対象のモーラ列

    Returns:
        bool: surface から導ける候補として説明できる場合は True
    """

    for candidate_moras in _get_surface_candidate_moras(
        surface, extracted_pronunciation
    ):
        if (
            _is_softmatch_node_pronunciation(
                surface,
                candidate_moras,
                target_moras,
            )
            is True
        ):
            return True

    return False


def _is_soft_equivalent_mora(left_mora: str, right_mora: str) -> bool:
    """
    2 つのモーラが安全な揺れとして同値かを判定する。

    Args:
        left_mora (str): 比較元モーラ
        right_mora (str): 比較先モーラ

    Returns:
        bool: 長音・四つ仮名・濁点差分まで含めて同値と見なせる場合は True
    """

    if left_mora == right_mora:
        return True

    if (left_mora, right_mora) in SAFE_MORA_SUBSTITUTIONS:
        return True

    normalized_left = _normalize_pronunciation_for_alignment(left_mora)
    normalized_right = _normalize_pronunciation_for_alignment(right_mora)
    if normalized_left == normalized_right:
        return True

    return _normalize_voicing_for_alignment(
        normalized_left
    ) == _normalize_voicing_for_alignment(normalized_right)


def _remap_labels_by_mora_alignment(
    extracted_moras: list[str],
    expected_moras: list[str],
    original_labels: dict[str, list[int]],
) -> dict[str, list[int]] | None:
    """
    モーラ列の編集距離に基づき、dataset 側ラベルを extracted 側へ安全に写像する。

    Args:
        extracted_moras (list[str]): OpenJTalk 由来のモーラ列
        expected_moras (list[str]): dataset 側モーラ列
        original_labels (dict[str, list[int]]): dataset 側ラベル列

    Returns:
        dict[str, list[int]] | None: 安全に写像できる場合は remap 後のラベル列。危険な差分が大きい場合は None。
    """

    extracted_len = len(extracted_moras)
    expected_len = len(expected_moras)
    if extracted_len == 0 and expected_len == 0:
        return {key: [] for key in original_labels.keys()}

    if any(
        len(label_values) != expected_len for label_values in original_labels.values()
    ):
        return None

    dp: list[list[int]] = [[0] * (expected_len + 1) for _ in range(extracted_len + 1)]
    backtrack: list[list[tuple[int, int, str] | None]] = [
        [None] * (expected_len + 1) for _ in range(extracted_len + 1)
    ]

    for extracted_index in range(1, extracted_len + 1):
        dp[extracted_index][0] = extracted_index
        backtrack[extracted_index][0] = (extracted_index - 1, 0, "insert")
    for expected_index in range(1, expected_len + 1):
        dp[0][expected_index] = expected_index
        backtrack[0][expected_index] = (0, expected_index - 1, "delete")

    for extracted_index in range(1, extracted_len + 1):
        for expected_index in range(1, expected_len + 1):
            substitution_cost = (
                0
                if _is_soft_equivalent_mora(
                    extracted_moras[extracted_index - 1],
                    expected_moras[expected_index - 1],
                )
                is True
                else 1
            )
            candidates = [
                (
                    dp[extracted_index - 1][expected_index - 1] + substitution_cost,
                    extracted_index - 1,
                    expected_index - 1,
                    "match" if substitution_cost == 0 else "substitute",
                ),
                (
                    dp[extracted_index - 1][expected_index] + 1,
                    extracted_index - 1,
                    expected_index,
                    "insert",
                ),
                (
                    dp[extracted_index][expected_index - 1] + 1,
                    extracted_index,
                    expected_index - 1,
                    "delete",
                ),
            ]
            best_cost, prev_extracted_index, prev_expected_index, operation = min(
                candidates,
                key=lambda candidate: candidate[0],
            )
            dp[extracted_index][expected_index] = best_cost
            backtrack[extracted_index][expected_index] = (
                prev_extracted_index,
                prev_expected_index,
                operation,
            )

    max_allowed_cost = max(2, extracted_len // 12)
    if dp[extracted_len][expected_len] > max_allowed_cost:
        return None

    extracted_index = extracted_len
    expected_index = expected_len
    aligned_pairs: list[tuple[int, int | None, str]] = []
    max_contiguous_unmatched = 0
    current_unmatched = 0
    while extracted_index > 0 or expected_index > 0:
        previous_step = backtrack[extracted_index][expected_index]
        if previous_step is None:
            return None

        previous_extracted_index, previous_expected_index, operation = previous_step
        if operation in {"match", "substitute"}:
            aligned_pairs.append(
                (extracted_index - 1, expected_index - 1, operation),
            )
        elif operation == "insert":
            aligned_pairs.append((extracted_index - 1, None, operation))

        if operation == "match":
            current_unmatched = 0
        else:
            current_unmatched += 1
            max_contiguous_unmatched = max(
                max_contiguous_unmatched,
                current_unmatched,
            )

        extracted_index = previous_extracted_index
        expected_index = previous_expected_index

    if max_contiguous_unmatched > 3:
        return None

    aligned_pairs.reverse()

    remapped_labels: dict[str, list[int]] = {key: [] for key in original_labels.keys()}
    for _, expected_index, operation in aligned_pairs:
        for label_key, label_values in original_labels.items():
            if expected_index is None:
                remapped_labels[label_key].append(0)
                continue
            remapped_labels[label_key].append(label_values[expected_index])

    if any(
        len(remapped_labels[label_key]) != extracted_len
        for label_key in remapped_labels.keys()
    ):
        return None

    return remapped_labels


def is_surface_aligned_mora_sequence(
    nodes: list[MarineFeature],
    expected_moras: list[str],
) -> bool:
    """
    OpenJTalk の形態素境界を保ったまま expected pron を安全に対応付けられるか判定する。

    Args:
        nodes (list[MarineFeature]): OpenJTalk 由来の形態素列
        expected_moras (list[str]): アノテーション由来の期待モーラ列

    Returns:
        bool: すべての形態素で安全な揺れとして吸収できる場合は True
    """

    extracted_node_moras: list[tuple[str, str]] = []
    for node in nodes:
        if node["pron"] is None:
            continue
        extracted_node_moras.append((node["surface"], node["pron"]))

    @cache
    def _search(node_index: int, expected_offset: int) -> bool:
        if node_index == len(extracted_node_moras):
            return expected_offset == len(expected_moras)

        max_span_end = min(
            len(extracted_node_moras),
            node_index + MAX_ALIGNMENT_NODE_SPAN,
        )
        for next_node_index in range(node_index + 1, max_span_end + 1):
            surface_parts = [
                surface
                for surface, _ in extracted_node_moras[node_index:next_node_index]
            ]
            if (
                next_node_index - node_index > 1
                and _can_merge_alignment_nodes(surface_parts) is not True
            ):
                continue

            surface = "".join(surface_parts)
            extracted_pronunciation = "".join(
                pronunciation
                for _, pronunciation in extracted_node_moras[node_index:next_node_index]
            )
            surface_candidate_moras = _get_surface_candidate_moras(
                surface,
                extracted_pronunciation,
            )
            for candidate_moras in surface_candidate_moras:
                next_expected_offset = expected_offset + len(candidate_moras)
                if next_expected_offset > len(expected_moras):
                    continue

                expected_node_moras = expected_moras[
                    expected_offset:next_expected_offset
                ]
                is_candidate_matched = _is_softmatch_node_pronunciation(
                    surface,
                    candidate_moras,
                    expected_node_moras,
                )
                if is_candidate_matched is not True:
                    # dataset 側の読みが surface から見て説明できない一方で、
                    # OpenJTalk / UniDic 側の候補はその surface を自然に説明できる場合は、
                    # 音声未使用学習用の typo / 語彙揺れ補正として候補側を優先する。
                    if len(candidate_moras) != len(expected_node_moras):
                        continue
                    if (
                        _is_supported_by_surface_candidates(
                            surface,
                            extracted_pronunciation,
                            expected_node_moras,
                        )
                        is True
                    ):
                        continue

                if _search(next_node_index, next_expected_offset) is True:
                    return True

        return False

    _search.cache_clear()
    return _search(0, 0)


def process(
    nodes: list[MarineFeature],
    feature_set: FeatureSet,
    accent_status_seq_level: str,
    script_id: str,
    surface: str,
    pron: str,
    **original_labels: str,
) -> PackedCorpusProcessResult:
    """
    形態素列とアノテーションからパッケージ化されたコーパス項目を生成する。

    Args:
        nodes (list[MarineFeature]): OpenJTalk 由来の形態素列
        feature_set (FeatureSet): 特徴量変換用の FeatureSet
        accent_status_seq_level (str): アクセントラベルのシーケンスレベル（"ap" または "mora"）
        script_id (str): スクリプト ID
        surface (str): 表層形
        pron (str): 発音（カンマ区切りラベルを含む）
        **original_labels (str): ラベル名をキー、カンマ区切り値を値とするキーワード引数

    Returns:
        PackedCorpusProcessResult: パッケージ化されたコーパス項目と wrong mora 情報のタプル
    """
    if len(nodes) == 0:
        if logger is not None:
            logger.debug(f"Empty nodes: {script_id}")
        return None, None

    # get punctuation id
    punctuation_ids = feature_set.get_punctuation_ids()

    # check accent seq level
    required_ap_accent = (
        "accent_status" in original_labels.keys() and accent_status_seq_level == "ap"
    )

    # init labels, features
    pending_labels: dict[str, list[int]] = {key: [] for key in original_labels.keys()}
    _original_labels = {
        key: [int(v) for v in value.split(",")]
        for key, value in original_labels.items()
    }

    # convert nodes to feature seqs
    feature = feature_set.convert_nodes_to_feature(nodes)

    # convert original pron in annotation to id seqs

    original_mora = cast(list[str], pron2mora(pron))

    expected_ids = feature_set.convert_feature_to_id("mora", original_mora)

    # verify the features is available
    # i.e., is the prounnounces is same w/o punctuation
    punct_removed_extracted_mora = feature["mora"][
        np.isin(
            feature["mora"], punctuation_ids, invert=True
        )  # 発音しない記号は除外している
    ]
    punct_removed_expected_mora = expected_ids[
        np.isin(expected_ids, punctuation_ids, invert=True)
    ]
    extracted_moras = cast(
        list[str],
        feature_set.convert_id_to_feature(
            "mora", punct_removed_extracted_mora
        ).tolist(),
    )
    expected_moras = cast(
        list[str],
        feature_set.convert_id_to_feature("mora", punct_removed_expected_mora).tolist(),
    )
    expected_txt = "".join(expected_moras)
    extracted_txt = "".join(extracted_moras)

    is_softmatched = is_softmatch_mora_sequence(
        extracted_moras,
        expected_moras,
    )
    is_surface_aligned = is_surface_aligned_mora_sequence(
        nodes,
        expected_moras,
    )
    remapped_labels = None
    should_remap_labels = (
        required_ap_accent is not True and extracted_moras != expected_moras
    )

    if should_remap_labels is True:
        remapped_labels = _remap_labels_by_mora_alignment(
            extracted_moras,
            expected_moras,
            _original_labels,
        )

    # AP-based accent labels are defined per accent phrase, not per mora.
    # Therefore, if the non-punctuation mora length differs, we cannot
    # safely remap the labels with the mora-level alignment heuristic.
    if required_ap_accent is True and len(extracted_moras) != len(expected_moras):
        is_softmatched = False
        is_surface_aligned = False

    if (
        is_softmatched is True
        or is_surface_aligned is True
        or remapped_labels is not None
    ):
        if remapped_labels is not None:
            _original_labels = remapped_labels
            expected_ids = feature_set.convert_feature_to_id("mora", extracted_moras)

        if len(feature["mora"]) != len(expected_ids):
            # 仕様を調べて後で対応
            try:
                _original_labels = insert_punctuation_by_extracted_features(
                    feature["mora"],
                    expected_ids,
                    _original_labels,
                    punctuation_ids,
                    accent_status_seq_level,
                )
            except Exception as ex:
                if logger is not None:
                    logger.debug(
                        f"Failed to insert punctuation labels: {script_id}",
                        exc_info=ex,
                    )
                return None, f"{script_id}|{extracted_txt}|{expected_txt}\n"

        labels: dict[str, LabelArray] = {}
        for key in pending_labels.keys():
            # verify for ap-based AN labels with AP labels
            if (
                required_ap_accent
                and key == "accent_status"
                and "accent_phrase_boundary" in pending_labels.keys()
            ):
                num_boundary = (
                    np.count_nonzero(
                        _original_labels["accent_phrase_boundary"] == AP_BOUNDARY_LABEL
                    )
                    + 1
                )
                accents = _original_labels["accent_status"]
                assert len(accents) == num_boundary, (
                    "Unmatched length of sequnce between ac and ap"
                )
                labels["accent_status"] = np.array(
                    _original_labels["accent_status"], dtype=np.uint8
                )
            else:
                labels[key] = np.array(_original_labels[key], dtype=np.uint8)
    else:
        if logger is not None:
            """
            print('mora from katakana.yaml',''.join(feature_set.convert_id_to_feature('mora', expected_ids)))
            print('mora from mecab',''.join(feature_set.convert_id_to_feature('mora', feature['mora'])))
            print(feature['mora'])
            print(expected_ids)
            """
            logger.debug(f"Wrong mora [{script_id}]:{expected_txt} != {extracted_txt}")
        return None, f"{script_id}|{extracted_txt}|{expected_txt}\n"

    return (script_id, feature, labels), None


def _separate_process_results(
    process_results: list[PackedCorpusProcessResult],
) -> tuple[list[PackedCorpusItem | None], list[str]]:
    """
    ワーカープロセスの結果をコーパス本体と診断情報に分離する。

    Args:
        process_results (list[PackedCorpusProcessResult]): `_process()` が返した結果の一覧

    Returns:
        tuple[list[PackedCorpusItem | None], list[str]]: コーパス項目一覧と、wrong mora 情報の行一覧
    """

    corpus_items: list[PackedCorpusItem | None] = []
    wrong_mora_infos: list[str] = []

    for corpus_item, wrong_mora_info in process_results:
        corpus_items.append(corpus_item)
        if wrong_mora_info is not None:
            wrong_mora_infos.append(wrong_mora_info)

    return corpus_items, wrong_mora_infos


def _load_target_ids(target_id_dir: Path) -> dict[str, set[str]]:
    id_groups: dict[str, set[str]] = {}

    assert logger is not None
    logger.info(f"Load ids from existing dataset ({target_id_dir})")
    for path in target_id_dir.glob("*/ids.pkl"):
        dataset_key = str(path.parent.name)
        id_groups[dataset_key] = set(load(path))
        logger.info(
            f"Loaded {len(id_groups[dataset_key]):,} ids for {dataset_key} in {target_id_dir}"
        )

    return id_groups


def _sort_corpus_by_script_id(corpus: list[PackedCorpusItem]) -> list[PackedCorpusItem]:
    return list(sorted(corpus, key=lambda item: item[0]))


def _remove_unavailable_script(
    corpus: list[PackedCorpusItem | None],
) -> list[PackedCorpusItem]:
    return [item for item in corpus if item is not None]


def _split_corpus_by_ids(
    corpus: list[PackedCorpusItem],
    id_groups: dict[str, set[str]],
) -> dict[str, list[PackedCorpusItem]]:
    _corpus: dict[str, list[PackedCorpusItem]] = {}

    for key, target_ids in id_groups.items():
        scripts = [item for item in corpus if item[0] in target_ids]
        assert len(scripts) == len(target_ids), (
            f"Not enough number of scripts for {key}: {len(scripts):,} != {len(target_ids):,}"
        )
        _corpus[key] = scripts

    # When specified the ID group including in val, test
    if set(["val", "test"]) == set(_corpus.keys()):
        _corpus["train"] = list(
            item
            for item in corpus
            if item[0] not in id_groups["val"] and item[0] not in id_groups["test"]
        )
    # When the ID group is specified fully (i.e., train, val and test)
    elif set(["train", "val", "test"]) == set(_corpus.keys()):
        pass
    else:
        raise NotImplementedError(
            f"Not supported ID group specification: {_corpus.keys()}"
        )

    return _corpus


def entry(argv: list[str] = sys.argv) -> None:
    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(args)

    # Init feature-set
    feature_set = FeatureSet(args.vocab_path, feature_table_key=args.feature_table_key)
    wrong_mora_info_path = str(args.out_dir / "wrong_mora_info.csv")

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)
    Path(wrong_mora_info_path).write_text("", encoding="utf-8")

    # Load corpus
    corpus = load_json_corpus(args.corpus_path)
    features = load_json_corpus(args.feature_path)

    assert len(corpus) == len(features), (
        f"Not match script size between corpus and feature files{len(corpus)} != {len(features)}"
    )
    assert [script["script_id"] for script in corpus] == [
        script["script_id"] for script in features
    ], "Not match script ids between corpus and feature files."

    if args.max_size > 0:
        assert len(corpus) > args.max_size, (
            f"Not enough number of script: {len(corpus)} < {args.max_size}"
        )
        corpus = corpus[: args.max_size]
        features = features[: args.max_size]

    # Process
    n_jobs = min(cpu_count(), args.n_jobs)

    if n_jobs > 1:
        logger.info(f"Processing {len(corpus):,} scripts with {n_jobs} jobs")
        with ProcessPoolExecutor(n_jobs) as executor:
            futures = [
                executor.submit(
                    process,
                    feature["nodes"],
                    feature_set,
                    args.accent_status_seq_level,
                    **script,
                )
                for script, feature in zip(corpus, features)
            ]
            process_results = [
                future.result()
                for future in tqdm(
                    futures, desc="Convert corpus to feature", leave=False
                )
            ]
    else:
        logger.info(f"Processing {len(corpus):,} scripts in a single thread")
        process_results = [
            process(
                feature["nodes"],
                feature_set,
                args.accent_status_seq_level,
                **script,
            )
            for script, feature in tqdm(
                zip(corpus, features), desc="Convert corpus to feature", leave=False
            )
        ]

    corpus, wrong_mora_infos = _separate_process_results(process_results)
    Path(wrong_mora_info_path).write_text("".join(wrong_mora_infos), encoding="utf-8")

    # Cleaing
    corpus = _sort_corpus_by_script_id(_remove_unavailable_script(corpus))
    logger.info(f"Cleaned corpus: {len(corpus):,}")

    if args.target_id_dir is None:
        if args.skip_corpus_split:
            corpus = {key: corpus for key in [args.single_corpus_key]}
        else:
            corpus = split_corpus(
                corpus, random_state=args.random_seed, absolute_test_size=args.test_size
            )
    else:
        id_groups = _load_target_ids(args.target_id_dir)
        corpus = _split_corpus_by_ids(corpus, id_groups)

    # Output results
    for key in corpus:
        phase_dir = args.out_dir / key
        phase_dir.mkdir(parents=True, exist_ok=True)

        _corpus = _sort_corpus_by_script_id(corpus[key])

        ids, features, labels = zip(*_corpus)
        logger.info(f"Saving {key} file with {len(labels):,} corpus in {args.out_dir}")

        dump(list(ids), phase_dir / "ids.pkl", compress=True)
        dump(list(features), phase_dir / "features.pkl", compress=True)
        dump(list(labels), phase_dir / "labels.pkl", compress=True)


if __name__ == "__main__":
    sys.exit(entry())
