from __future__ import annotations

from typing import cast

from marine.utils.g2p_util import pron2mora


ACCENT_MARKERS = {"[", "]"}
PHRASE_BOUNDARY_MARKERS = {"#", "_"}


def split_annotation_body_into_phrase_tokens(
    annotation: str,
) -> tuple[list[list[str]], list[str]]:
    """
    JSUT 互換 prosody 文字列をアクセント句ごとの token 列へ分割する。

    Args:
        annotation (str): `^...$` 形式の JSUT 互換 prosody 文字列

    Returns:
        tuple[list[list[str]], list[str]]: アクセント句 token 列と構文エラー一覧
    """

    structural_errors: list[str] = []

    if annotation.startswith("^") is False:
        structural_errors.append("missing_sentence_start")
        annotation_body = annotation
    else:
        annotation_body = annotation[1:]

    if annotation_body.endswith("?$") is True:
        annotation_body = annotation_body[:-2]
    elif annotation_body.endswith("$") is True:
        annotation_body = annotation_body[:-1]
    else:
        structural_errors.append("missing_sentence_end")

    question_marker_indexes = [
        character_index
        for character_index, character in enumerate(annotation_body)
        if character == "?"
    ]
    for question_marker_index in question_marker_indexes:
        if question_marker_index == len(annotation_body) - 1:
            structural_errors.append("dangling_question_marker")
            continue

        if annotation_body[question_marker_index + 1] not in PHRASE_BOUNDARY_MARKERS:
            structural_errors.append("unexpected_question_marker_position")

    annotation_body = annotation_body.replace("?", "")

    annotation_tokens = cast(list[str], pron2mora(annotation_body))
    phrase_tokens: list[list[str]] = []
    current_phrase_tokens: list[str] = []

    for token in annotation_tokens:
        if token in PHRASE_BOUNDARY_MARKERS:
            if len(current_phrase_tokens) == 0:
                structural_errors.append("empty_phrase_before_boundary")
            else:
                phrase_tokens.append(current_phrase_tokens)
                current_phrase_tokens = []
            continue

        current_phrase_tokens.append(token)

    if len(current_phrase_tokens) == 0:
        structural_errors.append("empty_phrase_at_sentence_end")
    else:
        phrase_tokens.append(current_phrase_tokens)

    if len(phrase_tokens) == 0:
        structural_errors.append("empty_sentence_body")

    return phrase_tokens, structural_errors


def validate_accent_phrase_tokens(phrase_tokens: list[str]) -> list[str]:
    """
    アクセント句 token 列の prosody 構文を検証する。

    Args:
        phrase_tokens (list[str]): 1 アクセント句分の token 列

    Returns:
        list[str]: 検出した構文エラー一覧
    """

    structural_errors: list[str] = []
    mora_tokens = [token for token in phrase_tokens if token not in ACCENT_MARKERS]
    open_marker_count = phrase_tokens.count("[")
    close_marker_count = phrase_tokens.count("]")

    if len(mora_tokens) == 0:
        structural_errors.append("empty_accent_phrase")
        return structural_errors

    if phrase_tokens[0] in ACCENT_MARKERS:
        structural_errors.append("accent_marker_at_phrase_start")

    for token_index in range(1, len(phrase_tokens)):
        if (
            phrase_tokens[token_index] in ACCENT_MARKERS
            and phrase_tokens[token_index - 1] in ACCENT_MARKERS
        ):
            structural_errors.append("adjacent_accent_markers")
            break

    if open_marker_count == 0 and close_marker_count == 0:
        structural_errors.append("missing_accent_marker")

    if open_marker_count > 1:
        structural_errors.append("multiple_accent_rise_markers")

    if close_marker_count > 1:
        structural_errors.append("multiple_accent_fall_markers")

    if open_marker_count > 0:
        open_marker_index = phrase_tokens.index("[")
        if len(mora_tokens) > 1 and open_marker_index == len(phrase_tokens) - 1:
            structural_errors.append("accent_rise_marker_at_phrase_end")
    else:
        open_marker_index = None

    if close_marker_count > 0:
        close_marker_index = phrase_tokens.index("]")
        if close_marker_index == len(phrase_tokens) - 1:
            structural_errors.append("accent_fall_marker_at_phrase_end")
        if len(mora_tokens) == 1:
            structural_errors.append("accent_fall_marker_in_single_mora_phrase")
    else:
        close_marker_index = None

    if open_marker_index is not None and close_marker_index is not None:
        if open_marker_index > close_marker_index:
            structural_errors.append("accent_marker_order_is_reversed")

    return structural_errors


def get_annotation_structure_errors(annotation: str) -> list[str]:
    """
    JSUT 互換 prosody 文字列全体の構文エラー一覧を返す。

    Args:
        annotation (str): 検証対象の JSUT 互換 prosody 文字列

    Returns:
        list[str]: 検出した構文エラー一覧
    """

    phrase_tokens_list, structural_errors = split_annotation_body_into_phrase_tokens(
        annotation
    )

    for phrase_index, phrase_tokens in enumerate(phrase_tokens_list):
        for phrase_error in validate_accent_phrase_tokens(phrase_tokens):
            structural_errors.append(f"phrase_{phrase_index}:{phrase_error}")

    return structural_errors
