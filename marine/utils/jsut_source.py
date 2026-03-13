from __future__ import annotations

import re
from typing import Any


KATAKANA_TO_HIRAGANA_TRANSLATION_TABLE = str.maketrans(
    {
        "ア": "あ",
        "イ": "い",
        "ウ": "う",
        "エ": "え",
        "オ": "お",
        "カ": "か",
        "キ": "き",
        "ク": "く",
        "ケ": "け",
        "コ": "こ",
        "サ": "さ",
        "シ": "し",
        "ス": "す",
        "セ": "せ",
        "ソ": "そ",
        "タ": "た",
        "チ": "ち",
        "ツ": "つ",
        "テ": "て",
        "ト": "と",
        "ナ": "な",
        "ニ": "に",
        "ヌ": "ぬ",
        "ネ": "ね",
        "ノ": "の",
        "ハ": "は",
        "ヒ": "ひ",
        "フ": "ふ",
        "ヘ": "へ",
        "ホ": "ほ",
        "マ": "ま",
        "ミ": "み",
        "ム": "む",
        "メ": "め",
        "モ": "も",
        "ヤ": "や",
        "ユ": "ゆ",
        "ヨ": "よ",
        "ラ": "ら",
        "リ": "り",
        "ル": "る",
        "レ": "れ",
        "ロ": "ろ",
        "ワ": "わ",
        "ヲ": "を",
        "ン": "ん",
        "ガ": "が",
        "ギ": "ぎ",
        "グ": "ぐ",
        "ゲ": "げ",
        "ゴ": "ご",
        "ザ": "ざ",
        "ジ": "じ",
        "ズ": "ず",
        "ゼ": "ぜ",
        "ゾ": "ぞ",
        "ダ": "だ",
        "ヂ": "ぢ",
        "ヅ": "づ",
        "デ": "で",
        "ド": "ど",
        "バ": "ば",
        "ビ": "び",
        "ブ": "ぶ",
        "ベ": "べ",
        "ボ": "ぼ",
        "パ": "ぱ",
        "ピ": "ぴ",
        "プ": "ぷ",
        "ペ": "ぺ",
        "ポ": "ぽ",
        "ァ": "ぁ",
        "ィ": "ぃ",
        "ゥ": "ぅ",
        "ェ": "ぇ",
        "ォ": "ぉ",
        "ャ": "ゃ",
        "ュ": "ゅ",
        "ョ": "ょ",
        "ッ": "っ",
        "ヮ": "ゎ",
        "ー": "ー",
        "、": "、",
        "。": "。",
        "？": "？",
        "！": "！",
        ",": "、",
        ".": "。",
        "?": "？",
        "!": "！",
    }
)
HIRAGANA_TO_KATAKANA_TRANSLATION_TABLE = str.maketrans(
    {
        value: chr(key)
        for key, value in KATAKANA_TO_HIRAGANA_TRANSLATION_TABLE.items()
        if len(value) == 1
    }
)

ANNOTATION_CONTROL_CHARACTERS = "^$[]"
PHRASE_BOUNDARY_TO_PUNCTUATION_TABLE = str.maketrans(
    {
        "#": "",
        "_": "、",
    }
)
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

TEXT_ENTRY_FIELD_ORDER = [
    "text_level0",
    "kana_level0",
    "text_level1",
    "text_level2",
    "kana_level2",
    "kana_level3",
    "phone_level3",
]


def strip_jsut_annotation_symbols(annotation: str) -> str:
    """
    JSUT 互換 prosody 文字列から制御記号を除いた本文を返す。

    Args:
        annotation (str): `^...$` 形式の JSUT 互換 prosody 文字列

    Returns:
        str: 制御記号を除去したカタカナ本文
    """

    normalized_annotation = annotation.translate(
        str.maketrans("", "", ANNOTATION_CONTROL_CHARACTERS),
    )
    normalized_annotation = normalized_annotation.replace("#", "").replace("_", "")
    normalized_annotation = normalized_annotation.replace("?", "")
    return normalized_annotation


def convert_katakana_to_hiragana(text: str) -> str:
    """
    カタカナ文字列をひらがなへ変換する。

    Args:
        text (str): 変換対象のカタカナ文字列

    Returns:
        str: ひらがなへ変換した文字列
    """

    return text.translate(KATAKANA_TO_HIRAGANA_TRANSLATION_TABLE)


def convert_hiragana_to_katakana(text: str) -> str:
    """
    ひらがな文字列をカタカナへ変換する。

    Args:
        text (str): 変換対象のひらがな文字列

    Returns:
        str: カタカナへ変換した文字列
    """

    return text.translate(HIRAGANA_TO_KATAKANA_TRANSLATION_TABLE)


def build_canonical_hiragana_from_annotation(annotation: str) -> str:
    """
    `e2e_symbol/hiragana.yaml` と同等の正準ひらがな表記を構築する。

    Args:
        annotation (str): ひらがなまたはカタカナの JSUT 互換 prosody 文字列

    Returns:
        str: 句読点を含む正準ひらがな表記
    """

    is_question_sentence = annotation.endswith("?$")
    annotation_body = annotation.translate(
        str.maketrans("", "", ANNOTATION_CONTROL_CHARACTERS),
    )
    annotation_body = annotation_body.translate(PHRASE_BOUNDARY_TO_PUNCTUATION_TABLE)
    annotation_body = annotation_body.replace("?", "")
    hiragana_text = convert_katakana_to_hiragana(annotation_body)

    if is_question_sentence is True:
        return f"{hiragana_text}？"

    return hiragana_text


def build_canonical_phone_from_annotation(phoneme_annotation: str) -> str:
    """
    `e2e_symbol/phoneme.yaml` から `phone_level3` 相当の正準音素列を構築する。

    Args:
        phoneme_annotation (str): 音素列の JSUT 互換 prosody 文字列

    Returns:
        str: `phone_level3` と同形式のハイフン区切り音素列
    """

    annotation_body = phoneme_annotation.translate(
        str.maketrans("", "", ANNOTATION_CONTROL_CHARACTERS + "?"),
    )
    annotation_tokens = [token for token in annotation_body.split("-") if token != ""]
    canonical_tokens: list[str] = []
    for annotation_token in annotation_tokens:
        if annotation_token == "#":
            continue
        if annotation_token == "_":
            canonical_tokens.append("pau")
            continue
        canonical_tokens.append(annotation_token)

    return "-".join(canonical_tokens)


def select_jsut_surface_text(text_item: dict[str, Any]) -> str:
    """
    JSUT 系データから surface として使う本文を選択する。

    `text_level2` は実際の発話文を表すため、存在する場合はそれを優先する。
    互換性のため `text_level2` が空の場合のみ `text_level0` へフォールバックする。

    Args:
        text_item (dict[str, Any]): `text.yaml` または `basic5000.yaml` の各項目

    Returns:
        str: surface として利用する本文
    """

    text_level2 = str(text_item.get("text_level2", "")).strip()
    if len(text_level2) > 0:
        return text_level2

    return str(text_item["text_level0"])


def reorder_text_entry_fields(
    text_item: dict[str, Any],
    reference_text_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    `text.yaml` / `basic5000.yaml` の項目順を安定化した辞書を返す。

    Args:
        text_item (dict[str, Any]): 整列対象の項目
        reference_text_item (dict[str, Any] | None): 可能であればこの項目順を優先する参照項目

    Returns:
        dict[str, Any]: 項目順を整えた辞書
    """

    ordered_keys: list[str] = []

    if reference_text_item is not None:
        ordered_keys.extend(
            key for key in reference_text_item.keys() if key in text_item
        )

    for field_name in TEXT_ENTRY_FIELD_ORDER:
        if field_name in text_item and field_name not in ordered_keys:
            ordered_keys.append(field_name)

    for key in text_item.keys():
        if key not in ordered_keys:
            ordered_keys.append(key)

    return {key: text_item[key] for key in ordered_keys}


def normalize_pronunciation_for_alignment(text: str) -> str:
    """
    長音や四つ仮名の代表的な揺れを吸収する比較用文字列へ正規化する。

    Args:
        text (str): 比較対象のカタカナ発音列

    Returns:
        str: 長音揺れを吸収した比較用文字列
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
    normalized_text = normalized_text.replace("ヲ", "オ")
    normalized_text = normalized_text.replace("ヅ", "ズ")
    normalized_text = normalized_text.replace("ヂ", "ジ")
    return normalized_text


def normalize_voicing_for_alignment(text: str) -> str:
    """
    濁点・半濁点差を吸収した比較用文字列へ正規化する。

    Args:
        text (str): 比較対象のカタカナ発音列

    Returns:
        str: 濁点差分を除いた比較用文字列
    """

    return text.translate(VOICE_NORMALIZATION_TABLE)
