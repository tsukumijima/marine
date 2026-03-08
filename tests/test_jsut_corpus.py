from logging import getLogger
from typing import cast

import numpy as np

from marine.bin.jsut2corpus import (
    binary_accent_to_high_low_accent as jsut_binary_accent_to_high_low_accent,
)
from marine.bin.jsut2corpus import parse_jsut_annotation
from marine.bin.make_raw_corpus import (
    binary_accent_to_high_low_accent as raw_binary_accent_to_high_low_accent,
)
from marine.types import AccentRepresentMode


logger = getLogger("test")


def test_jsut_parser():
    for (
        jsut_annotaion,
        accent_status_seq_level,
        accent_status_represent_mode,
        expect,
    ) in [
        (
            "^ム]シロ#ロ[ンゲノホ]ーガ#ハ[ゲヤス]イッテ#キ[ータゾ?$",
            "ap",
            "binary",
            {
                "pron": "ムシロロンゲノホーガハゲヤスイッテキータゾ",
                "accent_status": "1,5,4,0",
                "accent_phrase_boundary": "0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0",
                "intonation_phrase_boundary": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            },
        ),
        (
            "^ム]シロ#ロ[ンゲノホ]ーガ#ハ[ゲヤス]イッテ#キ[ータゾ?$",
            "mora",
            "binary",
            {
                "pron": "ムシロロンゲノホーガハゲヤスイッテキータゾ",
                "accent_status": "1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0",
                "accent_phrase_boundary": "0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0",
                "intonation_phrase_boundary": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            },
        ),
        (
            "^ム]シロ#ロ[ンゲノホ]ーガ#ハ[ゲヤス]イッテ#キ[ータゾ?$",
            "mora",
            "high_low",
            {
                "pron": "ムシロロンゲノホーガハゲヤスイッテキータゾ",
                "accent_status": "1,0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1",
                "accent_phrase_boundary": "0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0",
                "intonation_phrase_boundary": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            },
        ),
    ]:
        accent_status_represent_mode = cast(
            AccentRepresentMode,
            accent_status_represent_mode,
        )
        result = parse_jsut_annotation(
            jsut_annotaion, accent_status_seq_level, accent_status_represent_mode
        )
        assert result == expect


def test_binary_accent_to_high_low_accent_handles_final_accent() -> None:
    """語末アクセントでも high_low 変換が落ちずに処理できることを確認する。"""

    moras = np.array(["カ", "ナ"])
    binary_accents = np.array([0, 1])
    accent_phrase_boundaries = np.array([0, 0])

    expected = [0, 1]

    assert (
        jsut_binary_accent_to_high_low_accent(
            moras,
            binary_accents,
            accent_phrase_boundaries,
            accent_status_represent_mode="high_low",
        )
        == expected
    )
    assert (
        raw_binary_accent_to_high_low_accent(
            moras,
            binary_accents,
            accent_phrase_boundaries,
            accent_status_represent_mode="high_low",
        )
        == expected
    )
