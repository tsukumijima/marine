from marine.utils.jsut_source import (
    build_canonical_hiragana_from_annotation,
    build_canonical_phone_from_annotation,
    select_jsut_surface_text,
    strip_jsut_annotation_symbols,
)


def test_strip_jsut_annotation_symbols() -> None:
    annotation = "^キョ[ーシュクデスガ_ニ]ガツ#ニ]ジュウ#ナ[ノカノ#ヤ[クソクヲ?$"

    assert (
        strip_jsut_annotation_symbols(annotation)
        == "キョーシュクデスガニガツニジュウナノカノヤクソクヲ"
    )


def test_build_canonical_hiragana_from_annotation() -> None:
    annotation = "^きょ[ーしゅくですが_に]がつ#に]じゅう#な[のかの#や[くそくを?$"

    assert (
        build_canonical_hiragana_from_annotation(annotation)
        == "きょーしゅくですが、にがつにじゅうなのかのやくそくを？"
    )


def test_build_canonical_phone_from_annotation() -> None:
    phoneme_annotation = (
        "^-ky-o-[-o-sh-u-k-u-d-e-]-s-u-g-a-_-n-i-]-g-a-ts-u-#"
        "-n-i-]-j-u-u-#-n-a-[-n-o-k-a-$"
    )

    assert (
        build_canonical_phone_from_annotation(phoneme_annotation)
        == "ky-o-o-sh-u-k-u-d-e-s-u-g-a-pau-n-i-g-a-ts-u-n-i-j-u-u-n-a-n-o-k-a"
    )


def test_select_jsut_surface_text_prefers_text_level2() -> None:
    text_item = {
        "text_level0": "原文",
        "text_level2": "じっさいのはつわ",
    }

    assert select_jsut_surface_text(text_item) == "じっさいのはつわ"


def test_select_jsut_surface_text_falls_back_to_text_level0() -> None:
    text_item = {
        "text_level0": "原文",
        "text_level2": "",
    }

    assert select_jsut_surface_text(text_item) == "原文"
