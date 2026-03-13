from marine.bin.validate_yaml_corpus import (
    detect_exact_duplicate_annotations,
    detect_inconsistent_duplicate_surfaces,
    validate_structural_annotations,
)
from marine.utils.jsut_annotation import get_annotation_structure_errors


def test_get_annotation_structure_errors_accepts_valid_annotation() -> None:
    assert get_annotation_structure_errors("^ナ[ガシ#ハ]イ$") == []
    assert get_annotation_structure_errors("^フュ]ウジョンガ#カ[カ]ッタ$") == []
    assert get_annotation_structure_errors("^シ$") == []
    assert (
        get_annotation_structure_errors(
            "^ア[ル]イテ#イ[ク]ノ?_ソ[レト]モ#バ]スデ#イ[ク]ノ?$"
        )
        == []
    )


def test_get_annotation_structure_errors_detects_reversed_markers() -> None:
    structural_errors = get_annotation_structure_errors("^ア]イ[ウ$")

    assert "phrase_0:accent_marker_order_is_reversed" in structural_errors


def test_get_annotation_structure_errors_detects_multiple_rise_markers() -> None:
    structural_errors = get_annotation_structure_errors("^ア[イ[ウ$")

    assert "phrase_0:multiple_accent_rise_markers" in structural_errors


def test_validate_structural_annotations_reports_invalid_annotations() -> None:
    invalid_annotations = validate_structural_annotations(
        {
            "valid": "^ナ[ガシ#ハ]イ$",
            "invalid": "^ア]イ[ウ$",
        }
    )

    assert invalid_annotations == [
        ("invalid", ["phrase_0:accent_marker_order_is_reversed"])
    ]


def test_duplicate_detection_uses_actual_utterance_signature() -> None:
    texts = {
        "base": {
            "text_level0": "貴婦人は急いで病院へ行き、そのまま診察室へと連れて行かれた。",
            "text_level2": "吉雄は私がその箱を持ち上げるのを手伝ってくれた。",
            "kana_level3": "よしおわわたしがそのはこをもちあげるのをてつだってくれた",
        },
        "same_utterance": {
            "text_level0": "吉雄は私がその箱を持ち上げるのを手伝ってくれた。",
            "text_level2": "吉雄は私がその箱を持ち上げるのを手伝ってくれた。",
            "kana_level3": "よしおわわたしがそのはこをもちあげるのをてつだってくれた",
        },
    }
    annotations = {
        "base": "^ヨ[シオワ_ワ[タシガ$",
        "same_utterance": "^ヨ[シオワ_ワ[タシガ$",
    }

    assert detect_exact_duplicate_annotations(texts, annotations) == []


def test_inconsistent_duplicate_surface_detection_respects_actual_kana() -> None:
    texts = {
        "a": {
            "text_level0": "また、唐代中期以降は、直播き式であった稲作は、苗代式に変わっていった。",
            "text_level2": "また、唐代中期以降は、直播き式であった稲作は、苗代式に変わっていった。",
            "kana_level3": "また、とーだいちゅーきいこーわ、じかまきしきであったいなさくわ、なわしろしきにかわっていった",
        },
        "b": {
            "text_level0": "また、唐代中期以降は、直播き式であった稲作は、苗代式に変わっていった。",
            "text_level2": "また、唐代中期以降は、直播き式であった稲作は、苗代式に変わっていった。",
            "kana_level3": "また、とーだいちゅーきいこーわ、じかまきしきであったいなさくわ、いなしろしきにかわっていった",
        },
    }
    annotations = {
        "a": "^マ[タ$",
        "b": "^マ]タ$",
    }

    assert detect_inconsistent_duplicate_surfaces(texts, annotations) == []
