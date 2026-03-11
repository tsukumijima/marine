from marine.bin.validate_yaml_corpus import validate_structural_annotations
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
