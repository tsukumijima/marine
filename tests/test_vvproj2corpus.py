import json
from pathlib import Path

from marine.bin.vvproj2corpus import (
    build_annotation_from_accent_phrases,
    build_corpus_entries,
    build_phone_level3_from_accent_phrases,
    discover_project_paths,
    extract_kana_from_annotation,
    extract_project_audio_data,
    get_canonical_mora_text,
    sanitize_surface_text,
)


def test_discover_project_paths_sorts_naturally(tmp_path: Path) -> None:
    for file_name in [
        "project101-150.vvproj",
        "project1-50.vvproj",
        "project51-100.aisp",
    ]:
        (tmp_path / file_name).write_text("{}", encoding="utf-8")

    project_names = [path.name for path in discover_project_paths(tmp_path)]

    assert project_names == [
        "project1-50.vvproj",
        "project51-100.aisp",
        "project101-150.vvproj",
    ]


def test_build_annotation_from_accent_phrases_creates_jsut_style() -> None:
    accent_phrases = [
        {
            "moras": [
                {"text": "ナ", "consonant": "n", "vowel": "a"},
                {"text": "ガ", "consonant": "g", "vowel": "a"},
                {"text": "シ", "consonant": "sh", "vowel": "i"},
            ],
            "accent": 3,
            "pauseMora": None,
        },
        {
            "moras": [
                {"text": "ハ", "consonant": "h", "vowel": "a"},
                {"text": "イ", "vowel": "i"},
            ],
            "accent": 1,
            "pauseMora": {"text": "、", "vowel": "pau"},
        },
    ]

    annotation = build_annotation_from_accent_phrases(accent_phrases, "流し、")

    assert annotation == "^ナ[ガシ#ハ]イ$"
    assert extract_kana_from_annotation(annotation) == "ナガシハイ"


def test_build_phone_level3_from_accent_phrases_includes_pause() -> None:
    accent_phrases = [
        {
            "moras": [
                {"text": "ナ", "consonant": "n", "vowel": "a"},
                {"text": "ガ", "consonant": "g", "vowel": "a"},
            ],
            "accent": 2,
            "pauseMora": {"text": "、", "vowel": "pau"},
        },
    ]

    assert build_phone_level3_from_accent_phrases(accent_phrases) == "n-a-g-a-pau"


def test_get_canonical_mora_text_uses_phoneme_driven_mapping() -> None:
    assert (
        get_canonical_mora_text({"text": "ヒュ", "consonant": "fy", "vowel": "u"})
        == "フュ"
    )
    assert (
        get_canonical_mora_text({"text": "フュ", "consonant": "hy", "vowel": "u"})
        == "ヒュ"
    )
    assert (
        get_canonical_mora_text({"text": "グ", "consonant": "gw", "vowel": "e"})
        == "グェ"
    )
    assert (
        get_canonical_mora_text({"text": "テ", "consonant": "ty", "vowel": "u"})
        == "テュ"
    )
    assert (
        get_canonical_mora_text({"text": "ス", "consonant": "s", "vowel": "U"}) == "ス"
    )
    assert get_canonical_mora_text({"text": "イ", "vowel": "N"}) == "ン"


def test_extract_kana_from_annotation_recovers_pause_punctuation() -> None:
    assert extract_kana_from_annotation("^フュ[ジョン_ヴィラ$") == "フュジョン、ヴィラ"


def test_build_annotation_prefers_phoneme_consistent_mora_over_surface_spelling() -> (
    None
):
    accent_phrases = [
        {
            "moras": [
                {"text": "グ", "consonant": "g", "vowel": "u"},
                {"text": "エ", "vowel": "e"},
                {"text": "ル", "consonant": "r", "vowel": "u"},
            ],
            "accent": 2,
            "pauseMora": None,
        },
    ]

    annotation = build_annotation_from_accent_phrases(accent_phrases, "グェル")

    assert annotation == "^グ[エ]ル$"
    assert extract_kana_from_annotation(annotation) == "グエル"


def test_build_annotation_keeps_split_surface_like_foreign_mora_when_phones_are_split() -> (
    None
):
    accent_phrases = [
        {
            "moras": [
                {"text": "フ", "consonant": "f", "vowel": "u"},
                {"text": "ユ", "consonant": "y", "vowel": "u"},
                {"text": "ウ", "vowel": "u"},
            ],
            "accent": 1,
            "pauseMora": None,
        },
    ]

    annotation = build_annotation_from_accent_phrases(accent_phrases, "フュー")

    assert annotation == "^フ]ユウ$"
    assert extract_kana_from_annotation(annotation) == "フユウ"


def test_sanitize_surface_text_removes_leading_label() -> None:
    assert (
        sanitize_surface_text("ROHAN4600_0001:流し斬りが入る。") == "流し斬りが入る。"
    )
    assert sanitize_surface_text("普通の本文です。") == "普通の本文です。"
    assert sanitize_surface_text("Q: どうする？") == "Q: どうする？"
    assert sanitize_surface_text("2025: 開幕です") == "2025: 開幕です"


def test_generated_annotation_can_be_built_from_realistic_vvproj_shape(
    tmp_path: Path,
) -> None:
    vvproj_path = tmp_path / "project1-50.vvproj"
    vvproj_path.write_text(
        json.dumps(
            {
                "audioKeys": ["item1"],
                "audioItems": {
                    "item1": {
                        "text": "ROHAN4600_0001:流し斬りが入る。",
                        "query": {
                            "accentPhrases": [
                                {
                                    "moras": [
                                        {"text": "ナ", "consonant": "n", "vowel": "a"},
                                        {"text": "ガ", "consonant": "g", "vowel": "a"},
                                    ],
                                    "accent": 2,
                                    "pauseMora": None,
                                },
                            ],
                        },
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded_vvproj = json.loads(vvproj_path.read_text(encoding="utf-8"))
    accent_phrases = loaded_vvproj["audioItems"]["item1"]["query"]["accentPhrases"]
    annotation = build_annotation_from_accent_phrases(
        accent_phrases,
        sanitize_surface_text(loaded_vvproj["audioItems"]["item1"]["text"]),
    )

    assert annotation == "^ナ[ガ$"


def test_extract_project_audio_data_supports_aisp_talk_nesting() -> None:
    audio_items, audio_keys = extract_project_audio_data(
        Path("sample.aisp"),
        {
            "talk": {
                "audioItems": {"item1": {"text": "sample"}},
                "audioKeys": ["item1"],
            },
        },
    )

    assert audio_keys == ["item1"]
    assert audio_items["item1"]["text"] == "sample"


def test_build_corpus_entries_reads_aisp_projects(tmp_path: Path) -> None:
    aisp_path = tmp_path / "project1-50.aisp"
    aisp_path.write_text(
        json.dumps(
            {
                "talk": {
                    "audioKeys": ["item1"],
                    "audioItems": {
                        "item1": {
                            "text": "ROHAN4600_0001:流し斬りが入る。",
                            "query": {
                                "accentPhrases": [
                                    {
                                        "moras": [
                                            {
                                                "text": "ナ",
                                                "consonant": "n",
                                                "vowel": "a",
                                            },
                                            {
                                                "text": "ガ",
                                                "consonant": "g",
                                                "vowel": "a",
                                            },
                                        ],
                                        "accent": 2,
                                        "pauseMora": None,
                                    },
                                ],
                            },
                        },
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    text_entries, annotations = build_corpus_entries(
        source_dir=tmp_path,
        script_id_prefix="TEST",
        script_id_padding=4,
        accent_status_seq_level="mora",
        accent_status_represent_mode="binary",
        strip_text_prefix_pattern=None,
    )

    assert text_entries["TEST_0001"]["text_level0"] == "流し斬りが入る。"
    assert annotations["TEST_0001"] == "^ナ[ガ$"
