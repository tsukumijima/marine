from pathlib import Path
from typing import Any, cast

import numpy as np
from joblib import dump, load

import marine.bin.pack_corpus as pack_corpus
from marine.bin.pack_corpus import (
    TARGET_ID_METADATA_FILE_NAME,
    build_script_ids_signature,
    insert_punctuation_by_extracted_features,
    is_surface_aligned_mora_sequence,
    load_target_id_metadata,
    process,
    resolve_target_id_groups,
    split_script_ids,
)
from marine.data.feature.feature_set import FeatureSet
from marine.types import MarineFeature
from marine.utils.g2p_util import pron2mora


def test_surface_aligned_mora_sequence_accepts_long_vowel_variant() -> None:
    """長音表記揺れだけの差分は同一形態素として扱えることを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "悲鳴",
                "pron": "ヒメエ",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("ヒメイ")),
        )
        is True
    )


def test_surface_aligned_mora_sequence_accepts_surface_specific_variant() -> None:
    """表層ごとに安全と判断した読み分かれを許容できることを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "いう",
                "pron": "イウ",
                "pos": "動詞:自立:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
            {
                "surface": "人",
                "pron": "ニン",
                "pos": "名詞:接尾:地域:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
            {
                "surface": "良い",
                "pron": "ヨイ",
                "pos": "形容詞:自立:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("ユウジンイイ")),
        )
        is True
    )


def test_surface_aligned_mora_sequence_accepts_dataset_typo_like_reading() -> None:
    """surface として不自然な dataset 側読みは補正対象として救済することを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "線路",
                "pron": "センロ",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("ロセン")),
        )
        is True
    )


def test_surface_aligned_mora_sequence_accepts_nbest_alternative_reading() -> None:
    """n-best 読み候補に含まれる pronunciation 差分は許容できることを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "角",
                "pron": "カク",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
            {
                "surface": "等",
                "pron": "トウ",
                "pos": "接頭詞:名詞接続:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("カドナド")),
        )
        is True
    )


def test_surface_aligned_mora_sequence_accepts_multi_node_compound_reading() -> None:
    """OpenJTalk 側で分割された複合語でも、結合 surface の読み候補で救済できることを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "再",
                "pron": "サイ",
                "pos": "接頭詞:名詞接続:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
            {
                "surface": "入",
                "pron": "イレ",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("サイニュー")),
        )
        is True
    )


def test_surface_aligned_mora_sequence_accepts_length_changing_reading() -> None:
    """surface から説明可能な読み差であれば長さが変わっても許容できることを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "入れ",
                "pron": "イレ",
                "pos": "動詞:自立:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("ハイレ")),
        )
        is True
    )


def test_surface_aligned_mora_sequence_accepts_voicing_difference() -> None:
    """surface が一致していれば、濁点差分だけの読み違いは許容できることを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "換算",
                "pron": "カンサン",
                "pos": "名詞:サ変接続:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("カンザン")),
        )
        is True
    )


def test_surface_aligned_mora_sequence_accepts_split_kana_nodes_for_rare_mora() -> None:
    """rare モーラがカナ node に分割されても、surface から救済できることを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "ゲシ",
                "pron": "ゲシ",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
            {
                "surface": "ィ",
                "pron": "ィ",
                "pos": "フィラー:*:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 1,
            },
            {
                "surface": "グ",
                "pron": "グ",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 1,
            },
            {
                "surface": "ゥ",
                "pron": "ゥ",
                "pos": "フィラー:*:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 1,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("ゲシィグゥ")),
        )
        is True
    )


def test_surface_aligned_mora_sequence_accepts_kana_surface_as_pronunciation_hint() -> (
    None
):
    """カナ surface 自体が phonetic な場合は、その綴りを pronunciation 候補として使えることを確認する。"""

    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "グゥ",
                "pron": "グー",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
            {
                "surface": "シィ",
                "pron": "シー",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    assert (
        is_surface_aligned_mora_sequence(
            nodes,
            cast(list[str], pron2mora("グゥシィ")),
        )
        is True
    )


def test_insert_punctuation_by_extracted_features_allows_soft_pron_difference() -> None:
    """発音本体に安全な揺れがあっても punctuation 挿入だけは継続できることを確認する。"""

    labels = {
        "accent_status": [1, 0, 0],
        "accent_phrase_boundary": [0, 0, 0],
        "intonation_phrase_boundary": [0, 0, 0],
    }

    result = insert_punctuation_by_extracted_features(
        np.array([1, 2, 99, 3]),
        np.array([1, 20, 3]),
        labels,
        [99],
        "mora",
    )

    assert result["accent_status"] == [1, 0, 0, 0]
    assert result["accent_phrase_boundary"] == [0, 0, 0, 0]
    assert result["intonation_phrase_boundary"] == [0, 0, 0, 0]


def test_remap_labels_by_mora_alignment_accepts_small_insertion() -> None:
    """少数モーラの挿入差分であれば label を extracted 側へ写像できることを確認する。"""

    remapped_labels = cast(Any, pack_corpus)._remap_labels_by_mora_alignment(
        cast(list[str], pron2mora("ゴイッショニ")),
        cast(list[str], pron2mora("イッショニ")),
        {
            "accent_status": [1, 0, 0, 0],
            "accent_phrase_boundary": [0, 0, 0, 1],
        },
    )

    assert remapped_labels is not None
    assert remapped_labels["accent_status"] == [0, 1, 0, 0, 0]
    assert remapped_labels["accent_phrase_boundary"] == [0, 0, 0, 0, 1]


def test_process_drops_ap_label_length_mismatch_when_ap_count_cannot_match() -> None:
    """AP 数の整合が取れない場合は、長さ不一致を救済しないことを確認する。"""

    vocab_path = (
        Path(__file__).resolve().parent.parent
        / "recipe"
        / "common"
        / "database"
        / "20220912_jsut_vocab_min_2"
        / "vocab.pkl"
    )
    feature_set = FeatureSet(vocab_path, feature_table_key="open-jtalk")
    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "abc",
                "pron": "イレ",
                "pos": "動詞:自立:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    packed_item, wrong_mora_info = process(
        nodes,
        feature_set,
        "ap",
        script_id="test-script",
        surface="abc",
        pron="ハイレ",
        accent_status="1,2",
        accent_phrase_boundary="0,0,0",
        intonation_phrase_boundary="0,0,0",
    )

    assert packed_item is None
    assert wrong_mora_info == "test-script|イレ|ハイレ\n"


def test_process_remaps_ap_boundary_labels_without_touching_ap_accent() -> None:
    """AP ラベルでは accent_status を保ったまま境界ラベルだけを再配置できることを確認する。"""

    vocab_path = (
        Path(__file__).resolve().parent.parent
        / "recipe"
        / "common"
        / "database"
        / "20220912_jsut_vocab_min_2"
        / "vocab.pkl"
    )
    feature_set = FeatureSet(vocab_path, feature_table_key="open-jtalk")
    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "abc",
                "pron": "イレ",
                "pos": "動詞:自立:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    packed_item, wrong_mora_info = process(
        nodes,
        feature_set,
        "ap",
        script_id="test-script",
        surface="abc",
        pron="ハイレ",
        accent_status="1",
        accent_phrase_boundary="0,0,0",
        intonation_phrase_boundary="0,0,0",
    )

    assert packed_item is not None
    assert wrong_mora_info is None
    _, _, labels = packed_item
    assert labels["accent_status"].tolist() == [1]
    assert labels["accent_phrase_boundary"].tolist() == [0, 0]


def test_process_prefers_surface_aligned_node_pronunciation_for_ap_labels() -> None:
    """surface 候補で説明できる読みは AP ラベルを保ったまま expected pron へ寄せることを確認する。"""

    vocab_path = (
        Path(__file__).resolve().parent.parent
        / "recipe"
        / "common"
        / "database"
        / "20220912_jsut_vocab_min_2"
        / "vocab.pkl"
    )
    feature_set = FeatureSet(vocab_path, feature_table_key="open-jtalk")
    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "入れ",
                "pron": "イレ",
                "pos": "動詞:自立:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    packed_item, wrong_mora_info = process(
        nodes,
        feature_set,
        "ap",
        script_id="test-script",
        surface="入れ",
        pron="ハイレ",
        accent_status="1",
        accent_phrase_boundary="0,0,0",
        intonation_phrase_boundary="0,0,0",
    )

    assert packed_item is not None
    assert wrong_mora_info is None
    _, feature, labels = packed_item
    assert (
        feature["mora"].tolist()
        == feature_set.convert_feature_to_id(
            "mora",
            ["ハ", "イ", "レ"],
        ).tolist()
    )
    assert labels["accent_status"].tolist() == [1]
    assert labels["accent_phrase_boundary"].tolist() == [0, 0, 0]


def test_process_accepts_ap_label_when_ap_boundary_count_matches() -> None:
    """AP ラベルでは accent phrase 数と accent 数が一致すれば正常に pack できることを確認する。"""

    vocab_path = (
        Path(__file__).resolve().parent.parent
        / "recipe"
        / "common"
        / "database"
        / "20220912_jsut_vocab_min_2"
        / "vocab.pkl"
    )
    feature_set = FeatureSet(vocab_path, feature_table_key="open-jtalk")
    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "水",
                "pron": "ミズ",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
            {
                "surface": "を",
                "pron": "オ",
                "pos": "助詞:格助詞:一般:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 1,
            },
            {
                "surface": "買う",
                "pron": "カウ",
                "pos": "動詞:自立:*:*",
                "c_type": "五段・ワ行促音便",
                "c_form": "基本形",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )

    packed_item, wrong_mora_info = process(
        nodes,
        feature_set,
        "ap",
        script_id="test-script",
        surface="水を買う",
        pron="ミズオカウ",
        accent_status="1,2",
        accent_phrase_boundary="0,0,1,0,0",
        intonation_phrase_boundary="0,0,0,0,0",
    )

    assert packed_item is not None
    assert wrong_mora_info is None
    _, _, labels = packed_item
    assert labels["accent_status"].tolist() == [1, 2]


def test_process_remaps_same_length_rescued_reading(
    monkeypatch: Any,
) -> None:
    """同長の救済読みでも process() が remap 結果を採用することを確認する。"""

    vocab_path = (
        Path(__file__).resolve().parent.parent
        / "recipe"
        / "common"
        / "database"
        / "20220912_jsut_vocab_min_2"
        / "vocab.pkl"
    )
    feature_set = FeatureSet(vocab_path, feature_table_key="open-jtalk")
    nodes = cast(
        list[MarineFeature],
        [
            {
                "surface": "線路",
                "pron": "センロ",
                "pos": "名詞:一般:*:*",
                "c_type": "*",
                "c_form": "*",
                "accent_type": 0,
                "accent_con_type": "*",
                "chain_flag": 0,
            },
        ],
    )
    remapped_labels = {
        "accent_status": [0, 1, 0],
        "accent_phrase_boundary": [1, 0, 0],
        "intonation_phrase_boundary": [0, 0, 1],
    }

    def mock_is_surface_aligned_mora_sequence(
        nodes: list[MarineFeature],
        expected_moras: list[str],
    ) -> bool:
        del nodes, expected_moras
        return True

    def mock_remap_labels_by_mora_alignment(
        extracted_moras: list[str],
        expected_moras: list[str],
        original_labels: dict[str, list[int]],
    ) -> dict[str, list[int]]:
        del extracted_moras, expected_moras, original_labels
        return remapped_labels

    monkeypatch.setattr(
        pack_corpus,
        "is_surface_aligned_mora_sequence",
        mock_is_surface_aligned_mora_sequence,
    )
    monkeypatch.setattr(
        pack_corpus,
        "_remap_labels_by_mora_alignment",
        mock_remap_labels_by_mora_alignment,
    )

    packed_item, wrong_mora_info = process(
        nodes,
        feature_set,
        "mora",
        script_id="test-script",
        surface="線路",
        pron="ロセン",
        accent_status="1,0,0",
        accent_phrase_boundary="0,1,0",
        intonation_phrase_boundary="0,0,1",
    )

    assert packed_item is not None
    assert wrong_mora_info is None
    _, _, labels = packed_item
    assert labels["accent_status"].tolist() == remapped_labels["accent_status"]
    assert (
        labels["accent_phrase_boundary"].tolist()
        == remapped_labels["accent_phrase_boundary"]
    )
    assert (
        labels["intonation_phrase_boundary"].tolist()
        == remapped_labels["intonation_phrase_boundary"]
    )


def test_split_script_ids_uses_per_split_size() -> None:
    script_ids = [f"A_{index:04d}" for index in range(1, 31)]

    id_groups = split_script_ids(
        script_ids=script_ids,
        random_seed=12345,
        test_size=4,
    )

    assert len(id_groups["val"]) == 4
    assert len(id_groups["test"]) == 4


def test_resolve_target_id_groups_migrates_legacy_ids(tmp_path: Path) -> None:
    target_id_dir = tmp_path / "target_ids"
    (target_id_dir / "val").mkdir(parents=True)
    (target_id_dir / "test").mkdir(parents=True)

    dump(["A_0002", "A_0005"], target_id_dir / "val" / "ids.pkl", compress=True)
    dump(["A_0003", "A_0006"], target_id_dir / "test" / "ids.pkl", compress=True)

    resolved_groups = resolve_target_id_groups(
        script_ids=[f"A_{index:04d}" for index in range(1, 11)],
        target_id_dir=target_id_dir,
        random_seed=12345,
        test_size=2,
    )

    assert resolved_groups == {
        "val": {"A_0002", "A_0005"},
        "test": {"A_0003", "A_0006"},
    }
    assert (target_id_dir / TARGET_ID_METADATA_FILE_NAME).exists() is True

    metadata = load_target_id_metadata(target_id_dir)
    assert metadata is not None
    assert metadata["script_count"] == 10
    assert metadata["test_size"] == 2


def test_resolve_target_id_groups_regenerates_on_script_id_mismatch(
    tmp_path: Path,
) -> None:
    target_id_dir = tmp_path / "target_ids"
    script_ids = [f"A_{index:04d}" for index in range(1, 21)]

    initial_groups = resolve_target_id_groups(
        script_ids=script_ids,
        target_id_dir=target_id_dir,
        random_seed=12345,
        test_size=3,
    )
    initial_metadata = load_target_id_metadata(target_id_dir)
    assert initial_metadata is not None

    changed_script_ids = [
        script_id for script_id in script_ids if script_id != "A_0004"
    ]
    changed_script_ids.append("A_9999")
    changed_script_ids = sorted(changed_script_ids)

    regenerated_groups = resolve_target_id_groups(
        script_ids=changed_script_ids,
        target_id_dir=target_id_dir,
        random_seed=12345,
        test_size=3,
    )
    regenerated_metadata = load_target_id_metadata(target_id_dir)

    assert regenerated_metadata is not None
    assert regenerated_metadata["script_ids_hash"] == build_script_ids_signature(
        changed_script_ids
    )
    assert (
        regenerated_metadata["script_ids_hash"] != initial_metadata["script_ids_hash"]
    )
    assert len(regenerated_groups["val"]) == 3
    assert len(regenerated_groups["test"]) == 3
    assert regenerated_groups != initial_groups
    assert load(target_id_dir / "val" / "ids.pkl") == sorted(regenerated_groups["val"])
