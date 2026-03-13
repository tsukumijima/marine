from marine.bin.validate_yaml_corpus import (
    detect_exact_duplicate_annotations,
    detect_inconsistent_duplicate_surfaces,
    validate_structural_annotations,
)
from marine.utils.jsut_annotation import get_annotation_structure_errors


# JSUT BASIC5000 から抜粋した実際のアノテーション
# BASIC5000_0001: 水をマレーシアから買わなくてはならないのです。
ANNOTATION_0001 = "^ミ[ズヲ#マ[レ]ーシアカラ#カ[ワナ]クテワ#ナ[ラ]ナイノデス$"
# BASIC5000_0002: 木曜日、停戦会談は、何の進展もないまま終了しました。
ANNOTATION_0002 = (
    "^モ[クヨ]ービ_テ[ーセンカ]イダンワ_"
    "ナ[ンノ#シ[ンテンモ#ナ]イママ#シュ[ーリョーシマ]シタ$"
)
# BASIC5000_0003: 上院議員は私がデータをゆがめたと告発した。
ANNOTATION_0003 = "^ジョ[ーインギ]ーンワ_ワ[タシガ#デ]ータヲ#ユ[ガ]メタト#コ[クハツシタ$"
# BASIC5000_0006: 週に四回、フランスの授業があります。
ANNOTATION_0006 = "^シュ]ーニ#ヨ[ンカイ_フ[ランスノ#ジュ]ギョーガ#ア[リマ]ス$"
# BASIC5000_0206: 歩いて行くの、それとも、バスで行くの。(疑問文)
ANNOTATION_0206 = "^ア[ル]イテ#イ[ク]ノ?_ソ[レト]モ_バ]スデ#イ[ク]ノ?$"
# BASIC5000_1177: 貴婦人は急いで病院へ行き、そのまま診察室へと連れて行かれた。
ANNOTATION_1177 = (
    "^キ[フ]ジンワ#イ[ソ]イデ#ビョ[ーインエ#イ[キ_"
    "ソ[ノママ#シ[ンサツ]シツエト#ツ[レテイカレタ$"
)
# BASIC5000_1178: 吉雄は私がその箱を持ち上げるのを手伝ってくれた。
ANNOTATION_1178 = "^ヨ[シオワ_ワ[タシガ#ソ[ノ#ハ[コヲ#モ[チアゲ]ルノヲ#テ[ツダ]ッテクレタ$"
# BASIC5000_3256: また、唐代中期以降は、直播き式であった稲作は、苗代式に変わっていった。
# 「苗代」を「なわしろ」と読む場合
ANNOTATION_3256 = (
    "^マ[タ_ト]ーダイ#チュ[ーキイ]コーワ_"
    "ジ[カマキシキデア]ッタ#イ[ナ]サクワ_ナ[ワシロシキニ#カ[ワッテイ]ッタ$"
)
# BASIC5000_4158: 同じ本文だが「苗代」を「いなしろ」と読む場合
ANNOTATION_4158 = (
    "^マ[タ_ト]ーダイ#チュ[ーキイ]コーワ_"
    "ジ[カマキシキデア]ッタ#イ[ナ]サクワ_イ[ナシロシキニ#カ[ワッテイ]ッタ$"
)


def test_get_annotation_structure_errors_accepts_valid_annotation() -> None:
    # BASIC5000_0001
    assert get_annotation_structure_errors(ANNOTATION_0001) == []
    # BASIC5000_0006: 頭高型 (シュ]ー)
    assert get_annotation_structure_errors(ANNOTATION_0006) == []
    # BASIC5000_0206: 疑問文 (?$)
    assert get_annotation_structure_errors(ANNOTATION_0206) == []
    # BASIC5000_1177: 複数の IP を含む長文
    assert get_annotation_structure_errors(ANNOTATION_1177) == []


def test_get_annotation_structure_errors_detects_reversed_markers() -> None:
    # BASIC5000_0001 の最初の AP でアクセント記号の順序を逆にしたもの
    # ミ[ズ]ヲ (正) → ミ]ズ[ヲ (逆: ] が [ より先に出現)
    reversed_annotation = "^ミ]ズ[ヲ#マ[レ]ーシアカラ$"

    structural_errors = get_annotation_structure_errors(reversed_annotation)

    assert "phrase_0:accent_marker_order_is_reversed" in structural_errors


def test_get_annotation_structure_errors_detects_multiple_rise_markers() -> None:
    # BASIC5000_0001 の最初の AP に上昇記号を 2 つ入れたもの
    # ミ[ズヲ (正) → ミ[ズ[ヲ (上昇記号が 2 つ)
    double_rise_annotation = "^ミ[ズ[ヲ#マ[レ]ーシアカラ$"

    structural_errors = get_annotation_structure_errors(double_rise_annotation)

    assert "phrase_0:multiple_accent_rise_markers" in structural_errors


def test_validate_structural_annotations_reports_invalid_annotations() -> None:
    # BASIC5000_0001 の最初の AP でアクセント記号の順序を逆にしたもの
    reversed_annotation = "^ミ]ズ[ヲ#マ[レ]ーシアカラ$"
    invalid_annotations = validate_structural_annotations(
        {
            "BASIC5000_0001": ANNOTATION_0001,
            "INVALID": reversed_annotation,
        }
    )

    assert invalid_annotations == [
        ("INVALID", ["phrase_0:accent_marker_order_is_reversed"])
    ]


def test_duplicate_detection_does_not_flag_same_text_same_annotation() -> None:
    # BASIC5000_1178 / BASIC5000_1179: 同一テキスト・同一アノテーションの正規な重複
    texts = {
        "BASIC5000_1178": {
            "text": "吉雄は私がその箱を持ち上げるのを手伝ってくれた。",
            "annotation": ANNOTATION_1178,
        },
        "BASIC5000_1179": {
            "text": "吉雄は私がその箱を持ち上げるのを手伝ってくれた。",
            "annotation": ANNOTATION_1178,
        },
    }
    annotations = {
        "BASIC5000_1178": ANNOTATION_1178,
        "BASIC5000_1179": ANNOTATION_1178,
    }

    assert detect_exact_duplicate_annotations(texts, annotations) == []


def test_inconsistent_duplicate_surface_detection_flags_same_text_different_annotation() -> None:
    # BASIC5000_3256 / BASIC5000_4158: 同一テキストだが「苗代」の読みが異なる
    ## 「なわしろ」 vs 「いなしろ」 でアノテーションが異なる
    text = "また、唐代中期以降は、直播き式であった稲作は、苗代式に変わっていった。"
    texts = {
        "BASIC5000_3256": {
            "text": text,
            "annotation": ANNOTATION_3256,
        },
        "BASIC5000_4158": {
            "text": text,
            "annotation": ANNOTATION_4158,
        },
    }
    annotations = {
        "BASIC5000_3256": ANNOTATION_3256,
        "BASIC5000_4158": ANNOTATION_4158,
    }

    inconsistent = detect_inconsistent_duplicate_surfaces(texts, annotations)
    assert len(inconsistent) == 1


def test_inconsistent_duplicate_surface_detection_ignores_different_texts() -> None:
    # BASIC5000_0001 / BASIC5000_0003: 異なるテキストなので不整合にはならない
    texts = {
        "BASIC5000_0001": {
            "text": "水をマレーシアから買わなくてはならないのです。",
            "annotation": ANNOTATION_0001,
        },
        "BASIC5000_0003": {
            "text": "上院議員は私がデータをゆがめたと告発した。",
            "annotation": ANNOTATION_0003,
        },
    }
    annotations = {
        "BASIC5000_0001": ANNOTATION_0001,
        "BASIC5000_0003": ANNOTATION_0003,
    }

    assert detect_inconsistent_duplicate_surfaces(texts, annotations) == []
