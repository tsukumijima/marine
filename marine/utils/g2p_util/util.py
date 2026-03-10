UNACCENTED_MORA = "ン"
CONNECTABLE_MORA = set(["ッ", UNACCENTED_MORA])

NON_MORA_LIST = set(["ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ヮ"])

LONGVOWEL_CHARACTER = "ー"

FULL_PUNCTUATION = "、。？！"
HALF_PUNCTUATION = ",.?!"

# pyopenjtalk-plus / OpenJTalk のモーラ定義を基準に、利便のために OpenJTalk 側にはない表記もエイリアスとして追加
# これにより、pron2mora() が認識するモーラ集合と feature table の前提を一貫させやすくする
## (カタカナ, 子音, 母音) の順で定義し、子音がない場合は None を入れる
## 但し「ン」と「ッ」は母音のみという扱いで、「ン」は「N」、「ッ」は「cl」とする
MORA_LIST_MINIMUM: list[tuple[str, str | None, str]] = [
    ("ヴォ", "v", "o"),
    ("ヴェ", "v", "e"),
    ("ヴィ", "v", "i"),
    ("ヴァ", "v", "a"),
    ("ヴ", "v", "u"),
    ("ン", None, "N"),
    ("ワ", "w", "a"),
    ("ロ", "r", "o"),
    ("レ", "r", "e"),
    ("ル", "r", "u"),
    ("リョ", "ry", "o"),
    ("リュ", "ry", "u"),
    ("リャ", "ry", "a"),
    ("リェ", "ry", "e"),
    ("リ", "r", "i"),
    ("ラ", "r", "a"),
    ("ヨ", "y", "o"),
    ("ユ", "y", "u"),
    ("ヤ", "y", "a"),
    ("モ", "m", "o"),
    ("メ", "m", "e"),
    ("ム", "m", "u"),
    ("ミョ", "my", "o"),
    ("ミュ", "my", "u"),
    ("ミャ", "my", "a"),
    ("ミェ", "my", "e"),
    ("ミ", "m", "i"),
    ("マ", "m", "a"),
    ("ポ", "p", "o"),
    ("ボ", "b", "o"),
    ("ホ", "h", "o"),
    ("ペ", "p", "e"),
    ("ベ", "b", "e"),
    ("ヘ", "h", "e"),
    ("プ", "p", "u"),
    ("ブ", "b", "u"),
    ("フュ", "fy", "u"),  # pyopenjtalk-plus で追加されたモーラ
    ("フォ", "f", "o"),
    ("フェ", "f", "e"),
    ("フィ", "f", "i"),
    ("ファ", "f", "a"),
    ("フ", "f", "u"),
    ("ピョ", "py", "o"),
    ("ピュ", "py", "u"),
    ("ピャ", "py", "a"),
    ("ピェ", "py", "e"),
    ("ピ", "p", "i"),
    ("ビョ", "by", "o"),
    ("ビュ", "by", "u"),
    ("ビャ", "by", "a"),
    ("ビェ", "by", "e"),
    ("ビ", "b", "i"),
    ("ヒョ", "hy", "o"),
    ("ヒュ", "hy", "u"),
    ("ヒャ", "hy", "a"),
    ("ヒェ", "hy", "e"),
    ("ヒ", "h", "i"),
    ("パ", "p", "a"),
    ("バ", "b", "a"),
    ("ハ", "h", "a"),
    ("ノ", "n", "o"),
    ("ネ", "n", "e"),
    ("ヌ", "n", "u"),
    ("ニョ", "ny", "o"),
    ("ニュ", "ny", "u"),
    ("ニャ", "ny", "a"),
    ("ニェ", "ny", "e"),
    ("ニ", "n", "i"),
    ("ナ", "n", "a"),
    ("ドゥ", "d", "u"),
    ("ド", "d", "o"),
    ("トゥ", "t", "u"),
    ("ト", "t", "o"),
    ("デョ", "dy", "o"),
    ("デュ", "dy", "u"),
    ("デャ", "dy", "a"),
    ("デェ", "dy", "e"),  # pyopenjtalk-plus で追加されたモーラ
    ("ディ", "d", "i"),
    ("デ", "d", "e"),
    ("テョ", "ty", "o"),
    ("テュ", "ty", "u"),
    ("テャ", "ty", "a"),
    ("ティ", "t", "i"),
    ("テ", "t", "e"),
    ("ツォ", "ts", "o"),
    ("ツェ", "ts", "e"),
    ("ツィ", "ts", "i"),
    ("ツァ", "ts", "a"),
    ("ツ", "ts", "u"),
    ("ッ", None, "cl"),
    ("チョ", "ch", "o"),
    ("チュ", "ch", "u"),
    ("チャ", "ch", "a"),
    ("チェ", "ch", "e"),
    ("チ", "ch", "i"),
    ("ダ", "d", "a"),
    ("タ", "t", "a"),
    ("ゾ", "z", "o"),
    ("ソ", "s", "o"),
    ("ゼ", "z", "e"),
    ("セ", "s", "e"),
    ("ズィ", "z", "i"),
    ("ズ", "z", "u"),
    ("スィ", "s", "i"),
    ("ス", "s", "u"),
    ("ジョ", "j", "o"),
    ("ジュ", "j", "u"),
    ("ジャ", "j", "a"),
    ("ジェ", "j", "e"),
    ("ジ", "j", "i"),
    ("ショ", "sh", "o"),
    ("シュ", "sh", "u"),
    ("シャ", "sh", "a"),
    ("シェ", "sh", "e"),
    ("シ", "sh", "i"),
    ("ザ", "z", "a"),
    ("サ", "s", "a"),
    ("ゴ", "g", "o"),
    ("コ", "k", "o"),
    ("ゲ", "g", "e"),
    ("ケ", "k", "e"),
    ("グヮ", "gw", "a"),  # pyopenjtalk-plus で追加されたモーラ
    ("グォ", "gw", "o"),  # pyopenjtalk-plus で追加されたモーラ
    ("グェ", "gw", "e"),  # pyopenjtalk-plus で追加されたモーラ
    ("グゥ", "gw", "u"),  # pyopenjtalk-plus で追加されたモーラ
    ("グィ", "gw", "i"),  # pyopenjtalk-plus で追加されたモーラ
    ("グ", "g", "u"),
    ("クヮ", "kw", "a"),  # pyopenjtalk-plus で追加されたモーラ
    ("クォ", "kw", "o"),  # pyopenjtalk-plus で追加されたモーラ
    ("クェ", "kw", "e"),  # pyopenjtalk-plus で追加されたモーラ
    ("クゥ", "kw", "u"),  # pyopenjtalk-plus で追加されたモーラ
    ("クィ", "kw", "i"),  # pyopenjtalk-plus で追加されたモーラ
    ("ク", "k", "u"),
    ("ギョ", "gy", "o"),
    ("ギュ", "gy", "u"),
    ("ギャ", "gy", "a"),
    ("ギェ", "gy", "e"),
    ("ギ", "g", "i"),
    ("キョ", "ky", "o"),
    ("キュ", "ky", "u"),
    ("キャ", "ky", "a"),
    ("キェ", "ky", "e"),
    ("キ", "k", "i"),
    ("ガ", "g", "a"),
    ("カ", "k", "a"),
    ("オ", None, "o"),
    ("エ", None, "e"),
    ("ウォ", "w", "o"),
    ("ウェ", "w", "e"),
    ("ウィ", "w", "i"),
    ("ウ", None, "u"),
    ("イェ", "y", "e"),
    ("イ", None, "i"),
    ("ア", None, "a"),
]

# 上記に定義済みのモーラと同一音素列を表すエイリアス
MORA_LIST_ADDITIONAL: list[tuple[str, str | None, str]] = [
    ("ヴョ", "by", "o"),
    ("ヴュ", "by", "u"),
    ("ヴャ", "by", "a"),
    ("ヲ", None, "o"),
    ("ヱ", None, "e"),
    ("ヰ", None, "i"),
    ("ヮ", "w", "a"),
    ("ョ", "y", "o"),
    ("ュ", "y", "u"),
    ("ヅ", "z", "u"),
    ("ヂョ", "j", "o"),  # pyopenjtalk-plus には存在しないエイリアス
    ("ヂュ", "j", "u"),  # pyopenjtalk-plus には存在しないエイリアス
    ("ヂャ", "j", "a"),  # pyopenjtalk-plus には存在しないエイリアス
    ("ヂェ", "j", "e"),  # pyopenjtalk-plus には存在しないエイリアス
    ("ヂ", "j", "i"),
    ("シィ", "s", "i"),  # pyopenjtalk-plus で追加されたモーラ
    ("グァ", "gw", "a"),  # pyopenjtalk-plus で追加されたモーラのエイリアス
    ("クァ", "kw", "a"),  # pyopenjtalk-plus で追加されたモーラのエイリアス
    ("ヶ", "k", "e"),
    ("ャ", "y", "a"),
    ("ォ", None, "o"),
    ("ェ", None, "e"),
    ("ゥ", None, "u"),
    ("ィ", None, "i"),
    ("ァ", None, "a"),
]

PHON_TABLE = {
    kana: ([consonant] if consonant is not None else []) + [vowel]
    for kana, consonant, vowel in MORA_LIST_MINIMUM + MORA_LIST_ADDITIONAL
}
PHON_TABLE.update(
    {
        ",": [","],
        ".": ["."],
        "?": ["?"],
        "!": ["!"],
        "、": [","],
        "。": ["."],
        "？": ["?"],
        "！": ["!"],
    }
)

SUPPORTED_MORA = set(PHON_TABLE.keys())
CANONICAL_MORA_BY_PHONEMES = {
    tuple(([consonant] if consonant is not None else []) + [vowel]): kana
    for kana, consonant, vowel in MORA_LIST_MINIMUM
}


def get_phoneme(mora: str, current_phonemes: list[str]) -> list[str]:
    """
    Convert mora(single or double Katakana characters) to Phoneme
    """

    # If the current mora is a long-vowel symbol, add a vowel to the previous phoneme
    # e.g., ワー -> w + a + a -> w + aa
    if current_phonemes and mora == LONGVOWEL_CHARACTER:
        # cl (i.e., ッ) should not be copied from long-vowel (coco #28)
        if current_phonemes[-1] != "cl":
            current_phonemes[-1] = f"{current_phonemes[-1]}{current_phonemes[-1][-1]}"
        phoneme = []
    else:
        try:
            phoneme = PHON_TABLE[mora]
        except KeyError:
            raise ValueError(f"Not supported mora : {mora}")

    return phoneme
