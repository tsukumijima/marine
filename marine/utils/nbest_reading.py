import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast


def get_unidic_dicdir() -> Path | None:
    """
    n-best 読み探索で利用する UniDic 辞書ディレクトリを返す。

    Returns:
        Path | None: `fugashi-plus` が参照する UniDic 辞書ディレクトリ。取得できない場合は None。
    """

    env_dictionary_dir = os.environ.get("MARINE_UNIDIC_DIR")
    if env_dictionary_dir is not None and env_dictionary_dir != "":
        dictionary_dir = Path(env_dictionary_dir)
        if (dictionary_dir / "sys.dic").exists() is True:
            return dictionary_dir

    try:
        import unidic
    except ImportError:
        return None

    dictionary_dir = Path(unidic.DICDIR)
    if (dictionary_dir / "sys.dic").exists() is not True:
        return None

    return dictionary_dir


@lru_cache(maxsize=1)
def _get_tagger() -> Any | None:
    """
    n-best 読み候補取得用の fugashi-plus Tagger を初期化する。

    Returns:
        Any | None: 初期化済みの GenericTagger。初期化に失敗した場合は None。
    """

    try:
        from fugashi import GenericTagger
    except ImportError:
        return None

    dictionary_dir = get_unidic_dicdir()
    if dictionary_dir is None:
        return None

    resource_file = dictionary_dir / "mecabrc"
    if resource_file.exists() is not True:
        resource_file = dictionary_dir / "dicrc"

    try:
        return GenericTagger(f"-r {resource_file} -d {dictionary_dir}")
    except RuntimeError:
        return None


def _extract_pronunciation(feature: tuple[str, ...]) -> str | None:
    """
    UniDic 由来の feature tuple から pronunciation を取り出す。

    Args:
        feature (tuple[str, ...]): fugashi-plus が返す feature tuple

    Returns:
        str | None: pronunciation。取得できない場合は None。
    """

    for feature_index in (9, 6):
        if len(feature) <= feature_index:
            continue

        pronunciation = feature[feature_index]
        if pronunciation not in {"*", ""}:
            return pronunciation

    return None


@lru_cache(maxsize=4096)
def get_nbest_pronunciation_candidates(
    surface: str,
    max_paths: int = 32,
) -> tuple[str, ...]:
    """
    与えられた表層形に対する n-best pronunciation 候補を返す。

    Args:
        surface (str): 表層形
        max_paths (int): 探索する n-best の上限

    Returns:
        tuple[str, ...]: pronunciation 候補のタプル
    """

    if surface == "":
        return tuple()

    tagger = _get_tagger()
    if tagger is None:
        return tuple()

    candidates: set[str] = set()
    try:
        for nodes in tagger.nbestToNodeList(surface, num=max_paths):
            pronunciations: list[str] = []
            for node in nodes:
                feature = tuple(cast(str, value) for value in node.feature)
                pronunciation = _extract_pronunciation(feature)
                if pronunciation is None:
                    pronunciations = []
                    break
                pronunciations.append(pronunciation)

            if pronunciations:
                candidates.add(
                    "".join(pronunciations)
                    .replace("ヲ", "オ")
                    .replace("ヅ", "ズ")
                    .replace("ヂ", "ジ")
                )
    except RuntimeError:
        return tuple()

    return tuple(sorted(candidates))
