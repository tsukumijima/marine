import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from marine.utils.openjtalk_util import print_diff_hl


RECIPE_NAME_PATTERN = re.compile(r"^[^/]+$")


def get_parser() -> argparse.ArgumentParser:
    """
    `wrong_mora_info.csv` 可視化 CLI の引数を生成する。

    Returns:
        argparse.ArgumentParser: 引数パーサー
    """

    parser = argparse.ArgumentParser(
        description="Visualize diff between OpenJTalk mora output and manual annotation.",
    )
    parser.add_argument(
        "recipe_name",
        type=str,
        help="Recipe name under the recipe directory, or an absolute recipe path.",
    )
    parser.add_argument(
        "--wrong-mora-info-path",
        type=Path,
        default=None,
        help="Optional explicit path to wrong_mora_info.csv.",
    )
    parser.add_argument(
        "--corpus-yaml-path",
        type=Path,
        default=None,
        help="Optional explicit path to corpus.yaml.",
    )
    return parser


def resolve_recipe_dir(recipe_name: str) -> Path:
    """
    recipe 名または絶対パスから recipe ディレクトリを解決する。

    Args:
        recipe_name (str): recipe 名または絶対パス

    Returns:
        Path: 解決した recipe ディレクトリ
    """

    recipe_path = Path(recipe_name)
    if recipe_path.is_absolute() is True:
        return recipe_path

    if RECIPE_NAME_PATTERN.match(recipe_name) is None:
        raise ValueError(f"Invalid recipe name: {recipe_name}")

    return Path(__file__).resolve().parents[2] / "recipe" / recipe_name


def discover_wrong_mora_info_path(recipe_dir: Path) -> Path:
    """
    recipe 配下から `wrong_mora_info.csv` を探索する。

    Args:
        recipe_dir (Path): 対象 recipe ディレクトリ

    Returns:
        Path: 発見した `wrong_mora_info.csv` のパス
    """

    candidate_paths = sorted(
        recipe_dir.glob("outputs/*/feature_pack/wrong_mora_info.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    legacy_path = recipe_dir / "wrong_mora_info.csv"
    if legacy_path.exists() is True:
        candidate_paths.insert(0, legacy_path)

    if len(candidate_paths) == 0:
        raise FileNotFoundError(
            f"wrong_mora_info.csv was not found under the recipe: {recipe_dir}"
        )

    return candidate_paths[0]


def load_corpus_data(corpus_yaml_path: Path) -> dict[str, dict[str, Any]]:
    """
    `corpus.yaml` を script id 辞書として読み込む。

    Args:
        corpus_yaml_path (Path): `corpus.yaml` のパス

    Returns:
        dict[str, dict[str, Any]]: script id をキーとする辞書
    """

    with open(corpus_yaml_path, encoding="utf-8") as file:
        loaded_corpus_data = yaml.safe_load(file)

    if loaded_corpus_data is None:
        return {}

    return dict(loaded_corpus_data)


def iter_wrong_mora_rows(wrong_mora_info_path: Path) -> list[dict[str, str]]:
    """
    `wrong_mora_info.csv` を辞書リストとして読み込む。

    Args:
        wrong_mora_info_path (Path): `wrong_mora_info.csv` のパス

    Returns:
        list[dict[str, str]]: `wav`, `jtalk`, `annotation` を持つ辞書一覧
    """

    rows: list[dict[str, str]] = []

    with open(wrong_mora_info_path, encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="|")
        for row in reader:
            if len(row) != 3:
                continue

            rows.append(
                {
                    "wav": row[0],
                    "jtalk": row[1],
                    "annotation": row[2],
                }
            )

    return rows


def main(argv: list[str] | None = None) -> None:
    """
    CLI のエントリーポイント。

    Args:
        argv (list[str] | None): コマンドライン引数
    """

    parser = get_parser()
    args = parser.parse_args(argv)
    recipe_dir = resolve_recipe_dir(args.recipe_name)

    wrong_mora_info_path = args.wrong_mora_info_path
    if wrong_mora_info_path is None:
        wrong_mora_info_path = discover_wrong_mora_info_path(recipe_dir)

    corpus_yaml_path = args.corpus_yaml_path
    if corpus_yaml_path is None:
        corpus_yaml_path = recipe_dir / "data" / "corpus.yaml"

    corpus_data = load_corpus_data(corpus_yaml_path)
    wrong_mora_rows = iter_wrong_mora_rows(wrong_mora_info_path)

    for wrong_mora_row in wrong_mora_rows:
        wav_id = wrong_mora_row["wav"]
        print(f"\n========== {wav_id} ==========")

        corpus_item = corpus_data.get(wav_id)
        if corpus_item is not None and "text" in corpus_item:
            print(f"       Text : {corpus_item['text']}")

        print_diff_hl(wrong_mora_row["jtalk"], wrong_mora_row["annotation"])


if __name__ == "__main__":
    main(sys.argv[1:])
