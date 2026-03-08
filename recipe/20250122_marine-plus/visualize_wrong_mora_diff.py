import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from marine.utils.openjtalk_util import print_diff_hl


def get_parser() -> argparse.ArgumentParser:
    """
    `wrong_mora_info.csv` の可視化用 CLI 引数を生成する。

    Returns:
        argparse.ArgumentParser: 引数パーサー
    """

    parser = argparse.ArgumentParser(
        description="Visualize diff between OpenJTalk mora output and manual annotation.",
    )
    parser.add_argument(
        "wrong_mora_info_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to wrong_mora_info.csv. If omitted, discover it from the recipe directory.",
    )
    parser.add_argument(
        "--text-yaml-path",
        type=Path,
        default=None,
        help="Path to text.yaml. Defaults to data/text.yaml under this recipe.",
    )
    return parser


def discover_wrong_mora_info_path(recipe_dir: Path) -> Path:
    """
    レシピ配下から `wrong_mora_info.csv` の既定位置を探索する。

    Args:
        recipe_dir (Path): 対象レシピのディレクトリ

    Returns:
        Path: 発見した `wrong_mora_info.csv` のパス

    Raises:
        FileNotFoundError: `wrong_mora_info.csv` を発見できなかった場合
    """

    candidate_paths: list[Path] = []

    legacy_path = recipe_dir / "wrong_mora_info.csv"
    if legacy_path.exists() is True:
        candidate_paths.append(legacy_path)

    feature_pack_paths = sorted(
        recipe_dir.glob("outputs/*/feature_pack/wrong_mora_info.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidate_paths.extend(feature_pack_paths)

    if len(candidate_paths) == 0:
        searched_locations = [
            recipe_dir / "wrong_mora_info.csv",
            recipe_dir / "outputs" / "*" / "feature_pack" / "wrong_mora_info.csv",
        ]
        raise FileNotFoundError(
            "wrong_mora_info.csv was not found. "
            f"Searched locations: {searched_locations}",
        )

    return candidate_paths[0]


def load_text_data(text_yaml_path: Path) -> dict[str, dict[str, Any]]:
    """
    `text.yaml` を読み込み、script id ごとのテキスト辞書へ変換する。

    Args:
        text_yaml_path (Path): `text.yaml` のパス

    Returns:
        dict[str, dict[str, Any]]: script id をキーとする辞書
    """

    with open(text_yaml_path, encoding="utf-8") as file:
        loaded_text_data = yaml.safe_load(file)

    if loaded_text_data is None:
        return {}

    return loaded_text_data


def iter_wrong_mora_rows(wrong_mora_info_path: Path) -> list[dict[str, str]]:
    """
    `wrong_mora_info.csv` の各行を辞書として読み込む。

    Args:
        wrong_mora_info_path (Path): `wrong_mora_info.csv` のパス

    Returns:
        list[dict[str, str]]: `wav`, `jtalk`, `annotation` を持つ辞書の一覧
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


def visualize_wrong_mora_diff(
    wrong_mora_info_path: Path,
    text_yaml_path: Path,
) -> None:
    """
    `wrong_mora_info.csv` と `text.yaml` を使って mora 差分を標準出力へ表示する。

    Args:
        wrong_mora_info_path (Path): `wrong_mora_info.csv` のパス
        text_yaml_path (Path): `text.yaml` のパス
    """

    text_data = load_text_data(text_yaml_path)
    wrong_mora_rows = iter_wrong_mora_rows(wrong_mora_info_path)

    for wrong_mora_row in wrong_mora_rows:
        wav_id = wrong_mora_row["wav"]
        print(f"\n========== {wav_id} ==========")

        text_item = text_data.get(wav_id)
        if text_item is not None and "text_level0" in text_item:
            print(f"       Text : {text_item['text_level0']}")

        print_diff_hl(wrong_mora_row["jtalk"], wrong_mora_row["annotation"])


def main() -> None:
    """
    `wrong_mora_info.csv` 可視化 CLI のエントリーポイント。
    """

    parser = get_parser()
    args = parser.parse_args()
    recipe_dir = Path(__file__).resolve().parent

    wrong_mora_info_path = args.wrong_mora_info_path
    if wrong_mora_info_path is None:
        wrong_mora_info_path = discover_wrong_mora_info_path(recipe_dir)

    text_yaml_path = args.text_yaml_path
    if text_yaml_path is None:
        text_yaml_path = recipe_dir / "data" / "text.yaml"

    visualize_wrong_mora_diff(
        wrong_mora_info_path=wrong_mora_info_path,
        text_yaml_path=text_yaml_path,
    )


if __name__ == "__main__":
    main()
