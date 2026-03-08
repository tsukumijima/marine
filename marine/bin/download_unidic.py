import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm


NINJAL_BASE_URL = "https://clrd.ninjal.ac.jp/unidic_archive"
UNIDIC_VERSIONS: dict[str, tuple[str, str]] = {
    "csj-202512": ("CSJ 202512", f"{NINJAL_BASE_URL}/2512/unidic-csj-202512.zip"),
    "csj-202302": ("CSJ 202302", f"{NINJAL_BASE_URL}/2302/unidic-csj-202302.zip"),
    "csj-3.1.1": ("CSJ 3.1.1", f"{NINJAL_BASE_URL}/csj/3.1.1/unidic-csj-3.1.1.zip"),
    "csj-3.1.0": ("CSJ 3.1.0", f"{NINJAL_BASE_URL}/csj/3.1.0/unidic-csj-3.1.0.zip"),
    "csj-3.0.1.1": (
        "CSJ 3.0.1.1",
        f"{NINJAL_BASE_URL}/csj/3.0.1.1/unidic-csj-3.0.1.1.zip",
    ),
    "csj-3.0.1": ("CSJ 3.0.1", f"{NINJAL_BASE_URL}/csj/3.0.1/unidic-csj-3.0.1.zip"),
    "cwj-202512": ("CWJ 202512", f"{NINJAL_BASE_URL}/2512/unidic-cwj-202512.zip"),
    "cwj-202302": ("CWJ 202302", f"{NINJAL_BASE_URL}/2302/unidic-cwj-202302.zip"),
    "cwj-3.1.1": ("CWJ 3.1.1", f"{NINJAL_BASE_URL}/cwj/3.1.1/unidic-cwj-3.1.1.zip"),
    "cwj-3.1.0": ("CWJ 3.1.0", f"{NINJAL_BASE_URL}/cwj/3.1.0/unidic-cwj-3.1.0.zip"),
}
DEFAULT_UNIDIC_VERSION = "csj-202512"


def get_parser() -> argparse.ArgumentParser:
    """
    UniDic ダウンローダーの引数パーサーを生成する。

    Returns:
        argparse.ArgumentParser: 初期化済みの引数パーサー
    """

    parser = argparse.ArgumentParser(
        description="Download NINJAL UniDic archives for marine-plus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default=DEFAULT_UNIDIC_VERSION,
        help="UniDic version key to install",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available UniDic versions and exit",
    )
    return parser


def _download_archive(download_url: str, archive_path: Path) -> None:
    """
    指定 URL から UniDic アーカイブをダウンロードする。

    Args:
        download_url (str): ダウンロード URL
        archive_path (Path): 保存先アーカイブパス
    """

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() is True:
        print(f"Archive already exists. path: {archive_path}")
        return

    try:
        with urllib.request.urlopen(download_url, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", "0"))
            with archive_path.open("wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=archive_path.name,
                ) as progress_bar:
                    while True:
                        chunk = response.read(8192)
                        if chunk == b"":
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
    except (TimeoutError, urllib.error.URLError) as ex:
        if archive_path.exists() is True:
            archive_path.unlink()
        raise RuntimeError(
            "Failed to download UniDic archive due to a network error. "
            f"url: {download_url}"
        ) from ex


def _resolve_unidic_install_paths() -> tuple[Path, Path]:
    """
    `unidic` パッケージの辞書配置先と作業ディレクトリを返す。

    Returns:
        tuple[Path, Path]: `dicdir` と、その親ディレクトリのタプル
    """

    try:
        import unidic
    except ImportError as ex:
        raise RuntimeError(
            "The 'unidic' package is not installed. "
            "Install it with 'pip install unidic' or add it to your requirements "
            "before running this script."
        ) from ex

    dictionary_dir = Path(unidic.DICDIR)
    package_dir = dictionary_dir.parent
    package_dir.mkdir(parents=True, exist_ok=True)
    return dictionary_dir, package_dir


def _find_extracted_dictionary_dir(extract_root: Path) -> Path | None:
    """
    展開済みディレクトリ以下から UniDic 本体ディレクトリを探索する。

    Args:
        extract_root (Path): ZIP 展開先ルート

    Returns:
        Path | None: `sys.dic` を含む辞書ディレクトリ。見つからない場合は None。
    """

    if (extract_root / "sys.dic").exists() is True:
        return extract_root

    for candidate_dir in extract_root.rglob("*"):
        if (
            candidate_dir.is_dir() is True
            and (candidate_dir / "sys.dic").exists() is True
        ):
            return candidate_dir

    return None


def _extract_archive(archive_path: Path, install_dir: Path) -> None:
    """
    UniDic アーカイブを展開し、`unidic` パッケージの `dicdir` へ配置する。

    Args:
        archive_path (Path): ダウンロード済みの ZIP ファイル
        install_dir (Path): 展開後のインストール先ディレクトリ
    """

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    extract_root = install_dir.parent / "_unidic_extract_temp"
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        import zipfile_inflate64 as zipfile_module
    except ImportError:
        import zipfile as zipfile_module

    with zipfile_module.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(str(extract_root))

    source_dir = _find_extracted_dictionary_dir(extract_root)
    if source_dir is None:
        raise RuntimeError(
            f"Failed to find sys.dic in extracted archive. path: {archive_path}"
        )

    backup_dir = install_dir.parent / "_dicdir_backup"
    if backup_dir.exists() is True:
        shutil.rmtree(backup_dir)
    if install_dir.exists() is True:
        install_dir.rename(backup_dir)

    if source_dir == extract_root:
        extract_root.rename(install_dir)
    else:
        shutil.move(str(source_dir), str(install_dir))

    if (
        extract_root.exists() is True
        and extract_root != install_dir
        and extract_root != source_dir
    ):
        shutil.rmtree(str(extract_root))

    mecabrc_path = install_dir / "mecabrc"
    dicrc_path = install_dir / "dicrc"
    if mecabrc_path.exists() is not True and dicrc_path.exists() is True:
        mecabrc_path.write_text(
            dicrc_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if backup_dir.exists() is True:
        shutil.rmtree(backup_dir)


def _print_available_versions() -> None:
    """
    利用可能な UniDic バージョン一覧を表示する。
    """

    print("Available UniDic versions:")
    for version_key, (label, download_url) in UNIDIC_VERSIONS.items():
        default_suffix = " (default)" if version_key == DEFAULT_UNIDIC_VERSION else ""
        print(f"  {version_key:<20} {label}{default_suffix}")
        print(f"    {download_url}")


def entry(argv: list[str] = sys.argv) -> None:
    """
    UniDic ダウンロード CLI のエントリポイント。

    Args:
        argv (list[str]): コマンドライン引数
    """

    args = get_parser().parse_args(argv[1:])
    if args.list is True:
        _print_available_versions()
        return

    version = args.version.lower()
    if version not in UNIDIC_VERSIONS:
        raise SystemExit(f"Unknown UniDic version: {version}")

    label, download_url = UNIDIC_VERSIONS[version]
    install_dir, package_dir = _resolve_unidic_install_paths()
    archive_path = package_dir / "unidic-download.zip"

    print(f"Selected version: {label}")
    print(f"Download URL: {download_url}")
    print(f"Archive path: {archive_path}")
    print(f"Install path: {install_dir}")

    _download_archive(download_url, archive_path)
    _extract_archive(archive_path, install_dir)
    (install_dir / "version").write_text(version, encoding="utf-8")

    print("UniDic download completed.")
    print(f"Installed version: {version}")
    print(f"Dictionary dir: {install_dir}")


if __name__ == "__main__":
    entry()
