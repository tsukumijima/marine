import argparse
import tarfile
import tempfile
from pathlib import Path
from shutil import copy2
from typing import cast

from omegaconf import DictConfig, OmegaConf


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package a trained model directory into model.tar.gz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Directory containing config.yaml and checkpoint files",
    )
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="best.pth",
        help="Checkpoint filename to package as model.pth",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("model.tar.gz"),
        help="Output tar.gz path",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=None,
        help="Optional explicit path to vocab.pkl",
    )
    return parser


def resolve_vocab_path(
    checkpoint_dir: Path,
    checkpoint_config: DictConfig,
    explicit_vocab_path: Path | None,
) -> Path:
    """
    Package 対象に含める `vocab.pkl` のパスを解決する。

    Args:
        checkpoint_dir (Path): 学習済み checkpoint ディレクトリ
        checkpoint_config (DictConfig): checkpoint の設定
        explicit_vocab_path (Path | None): コマンドラインで明示された `vocab.pkl`

    Returns:
        Path: 使用する `vocab.pkl` のパス

    Raises:
        FileNotFoundError: `vocab.pkl` を解決できない場合
    """

    if explicit_vocab_path is not None:
        if explicit_vocab_path.exists() is False:
            raise FileNotFoundError(
                f"Explicit vocab path not found: {explicit_vocab_path}"
            )
        return explicit_vocab_path

    config_vocab_path = checkpoint_config.model.vocab_path
    if config_vocab_path is not None:
        resolved_vocab_path = Path(str(config_vocab_path))
        if resolved_vocab_path.exists() is True:
            return resolved_vocab_path

    inferred_vocab_path = checkpoint_dir.parent.parent / "vocab" / "vocab.pkl"
    if inferred_vocab_path.exists() is True:
        return inferred_vocab_path

    raise FileNotFoundError(
        "Failed to resolve vocab.pkl. Please specify --vocab_path explicitly."
    )


def package_model(
    checkpoint_dir: Path,
    checkpoint_filename: str,
    output_path: Path,
    vocab_path: Path | None,
) -> Path:
    """
    学習済みモデルを `Predictor` が読み込める `model.tar.gz` にまとめる。

    Args:
        checkpoint_dir (Path): 学習済み checkpoint ディレクトリ
        checkpoint_filename (str): `model.pth` として同梱する checkpoint 名
        output_path (Path): 出力する tar.gz のパス
        vocab_path (Path | None): 明示指定された `vocab.pkl` のパス

    Returns:
        Path: 生成した tar.gz のパス

    Raises:
        FileNotFoundError: 入力ファイルが不足している場合
    """

    if checkpoint_dir.exists() is False:
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_path = checkpoint_dir / checkpoint_filename
    if checkpoint_path.exists() is False:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    config_path = checkpoint_dir / "config.yaml"
    if config_path.exists() is False:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    checkpoint_config = cast(DictConfig, OmegaConf.load(config_path))
    resolved_vocab_path = resolve_vocab_path(
        checkpoint_dir=checkpoint_dir,
        checkpoint_config=checkpoint_config,
        explicit_vocab_path=vocab_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    packaged_config = OmegaConf.create(
        OmegaConf.to_container(checkpoint_config, resolve=True)
    )
    packaged_config.model.vocab_path = None

    with tempfile.TemporaryDirectory(prefix="marine_plus_package_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        staged_model_path = temporary_path / "model.pth"
        staged_config_path = temporary_path / "config.yaml"
        staged_vocab_path = temporary_path / "vocab.pkl"

        copy2(checkpoint_path, staged_model_path)
        copy2(resolved_vocab_path, staged_vocab_path)
        OmegaConf.save(packaged_config, staged_config_path)

        with tarfile.open(output_path, "w:gz") as archive:
            archive.add(staged_model_path, arcname="model.pth")
            archive.add(staged_config_path, arcname="config.yaml")
            archive.add(staged_vocab_path, arcname="vocab.pkl")

    return output_path


def entry() -> None:
    args = get_parser().parse_args()
    package_model(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_filename=args.checkpoint_filename,
        output_path=args.output_path,
        vocab_path=args.vocab_path,
    )
    print(f"Packaged model archive created at: {args.output_path}")


if __name__ == "__main__":
    entry()
