import json
import logging
import random
import sys
import time
from pathlib import Path
from shutil import copyfile
from typing import Any, cast

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from marine.bin.test import test_model
from marine.data.feature.feature_set import FeatureSet
from marine.data.util import load_dataset
from marine.logger import getLogger
from marine.models import (
    AttentionBasedLSTMDecoder,
    CRFDecoder,
    LinearDecoder,
    init_model,
)
from marine.models.base_model import BaseModel
from marine.utils.metrics import MultiTaskMetrics
from marine.utils.openjtalk_metrics import OpenJTalkPracticalMetrics
from marine.utils.util import (
    init_seed,
    log_scores,
    pack_inputs,
    pack_outputs,
    pad_incomplete_accent_logits,
    plot_batch_attention,
)


DecoderModule = CRFDecoder | LinearDecoder | AttentionBasedLSTMDecoder
logger: logging.Logger | None = None


def _flatten_metric_scores(
    phase: str,
    loss: dict[str, float],
    task_scores: dict[str, dict[str, float]],
    extra_scores: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """
    checkpoint 選定用に評価値をフラットな辞書へ展開する。

    Args:
        phase (str): 評価フェーズ名
        loss (dict[str, float]): タスクごとの損失
        task_scores (dict[str, dict[str, float]]): `MultiTaskMetrics` の集計結果
        extra_scores (dict[str, dict[str, float]] | None): 追加評価指標

    Returns:
        dict[str, float]: `phase/task/metric` 形式で展開したスカラー値
    """

    flattened_scores = {
        f"{phase}/loss": sum(loss.values()),
    }

    for task, metric_scores in task_scores.items():
        for metric_name, score in metric_scores.items():
            flattened_scores[f"{phase}/{task}/{metric_name}"] = score

    if extra_scores is not None:
        for score_group, metric_scores in extra_scores.items():
            for metric_name, score in metric_scores.items():
                flattened_scores[f"{phase}/{score_group}/{metric_name}"] = score

    return flattened_scores


def _format_metric_value(metric_name: str, metric_value: float) -> str:
    """
    ログ表示用に評価値を整形する。

    Args:
        metric_name (str): 評価指標名
        metric_value (float): 評価値

    Returns:
        str: 表示用の文字列
    """

    if metric_name.endswith("/loss"):
        return f"{metric_value:.4f}"

    return f"{metric_value:.4%}"


def _is_better_metric(
    current_metric_value: float,
    best_metric_value: float,
    metric_mode: str,
) -> bool:
    """
    現在の評価値が既存ベストを更新したかどうかを判定する。

    Args:
        current_metric_value (float): 今回の評価値
        best_metric_value (float): 既存ベストの評価値
        metric_mode (str): `min` または `max`

    Returns:
        bool: ベスト更新時は `True`
    """

    if metric_mode == "min":
        return current_metric_value < best_metric_value

    if metric_mode == "max":
        return current_metric_value > best_metric_value

    raise ValueError(f"Unsupported best metric mode: {metric_mode}")


def _resolve_scheduler_metric_name(config: DictConfig) -> str:
    """
    学習率 scheduler の更新に使う評価指標名を返す。

    Args:
        config (DictConfig): 学習設定

    Returns:
        str: scheduler 更新に使う評価指標名
    """

    scheduler_metric_name = OmegaConf.select(
        config,
        "train.scheduler_metric",
        default=None,
    )
    if scheduler_metric_name is None:
        return "val/loss"

    return str(scheduler_metric_name)


def _resolve_secondary_best_metric_name(config: DictConfig) -> str | None:
    """
    checkpoint 選定の副指標名を返す。

    Args:
        config (DictConfig): 学習設定

    Returns:
        str | None: 副指標名。未設定時は `None`
    """

    secondary_best_metric_name = OmegaConf.select(
        config,
        "train.secondary_best_metric",
        default=None,
    )
    if secondary_best_metric_name is None:
        return None

    return str(secondary_best_metric_name)


def _resolve_secondary_best_metric_mode(config: DictConfig) -> str:
    """
    checkpoint 選定の副指標モードを返す。

    Args:
        config (DictConfig): 学習設定

    Returns:
        str: `min` または `max`
    """

    secondary_best_metric_mode = OmegaConf.select(
        config,
        "train.secondary_best_metric_mode",
        default=None,
    )
    if secondary_best_metric_mode is None:
        return str(config.train.best_metric_mode)

    return str(secondary_best_metric_mode)


def _validate_metric_availability(
    flattened_scores: dict[str, float],
    metric_name: str,
    metric_role: str,
) -> None:
    """
    指定された評価指標が集計結果に含まれているかを検証する。

    Args:
        flattened_scores (dict[str, float]): 集計済みの評価値
        metric_name (str): 検証対象の評価指標名
        metric_role (str): エラーメッセージ用の役割名
    """

    if metric_name in flattened_scores:
        return

    available_metric_names = ", ".join(sorted(flattened_scores.keys()))
    raise KeyError(
        f"Configured {metric_role} is not available. "
        f"{metric_role}: {metric_name}, "
        f"available_metrics: {available_metric_names}"
    )


def _compare_metric_values(
    current_metric_value: float,
    best_metric_value: float,
    metric_mode: str,
) -> int:
    """
    評価値同士の優劣を比較する。

    Args:
        current_metric_value (float): 今回の評価値
        best_metric_value (float): 既存ベストの評価値
        metric_mode (str): `min` または `max`

    Returns:
        int: 改善時は `1`、同値時は `0`、劣化時は `-1`
    """

    if torch.isclose(
        torch.tensor(current_metric_value),
        torch.tensor(best_metric_value),
    ):
        return 0

    if _is_better_metric(current_metric_value, best_metric_value, metric_mode) is True:
        return 1

    return -1


def save_checkpoint(
    config: DictConfig,
    checkpoint_dir: Path,
    model: nn.Module | None,
    optimizer: Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    is_best: bool,
) -> None:
    assert logger is not None
    if model is None:
        return

    # save config file
    if epoch == 0:
        if not config.train.save_vocab_path:
            config.model.vocab_path = None

        config_path = checkpoint_dir / "config.yaml"
        logger.info(f"Save config: {config_path}")

        OmegaConf.save(config, config_path)

    is_interval = epoch > 0 and (epoch + 1) % config.train.checkpoint_interval == 0

    # save checkpoint when epoch is interval
    if is_interval or is_best:
        states = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer": (
                optimizer.state_dict()
                if config.train.save_optimizer_state and optimizer is not None
                else None
            ),
            "scheduler": (
                scheduler.state_dict()
                if config.train.save_optimizer_state and scheduler is not None
                else None
            ),
        }

        if is_interval and is_best:
            interval_checkpoint_path = checkpoint_dir / f"epoch_{epoch:05d}.pth"
            best_checkpoint_path = checkpoint_dir / "best.pth"
            lastest_path = checkpoint_dir / "latest.pth"
            logger.info(f"Save interval checkpoint: {interval_checkpoint_path}")
            logger.info(f"Save best checkpoint: {best_checkpoint_path}")
            torch.save(states, interval_checkpoint_path)
            copyfile(interval_checkpoint_path, best_checkpoint_path)
            copyfile(interval_checkpoint_path, lastest_path)

        elif is_interval:
            interval_checkpoint_path = checkpoint_dir / f"epoch_{epoch:05d}.pth"
            lastest_path = checkpoint_dir / "latest.pth"
            logger.info(f"Save interval checkpoint: {interval_checkpoint_path}")
            torch.save(states, interval_checkpoint_path)
            copyfile(interval_checkpoint_path, lastest_path)

        else:
            best_checkpoint_path = checkpoint_dir / "best.pth"
            logger.info(f"Save best checkpoint: {best_checkpoint_path}")
            torch.save(states, best_checkpoint_path)


def train_model(
    model: BaseModel,
    criterions: dict[str, Any],
    optimizer: Optimizer,
    scheduler: Any,
    dataloader: dict[str, Any],
    tasks: list[str],
    tensorboard_writer: SummaryWriter | None,
    checkpoint_dir: Path,
    config: DictConfig,
    feature_set: FeatureSet,
    num_epochs: int = 10,
    device: str = "cpu",
) -> None:
    assert logger is not None
    phases = ["train", "val"]
    best_metric_mode = config.train.best_metric_mode
    best_metric_value = float("inf") if best_metric_mode == "min" else float("-inf")
    secondary_best_metric_name = _resolve_secondary_best_metric_name(config)
    secondary_best_metric_mode = _resolve_secondary_best_metric_mode(config)
    secondary_best_metric_value = (
        float("inf") if secondary_best_metric_mode == "min" else float("-inf")
    )
    max_grad_norm = OmegaConf.select(
        config,
        "train.max_grad_norm",
        default=None,
    )
    early_stopping_patience = OmegaConf.select(
        config,
        "train.early_stopping_patience",
        default=None,
    )
    early_stopping_min_epochs = OmegaConf.select(
        config,
        "train.early_stopping_min_epochs",
        default=0,
    )
    epochs_without_improvement = 0
    fig_logging_targets = random.choices(range(config.data.batch_size), k=10)

    if "accent_status" in tasks:
        has_att_based_model = isinstance(
            cast(DecoderModule, model.decoders["accent_status"]),
            AttentionBasedLSTMDecoder,
        )
    else:
        has_att_based_model = False

    for epoch in range(num_epochs):
        logger.info("-" * 10)
        logger.info(f"Epoch {epoch}/{num_epochs - 1}")
        logger.info("-" * 10)

        since = time.time()

        for phase in phases:
            is_train = phase == "train"
            phase_dataloader: Any = dataloader[phase]

            if is_train:
                model.train()
            else:
                model.eval()

            # for logging
            running_loss = {task: 0.0 for task in tasks}
            metrics = MultiTaskMetrics(
                phase,
                config.data.output_sizes,
                accent_represent_mode=config.data.represent_mode,
                require_ap_level_f1_score=has_att_based_model,
                device=device,
            )
            openjtalk_metrics = OpenJTalkPracticalMetrics(
                accent_represent_mode=config.data.represent_mode,
            )

            for batch_index, (
                inputs,
                outputs,
                morph_boundaries,
                script_ids,
            ) in enumerate(tqdm(phase_dataloader, desc=f"{phase}: ", leave=False)):
                # pack inputs to device
                inputs = pack_inputs(inputs, config.data.input_keys, device)
                outputs = pack_outputs(outputs, device)

                # total loss of the tasks on a single batch
                batch_loss: torch.Tensor | None = None
                batch_predicts: dict[str, torch.Tensor] = {}
                prev_decoder_output: dict[str, torch.Tensor] = {}

                optimizer.zero_grad()

                for index, task in enumerate(tasks):
                    output, output_mask = outputs[task]["label"], outputs[task]["mask"]

                    # predict
                    with torch.set_grad_enabled(is_train):
                        decoder_outputs = model(task, **inputs)

                        decoder = cast(DecoderModule, model.decoders[task])
                        if isinstance(decoder, CRFDecoder):
                            linear_logits, crf_logits = decoder_outputs
                            try:
                                loss = criterions[task](
                                    linear_logits, output, output_mask
                                )
                            except Exception:
                                print(script_ids)
                                sys.exit(1)
                            logits = crf_logits
                        elif isinstance(decoder, LinearDecoder):
                            logits = decoder_outputs
                            loss = criterions[task](logits, output, output_mask)
                        else:
                            logits, attentions, ap_lengths = decoder_outputs
                            # plot attention when first batch on eval
                            if (
                                not is_train
                                and tensorboard_writer
                                and batch_index == 0
                                and task == "accent_status"
                            ):
                                plot_batch_attention(
                                    inputs,
                                    logits,
                                    ap_lengths,
                                    attentions,
                                    feature_set,
                                    plot_targets=fig_logging_targets,
                                    tensorboard_writer=tensorboard_writer,
                                    phase=phase,
                                    epoch=epoch,
                                    script_ids=script_ids,
                                )

                            # resize logits
                            logits = pad_incomplete_accent_logits(logits, output_mask)
                            loss = criterions[task](logits, output, output_mask)

                        batch_loss = loss if batch_loss is None else batch_loss + loss
                        running_loss[task] += float(loss.item())

                        # logits: (B, T, dim) -> (B, T)
                        predicts = torch.argmax(logits, dim=2)
                        batch_predicts[task] = predicts

                        # Update metrics
                        # for ap-based seq
                        if task == "accent_status" and has_att_based_model:
                            metrics.update(
                                task,
                                predicts,
                                output,
                                output_mask,
                                predicted_accent_phrase_boundaries=inputs[
                                    "prev_decoder_outputs"
                                ]["accent_phrase_boundary"],
                                target_accent_phrase_boundaries=outputs[
                                    "accent_phrase_boundary"
                                ]["label"],
                                mora_seq_masks=outputs["accent_phrase_boundary"][
                                    "mask"
                                ],
                            )

                        # for mora-based seq
                        else:
                            try:
                                metrics.update(task, predicts, output, output_mask)
                            except Exception:
                                print(script_ids)
                                sys.exit(-1)

                        # cascade prev outputs
                        if index < len(tasks) - 1:
                            prev_decoder_output[task] = predicts

                            if is_train:
                                # Use golden label as teacher forcing
                                inputs["prev_decoder_outputs"][task] = output

                                if (
                                    index + 1 < len(tasks)
                                    and tasks[index + 1] == "accent_status"
                                    and has_att_based_model
                                ):
                                    inputs["decoder_targets"] = outputs[
                                        "accent_status"
                                    ]["label"]
                            else:
                                _output = prev_decoder_output[task]
                                inputs["prev_decoder_outputs"][task] = _output
                                inputs["decoder_targets"] = None

                openjtalk_metrics.update(
                    predicts=batch_predicts,
                    outputs=outputs,
                    morph_boundaries=morph_boundaries,
                    is_ap_based_accent_status=has_att_based_model,
                )

                if is_train and batch_loss is not None:
                    batch_loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=max_grad_norm,
                        )
                    optimizer.step()

            epoch_loss = {
                task: running_loss[task] / len(dataloader[phase]) for task in tasks
            }
            openjtalk_score_group = {
                "openjtalk_compatible": openjtalk_metrics.compute(),
            }

            # Logging scores
            log_scores(
                phase,
                epoch,
                tasks,
                metrics,
                loss=epoch_loss,
                extra_scores=openjtalk_score_group,
                tensorboard_writer=tensorboard_writer,
            )

            if not is_train:
                task_scores = metrics.compute()
                flattened_scores = _flatten_metric_scores(
                    phase,
                    epoch_loss,
                    task_scores,
                    openjtalk_score_group,
                )

                if isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    scheduler_metric_name = _resolve_scheduler_metric_name(config)
                    _validate_metric_availability(
                        flattened_scores,
                        scheduler_metric_name,
                        "scheduler_metric",
                    )

                    scheduler_metric_value = flattened_scores[scheduler_metric_name]
                    scheduler.step(scheduler_metric_value)
                    logger.info(
                        "val / scheduler | "
                        f"metric : {scheduler_metric_name} | "
                        f"score : {_format_metric_value(scheduler_metric_name, scheduler_metric_value)}"
                    )
                else:
                    scheduler.step()

                epoch_lr = optimizer.param_groups[0]["lr"]
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(
                        f"{phase}_learning_rate", epoch_lr, epoch
                    )

                best_metric_name = config.train.best_metric

                _validate_metric_availability(
                    flattened_scores,
                    best_metric_name,
                    "best_metric",
                )

                current_best_metric_value = flattened_scores[best_metric_name]
                logger.info(
                    "val / checkpoint_selection | "
                    f"metric : {best_metric_name} | "
                    f"mode : {best_metric_mode} | "
                    f"score : {_format_metric_value(best_metric_name, current_best_metric_value)}"
                )

                current_secondary_best_metric_value: float | None = None
                if secondary_best_metric_name is not None:
                    _validate_metric_availability(
                        flattened_scores,
                        secondary_best_metric_name,
                        "secondary_best_metric",
                    )
                    current_secondary_best_metric_value = flattened_scores[
                        secondary_best_metric_name
                    ]
                    logger.info(
                        "val / checkpoint_selection | "
                        f"secondary_metric : {secondary_best_metric_name} | "
                        f"mode : {secondary_best_metric_mode} | "
                        f"score : {_format_metric_value(secondary_best_metric_name, current_secondary_best_metric_value)}"
                    )

                # Save checkpoints
                primary_metric_comparison = _compare_metric_values(
                    current_best_metric_value,
                    best_metric_value,
                    best_metric_mode,
                )
                is_best = primary_metric_comparison > 0

                if (
                    is_best is False
                    and primary_metric_comparison == 0
                    and secondary_best_metric_name is not None
                    and current_secondary_best_metric_value is not None
                ):
                    secondary_metric_comparison = _compare_metric_values(
                        current_secondary_best_metric_value,
                        secondary_best_metric_value,
                        secondary_best_metric_mode,
                    )
                    is_best = secondary_metric_comparison > 0

                if is_best is True:
                    best_metric_value = current_best_metric_value
                    epochs_without_improvement = 0
                    if current_secondary_best_metric_value is not None:
                        secondary_best_metric_value = (
                            current_secondary_best_metric_value
                        )
                    logger.info(
                        "val / checkpoint_selection | "
                        f"new best score : {_format_metric_value(best_metric_name, best_metric_value)}"
                    )
                else:
                    epochs_without_improvement += 1
                    logger.info(
                        "val / checkpoint_selection | "
                        f"epochs_without_improvement : {epochs_without_improvement}"
                    )

                save_checkpoint(
                    config,
                    checkpoint_dir,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    is_best,
                )

        time_elapsed = time.time() - since
        logger.info(f"complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        if (
            early_stopping_patience is not None
            and epoch + 1 >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            logger.info(
                "Stop training early. "
                f"epoch : {epoch}, "
                f"early_stopping_patience : {early_stopping_patience}, "
                f"epochs_without_improvement : {epochs_without_improvement}"
            )
            break


@hydra.main(config_path="conf/train", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.train.verbose)
    logger.info(OmegaConf.to_yaml(config))
    assert logger is not None

    init_seed(config.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = config.train.num_epochs

    dataloader = load_dataset(config)
    tasks = config.data.output_keys

    checkpoint_dir = (
        Path(to_absolute_path(config.train.out_dir)) / config.train.model_name
    )

    if not checkpoint_dir.exists():
        logger.info(f"Created checkpoint dir at {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True)

    # Setup tensorboard summary writer
    if config.train.tensorboard_event_path is None:
        tensorboard_event_path = f"tensorboard/{checkpoint_dir.name}"
    else:
        tensorboard_event_path = config.train.tensorboard_event_path
    tensorboard_event_path = Path(to_absolute_path(tensorboard_event_path))

    logger.info(f"TensorBoard event log path: {tensorboard_event_path}")
    tensorboard_writer = SummaryWriter(tensorboard_event_path)

    # Setup test result logging
    if config.train.test_log_dir is not None:
        log_dir = Path(to_absolute_path(config.train.test_log_dir))
    else:
        log_dir = Path(to_absolute_path("logs")) / checkpoint_dir.name

    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    log_path = log_dir / f"{checkpoint_dir.name}_test_log.json"

    # Init feature set
    feature_set = FeatureSet(
        Path(to_absolute_path(config.model.vocab_path)),
        feature_table_key=config.data.feature_table_key,
        feature_keys=config.data.input_keys,
    )

    # Init model
    model, criterions, optimizer, scheduler = init_model(
        tasks, config, feature_set, device, is_train=True
    )

    # Train
    train_model(
        model,
        criterions,
        optimizer,
        scheduler,
        dataloader,
        tasks,
        tensorboard_writer,
        checkpoint_dir,
        config,
        feature_set,
        num_epochs=num_epochs,
        device=device,
    )

    # Init test model
    _test_model = init_model(tasks, config, feature_set, device)
    logs = test_model(
        _test_model,
        checkpoint_dir,
        config.train.test_checkpoint_filename,
        tasks,
        dataloader,
        config,
        feature_set,
        tensorboard_writer=tensorboard_writer,
        device=device,
        logger=logger,
    )

    if config.train.save_test_log:
        # save test result log
        with open(log_path, "w", encoding="utf-8") as file:
            json.dump(
                logs,
                file,
                ensure_ascii=False,
                indent=4,
                separators=(",", ": "),
            )

    tensorboard_writer.close()


def entry() -> None:
    my_app()


if __name__ == "__main__":
    my_app()
