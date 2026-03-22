from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from marine.types import (
    AccentRepresentMode,
    MarineLabel,
    MorphBoundaryArray,
    OpenJTalkFormatLabel,
)
from marine.utils.openjtalk_util import convert_open_jtalk_format_label
from marine.utils.util import convert_single_ap_based_accent_to_mora_based_accent


class OpenJTalkPracticalMetrics:
    """
    OpenJTalk 互換の評価指標を集計する。

    学習時の内部表現であるモーラ列 / AP 列をそのまま比較するのではなく、
    実運用で pyopenjtalk へ渡す形に近い形態素単位のラベルへ変換してから評価する。

    現在の `openjtalk_accent_status_sentence_accuracy` は、予測した AP 境界で
    アクセント核位置を形態素単位へ射影したうえでの文単位 exact match である。
    そのため、アクセント核そのものだけでなく AP 境界の誤りの影響も含む。
    AP 境界の影響を切り分けたい場合は `openjtalk_gt_ap_accent_status_morph_accuracy` を使用する。

    Args:
        accent_represent_mode (AccentRepresentMode): アクセント核位置の表現モード
        accent_nucleus_label (int): パディング後のアクセント核ラベル
        accent_phrase_boundary_label (int): パディング後のアクセント句境界ラベル
        morph_boundary_label (int): 形態素境界ラベル
    """

    def __init__(
        self,
        accent_represent_mode: AccentRepresentMode = "binary",
        accent_nucleus_label: int = 2,
        accent_phrase_boundary_label: int = 2,
        morph_boundary_label: int = 1,
    ) -> None:
        self.accent_represent_mode: AccentRepresentMode = accent_represent_mode
        self.accent_nucleus_label = accent_nucleus_label
        self.accent_phrase_boundary_label = accent_phrase_boundary_label
        self.morph_boundary_label = morph_boundary_label
        self.reset()

    def reset(self) -> None:
        """
        集計状態を初期化する。
        """

        self.correct_accent_status_sentences = 0
        self.correct_joint_sentences = 0
        self.correct_gt_ap_accent_status_morphs = 0
        self.correct_joint_morphs = 0
        self.total_morphs = 0
        self.total_sentences = 0

    def update(
        self,
        predicts: dict[str, torch.Tensor],
        outputs: dict[str, dict[str, torch.Tensor]],
        morph_boundaries: Sequence[MorphBoundaryArray],
        is_ap_based_accent_status: bool,
    ) -> None:
        """
        バッチ単位で OpenJTalk 互換の一致数を加算する。

        Args:
            predicts (dict[str, torch.Tensor]): 各タスクの予測ラベル列
            outputs (dict[str, dict[str, torch.Tensor]]): 各タスクの正解ラベル列とマスク
            morph_boundaries (Sequence[MorphBoundaryArray]): 各サンプルの形態素境界
            is_ap_based_accent_status (bool): `accent_status` が AP ベースかどうか
        """

        if (
            "accent_status" not in predicts
            or "accent_phrase_boundary" not in predicts
            or "accent_status" not in outputs
            or "accent_phrase_boundary" not in outputs
        ):
            return

        for sample_index, morph_boundary in enumerate(morph_boundaries):
            predicted_accent_phrase_boundary = self._select_masked_tensor(
                predicts=predicts,
                outputs=outputs,
                task="accent_phrase_boundary",
                sample_index=sample_index,
                is_target=False,
            )
            target_accent_phrase_boundary = self._select_masked_tensor(
                predicts=predicts,
                outputs=outputs,
                task="accent_phrase_boundary",
                sample_index=sample_index,
                is_target=True,
            )
            predicted_accent_status = self._select_masked_tensor(
                predicts=predicts,
                outputs=outputs,
                task="accent_status",
                sample_index=sample_index,
                is_target=False,
            )
            target_accent_status = self._select_masked_tensor(
                predicts=predicts,
                outputs=outputs,
                task="accent_status",
                sample_index=sample_index,
                is_target=True,
            )
            predicted_label = self._convert_to_openjtalk_label(
                accent_status=predicted_accent_status,
                accent_phrase_boundary=predicted_accent_phrase_boundary,
                morph_boundary=morph_boundary,
                is_ap_based_accent_status=is_ap_based_accent_status,
            )
            target_label = self._convert_to_openjtalk_label(
                accent_status=target_accent_status,
                accent_phrase_boundary=target_accent_phrase_boundary,
                morph_boundary=morph_boundary,
                is_ap_based_accent_status=is_ap_based_accent_status,
            )
            gt_ap_predicted_label = self._convert_to_openjtalk_label(
                accent_status=predicted_accent_status,
                accent_phrase_boundary=target_accent_phrase_boundary,
                morph_boundary=morph_boundary,
                is_ap_based_accent_status=is_ap_based_accent_status,
            )

            if predicted_label["accent_status"] == target_label["accent_status"]:
                self.correct_accent_status_sentences += 1

            if predicted_label == target_label:
                self.correct_joint_sentences += 1

            self.correct_gt_ap_accent_status_morphs += self._count_equal_items(
                gt_ap_predicted_label["accent_status"],
                target_label["accent_status"],
            )
            self.correct_joint_morphs += self._count_joint_equal_morphs(
                predicted_label,
                target_label,
            )
            self.total_morphs += len(target_label["accent_status"])
            self.total_sentences += 1

    def compute(self) -> dict[str, float]:
        """
        集計済みの OpenJTalk 互換評価指標を返す。

        Returns:
            dict[str, float]: 文単位 / 形態素単位の評価値
        """

        if self.total_sentences == 0 or self.total_morphs == 0:
            return {
                "openjtalk_accent_status_sentence_accuracy": 0.0,
                "openjtalk_gt_ap_accent_status_morph_accuracy": 0.0,
                "openjtalk_joint_morph_accuracy": 0.0,
                "openjtalk_joint_sentence_accuracy": 0.0,
            }

        return {
            "openjtalk_accent_status_sentence_accuracy": (
                self.correct_accent_status_sentences / self.total_sentences
            ),
            "openjtalk_gt_ap_accent_status_morph_accuracy": (
                self.correct_gt_ap_accent_status_morphs / self.total_morphs
            ),
            "openjtalk_joint_morph_accuracy": (
                self.correct_joint_morphs / self.total_morphs
            ),
            "openjtalk_joint_sentence_accuracy": (
                self.correct_joint_sentences / self.total_sentences
            ),
        }

    def _convert_to_openjtalk_label(
        self,
        accent_status: torch.Tensor,
        accent_phrase_boundary: torch.Tensor,
        morph_boundary: MorphBoundaryArray,
        is_ap_based_accent_status: bool,
    ) -> OpenJTalkFormatLabel:
        """
        単一サンプルのラベル列を OpenJTalk 互換ラベルへ変換する。

        Args:
            accent_status (torch.Tensor): 単一サンプルのアクセント核ラベル列
            accent_phrase_boundary (torch.Tensor): 単一サンプルのアクセント句境界ラベル列
            morph_boundary (MorphBoundaryArray): 形態素境界
            is_ap_based_accent_status (bool): `accent_status` が AP ベースかどうか

        Returns:
            OpenJTalkFormatLabel: 形態素単位へ射影した OpenJTalk 互換ラベル
        """

        if is_ap_based_accent_status is True:
            # AP ベースの `convert_single_ap_based_accent_to_mora_based_accent()` は、
            # `accent_represent_mode` と `accent_phrase_boundary_label` を使って0/1 のモーラ列へ復元する
            ## そのままでは `convert_open_jtalk_format_label()` が期待する
            ## `0=pad, 1=非アクセント核, 2=アクセント核` の符号化と 1 ずれるため、
            ## pad を避けつつ `1/2` へそろえる目的で +1 する
            mora_based_accent_status = (
                convert_single_ap_based_accent_to_mora_based_accent(
                    accent_status,
                    accent_phrase_boundary,
                    mode=self.accent_represent_mode,
                    accent_phrase_boundary_label=self.accent_phrase_boundary_label,
                )
                + 1
            )
            mora_based_accent_status = mora_based_accent_status.tolist()
        else:
            mora_based_accent_status = accent_status.tolist()

        mora_based_labels = cast(
            MarineLabel,
            {
                "accent_status": [mora_based_accent_status],
                "accent_phrase_boundary": [accent_phrase_boundary.tolist()],
            },
        )

        return convert_open_jtalk_format_label(
            mora_based_labels,
            [morph_boundary],
            accent_nucleus_label=self.accent_nucleus_label,
            accent_phrase_boundary_label=self.accent_phrase_boundary_label,
            morph_boundary_label=self.morph_boundary_label,
        )

    def _count_equal_items(
        self,
        predicts: Sequence[int],
        targets: Sequence[int],
    ) -> int:
        """
        2 系列の同位置一致数を返す。

        Args:
            predicts (Sequence[int]): 予測ラベル列
            targets (Sequence[int]): 正解ラベル列

        Returns:
            int: 同位置で一致したラベル数
        """

        assert len(predicts) == len(targets), (
            "Not match sequence lengths between predicted and target labels"
        )

        return sum(
            int(predicted_item == target_item)
            for predicted_item, target_item in zip(predicts, targets)
        )

    def _count_joint_equal_morphs(
        self,
        predicts: OpenJTalkFormatLabel,
        targets: OpenJTalkFormatLabel,
    ) -> int:
        """
        形態素単位で `accent_status` と `accent_phrase_boundary` の両方が一致した数を返す。

        Args:
            predicts (OpenJTalkFormatLabel): 予測ラベル
            targets (OpenJTalkFormatLabel): 正解ラベル

        Returns:
            int: joint 一致した形態素数
        """

        assert len(predicts["accent_status"]) == len(targets["accent_status"]), (
            "Not match morph counts between predicted and target accent_status labels"
        )
        assert len(predicts["accent_phrase_boundary"]) == len(
            targets["accent_phrase_boundary"]
        ), "Not match morph counts between predicted and target accent phrase labels"

        return sum(
            int(
                predicted_accent_status == target_accent_status
                and predicted_accent_phrase_boundary == target_accent_phrase_boundary
            )
            for (
                predicted_accent_status,
                target_accent_status,
                predicted_accent_phrase_boundary,
                target_accent_phrase_boundary,
            ) in zip(
                predicts["accent_status"],
                targets["accent_status"],
                predicts["accent_phrase_boundary"],
                targets["accent_phrase_boundary"],
            )
        )

    def _select_masked_tensor(
        self,
        predicts: dict[str, torch.Tensor],
        outputs: dict[str, dict[str, torch.Tensor]],
        task: str,
        sample_index: int,
        is_target: bool,
    ) -> torch.Tensor:
        """
        単一サンプルの有効長部分だけを切り出したラベル列を返す。

        Args:
            predicts (dict[str, torch.Tensor]): 各タスクの予測ラベル列
            outputs (dict[str, dict[str, torch.Tensor]]): 各タスクの正解ラベル列とマスク
            task (str): 対象タスク
            sample_index (int): バッチ内インデックス
            is_target (bool): 正解ラベルを使う場合は `True`

        Returns:
            torch.Tensor: パディングを除いたラベル列
        """

        task_mask = outputs[task]["mask"][sample_index]
        source = outputs[task]["label"] if is_target is True else predicts[task]
        return source[sample_index][task_mask].detach().cpu()
