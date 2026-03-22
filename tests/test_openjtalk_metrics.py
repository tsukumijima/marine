import numpy as np
import torch
from numpy.testing import assert_almost_equal

from marine.utils.openjtalk_metrics import OpenJTalkPracticalMetrics


def test_openjtalk_practical_metrics_mora_based() -> None:
    metrics = OpenJTalkPracticalMetrics(accent_represent_mode="binary")

    predicts = {
        "accent_status": torch.tensor(
            [
                [1, 2, 1, 1],
                [1, 2, 1, 1],
            ]
        ),
        "accent_phrase_boundary": torch.tensor(
            [
                [1, 1, 2, 1],
                [1, 1, 1, 1],
            ]
        ),
    }
    outputs = {
        "accent_status": {
            "label": torch.tensor(
                [
                    [1, 2, 1, 1],
                    [1, 2, 1, 1],
                ]
            ),
            "mask": torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            ),
        },
        "accent_phrase_boundary": {
            "label": torch.tensor(
                [
                    [1, 1, 2, 1],
                    [1, 1, 2, 1],
                ]
            ),
            "mask": torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            ),
        },
    }
    morph_boundaries = [
        np.array([0, 0, 1, 0], dtype=np.uint8),
        np.array([0, 0, 1, 0], dtype=np.uint8),
    ]

    metrics.update(
        predicts=predicts,
        outputs=outputs,
        morph_boundaries=morph_boundaries,
        is_ap_based_accent_status=False,
    )

    scores = metrics.compute()
    assert_almost_equal(scores["openjtalk_accent_status_sentence_accuracy"], 1.0)
    assert_almost_equal(scores["openjtalk_gt_ap_accent_status_morph_accuracy"], 1.0)
    assert_almost_equal(scores["openjtalk_joint_morph_accuracy"], 0.75)
    assert_almost_equal(scores["openjtalk_joint_sentence_accuracy"], 0.5)


def test_openjtalk_practical_metrics_ap_based() -> None:
    metrics = OpenJTalkPracticalMetrics(accent_represent_mode="binary")

    predicts = {
        "accent_status": torch.tensor(
            [
                [3, 1],
                [2, 1],
            ]
        ),
        "accent_phrase_boundary": torch.tensor(
            [
                [1, 1, 2, 1],
                [1, 1, 2, 1],
            ]
        ),
    }
    outputs = {
        "accent_status": {
            "label": torch.tensor(
                [
                    [3, 1],
                    [3, 1],
                ]
            ),
            "mask": torch.tensor(
                [
                    [True, True],
                    [True, True],
                ]
            ),
        },
        "accent_phrase_boundary": {
            "label": torch.tensor(
                [
                    [1, 1, 2, 1],
                    [1, 1, 2, 1],
                ]
            ),
            "mask": torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            ),
        },
    }
    morph_boundaries = [
        np.array([0, 0, 1, 0], dtype=np.uint8),
        np.array([0, 0, 1, 0], dtype=np.uint8),
    ]

    metrics.update(
        predicts=predicts,
        outputs=outputs,
        morph_boundaries=morph_boundaries,
        is_ap_based_accent_status=True,
    )

    scores = metrics.compute()
    assert_almost_equal(scores["openjtalk_accent_status_sentence_accuracy"], 0.5)
    assert_almost_equal(scores["openjtalk_gt_ap_accent_status_morph_accuracy"], 0.75)
    assert_almost_equal(scores["openjtalk_joint_morph_accuracy"], 0.75)
    assert_almost_equal(scores["openjtalk_joint_sentence_accuracy"], 0.5)


def test_openjtalk_practical_metrics_gt_ap_accent_status_separates_boundary_error() -> (
    None
):
    metrics = OpenJTalkPracticalMetrics(accent_represent_mode="binary")

    predicts = {
        "accent_status": torch.tensor([[1, 1, 1, 2]]),
        "accent_phrase_boundary": torch.tensor([[1, 1, 1, 1]]),
    }
    outputs = {
        "accent_status": {
            "label": torch.tensor([[1, 1, 1, 2]]),
            "mask": torch.tensor([[True, True, True, True]]),
        },
        "accent_phrase_boundary": {
            "label": torch.tensor([[1, 1, 2, 1]]),
            "mask": torch.tensor([[True, True, True, True]]),
        },
    }
    morph_boundaries = [
        np.array([0, 0, 1, 0], dtype=np.uint8),
    ]

    metrics.update(
        predicts=predicts,
        outputs=outputs,
        morph_boundaries=morph_boundaries,
        is_ap_based_accent_status=False,
    )

    scores = metrics.compute()
    assert_almost_equal(scores["openjtalk_accent_status_sentence_accuracy"], 0.0)
    assert_almost_equal(scores["openjtalk_gt_ap_accent_status_morph_accuracy"], 1.0)
    assert_almost_equal(scores["openjtalk_joint_morph_accuracy"], 0.0)
    assert_almost_equal(scores["openjtalk_joint_sentence_accuracy"], 0.0)
