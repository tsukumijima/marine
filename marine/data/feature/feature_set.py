from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from marine.data.feature.feature_table import (
    FEATURE_TABLES,
    PUNCTUATIONS,
    parse_accent_con_type,
)
from marine.types import BatchFeature, MarineFeature
from marine.utils.g2p_util import pron2mora


logger = getLogger(__name__)
FEATURE_ID_DTYPE = np.int64


class FeatureSet:
    """
    A converter for embedding features
    """

    def __init__(
        self,
        vocab_path: str | Path,
        feature_table_key: str = "unidic-csj",
        feature_keys: list[str] | None = None,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
    ) -> None:
        self.vocab_path = Path(vocab_path)

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.default_tokens = [self.pad_token, self.unk_token]
        self.feature_table = FEATURE_TABLES[feature_table_key]

        if feature_keys:
            self.feature_keys = feature_keys
        else:
            self.feature_keys = list(self.feature_table.keys())

        self.feature_to_id = {key: {} for key in self.feature_keys}
        self.id_to_feature = {key: {} for key in self.feature_keys}

        self.init_feature_set()

    def _load_vocab(self) -> list[str]:
        if not self.vocab_path.exists():
            logger.error(f"Vocab has not found : {self.vocab_path}")
            raise FileNotFoundError(f"Vocab has not found : {self.vocab_path}")

        vocab = joblib.load(self.vocab_path)
        logger.info(f"Vocab loaded from {self.vocab_path} : {len(vocab)} words")

        return vocab

    def init_feature_set(self) -> None:
        self._load_vocab()

        for key in self.feature_keys:
            if key == "surface":
                feature_set = self.default_tokens + self._load_vocab()
            else:
                if key not in self.feature_table.keys():
                    raise ValueError(
                        f"Feature key must be one of {self.feature_table.keys()}"
                    )
                feature_set = self.default_tokens + (self.feature_table[key] or [])

            feature_to_id = {
                feature_value: index for index, feature_value in enumerate(feature_set)
            }
            id_to_feature = {
                index: feature_value for feature_value, index in feature_to_id.items()
            }
            self.feature_to_id[key] = feature_to_id
            self.id_to_feature[key] = id_to_feature

    def convert_feature_to_id(
        self, feature_key: str, features: Sequence[str | int]
    ) -> NDArray[np.int64]:
        if feature_key not in self.feature_to_id:
            raise ValueError(
                f"Not initialized feature key: the key must be one of {self.feature_to_id}"
            )

        return np.array(
            [
                self.feature_to_id[feature_key].get(
                    value, self.feature_to_id[feature_key][self.unk_token]
                )
                for value in features
            ],
            dtype=FEATURE_ID_DTYPE,
        )

    def convert_id_to_feature(
        self,
        feature_key: str,
        ids: Sequence[int] | NDArray[np.int64],
    ) -> NDArray[Any]:
        if feature_key not in self.id_to_feature:
            raise ValueError(
                f"Not initialized feature key: the key must be one of {self.id_to_feature}"
            )

        return np.array(
            [
                self.id_to_feature[feature_key].get(value, self.unk_token)
                for value in ids
            ]
        )

    def convert_nodes_to_feature(self, nodes: list[MarineFeature]) -> BatchFeature:
        """
        形態素列を埋め込み用の BatchFeature に変換する。

        Args:
            nodes (list[MarineFeature]): 形態素情報のリスト。各要素は surface, pron, pos,
                c_type, c_form, accent_type, accent_con_type, chain_flag を持つ辞書。

        Returns:
            BatchFeature: 特徴量キーをキー、NDArray を値とする辞書
        """

        # Feature IDs are consumed by embedding layers as integer indices.
        # They must preserve the original vocabulary index without overflow.
        features: dict[str, NDArray[Any]] = {
            key: np.array([], dtype=FEATURE_ID_DTYPE) for key in self.feature_to_id
        }

        # init morph boundary for inference
        features["morph_boundary"] = np.array([], dtype=np.uint8)

        for node in nodes:
            mora = self.convert_feature_to_id(
                "mora",
                cast(
                    list[str | int],
                    pron2mora(node["pron"]) if node["pron"] else [node["surface"]],
                ),
            )

            morph_boundary = np.array([1] + ([0] * (len(mora) - 1)), dtype=np.uint8)

            # Push features
            features["mora"] = np.concatenate([features["mora"], mora], axis=0)
            features["morph_boundary"] = np.concatenate(
                [features["morph_boundary"], morph_boundary], axis=0
            )

            for key, table in self.feature_to_id.items():
                if key in ["mora", "morph_boundary"]:
                    continue

                if key == "accent_con_type":
                    value = parse_accent_con_type(
                        node["accent_con_type"], node["pos"], unk_token=self.unk_token
                    )
                else:
                    value = node[key]
                feature = table.get(value, table[self.unk_token])
                feature = np.full(len(mora), feature, dtype=FEATURE_ID_DTYPE)
                features[key] = np.concatenate([features[key], feature], axis=0)

        # First Mora could not be boundary
        # (boundary should be [0, 0, 1, 0, 0 ...])
        features["morph_boundary"][0] = 0

        return cast(BatchFeature, features)

    def get_punctuation_ids(self) -> list[int]:
        return [self.feature_to_id["mora"][punctuation] for punctuation in PUNCTUATIONS]
