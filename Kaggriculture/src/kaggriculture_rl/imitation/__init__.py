"""Replay imitation utilities for behavior-cloning warm starts."""

from kaggriculture_rl.imitation.replay_dataset import (
    DEFAULT_FEATURE_DIM,
    MAX_HANDS,
    MAX_MARKET_ORDERS,
    ReplayDataset,
    build_replay_dataset,
    encode_observation,
    load_replay_dataset,
)

__all__ = [
    "DEFAULT_FEATURE_DIM",
    "MAX_HANDS",
    "MAX_MARKET_ORDERS",
    "BCConfig",
    "BehaviorCloningPolicy",
    "PolicyPrediction",
    "ReplayDataset",
    "build_replay_dataset",
    "encode_observation",
    "load_replay_dataset",
    "save_checkpoint",
    "train_behavior_cloning",
]


def __getattr__(name: str) -> object:
    if name in {
        "BCConfig",
        "BehaviorCloningPolicy",
        "PolicyPrediction",
        "save_checkpoint",
        "train_behavior_cloning",
    }:
        from kaggriculture_rl.imitation import bc

        return getattr(bc, name)
    raise AttributeError(name)
