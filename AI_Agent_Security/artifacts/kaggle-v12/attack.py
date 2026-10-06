"""Kaggle submission entry point for content-guided trace exploration."""

from __future__ import annotations

from aicomp_sdk.attacks import (
    AttackAlgorithmBase,
    AttackCandidate,
    AttackEnvProtocol,
    AttackRunConfig,
)

from src.strategies.content_guided import ContentGuidedSearch


class AttackAlgorithm(AttackAlgorithmBase):
    """Discover and expand references while retaining diverse replay cells."""

    def run(
        self,
        env: AttackEnvProtocol,
        config: AttackRunConfig,
    ) -> list[AttackCandidate]:
        return ContentGuidedSearch().run(env, config)
