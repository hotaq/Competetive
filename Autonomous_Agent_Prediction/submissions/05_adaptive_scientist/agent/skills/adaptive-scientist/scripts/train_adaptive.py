#!/usr/bin/env python3
"""Train fixed anchors plus one allowlisted, profile-directed candidate view."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC

import model_library as lib
from profile_dataset import ALLOWED_RECIPES, build_profile


SEED = 42
MAX_EXPORTED_CANDIDATES = 6
MAX_PLAN_BYTES = 8_192
MIN_BLEND_GAIN = 0.0005
MAX_DIVERSITY_CORRELATION = 0.985


def _read_json(path: Path, max_bytes: int = MAX_PLAN_BYTES) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON file is too large: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _load_profile(data_dir: Path, profile_path: Path) -> dict:
    if profile_path.is_file():
        profile = _read_json(profile_path, max_bytes=64_000)
    else:
        profile = build_profile(data_dir)
        profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    default_recipe = profile.get("default_recipe")
    if default_recipe not in ALLOWED_RECIPES:
        profile["default_recipe"] = "mixed-safe"
    return profile


def _validated_recipe(profile: dict, plan_path: Path | None) -> tuple[str, dict]:
    fallback = str(profile.get("default_recipe", "mixed-safe"))
    if fallback not in ALLOWED_RECIPES:
        fallback = "mixed-safe"
    diagnostic = {"source": "profile", "valid": True, "recipe_id": fallback}
    if plan_path is None:
        return fallback, diagnostic
    try:
        plan = _read_json(plan_path)
        allowed_keys = {"recipe_id", "reason", "confidence", "stage"}
        if not set(plan).issubset(allowed_keys):
            raise ValueError("Planner output contains unsupported fields")
        recipe = plan.get("recipe_id")
        if recipe not in ALLOWED_RECIPES:
            raise ValueError("Planner recipe is not allowlisted")
        if "reason" in plan and not isinstance(plan["reason"], str):
            raise ValueError("Planner reason must be text")
        diagnostic = {
            "source": "planner",
            "valid": True,
            "recipe_id": recipe,
            "reason": str(plan.get("reason", ""))[:300],
        }
        return str(recipe), diagnostic
    except Exception as error:
        diagnostic = {
            "source": "profile_fallback",
            "valid": False,
            "recipe_id": fallback,
            "error": type(error).__name__,
        }
        return fallback, diagnostic


def _safe_add(
    candidates: list[lib.PredictionSet],
    producer: Callable[[], lib.PredictionSet],
    name: str,
) -> None:
    lib.add_candidate(candidates, producer, name)


def _missing_indicator_preprocessor(
    categorical: list[str], numeric: list[str]
) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _ordinal_lightgbm(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
    numeric: list[str],
    ordinal_like: list[str],
    folds: StratifiedKFold,
) -> lib.PredictionSet:
    promoted = [column for column in ordinal_like if column in numeric]
    if not promoted:
        raise ValueError("No ordinal-like columns to promote")
    train_view = train_x.copy()
    test_view = test_x.copy()
    for column in promoted:
        train_view[column] = train_view[column].astype("string")
        test_view[column] = test_view[column].astype("string")
    return lib.cross_validated_lightgbm(
        "ordinal_native_lightgbm",
        train_view,
        test_view,
        target,
        categorical + promoted,
        [column for column in numeric if column not in promoted],
        folds,
    )


def _candidate_by_name(
    candidates: list[lib.PredictionSet], names: set[str]
) -> lib.PredictionSet | None:
    matches = [candidate for candidate in candidates if candidate.name in names]
    return max(matches, key=lambda item: item.auc) if matches else None


def _append_unique(
    selected: list[lib.PredictionSet], candidate: lib.PredictionSet | None
) -> None:
    if candidate is None:
        return
    if candidate.name not in {item.name for item in selected}:
        selected.append(candidate)


def _choose_exports(
    core: list[lib.PredictionSet],
    specialists: list[lib.PredictionSet],
    meta: list[lib.PredictionSet],
    recipe_candidates: list[lib.PredictionSet],
    target: np.ndarray,
) -> tuple[list[lib.PredictionSet], set[str], dict]:
    if not core:
        raise RuntimeError("No core candidate completed")
    ranked_core = sorted(core, key=lambda item: item.auc, reverse=True)
    anchors: list[lib.PredictionSet] = []
    _append_unique(
        anchors,
        _candidate_by_name(ranked_core, {"catboost_d6", "catboost_d8"}),
    )
    _append_unique(anchors, _candidate_by_name(ranked_core, {"lightgbm_native"}))
    _append_unique(anchors, _candidate_by_name(ranked_core, {"logistic"}))

    for candidate in ranked_core:
        if len(anchors) >= 3:
            break
        if all(candidate.family != retained.family for retained in anchors):
            _append_unique(anchors, candidate)
    for candidate in ranked_core:
        if len(anchors) >= 3:
            break
        _append_unique(anchors, candidate)

    guarded, guarded_diagnostics = lib.select_guarded_candidates(
        core,
        specialists,
        meta,
        target,
    )
    exported = list(anchors)
    for candidate in sorted(recipe_candidates, key=lambda item: item.auc, reverse=True):
        _append_unique(exported, candidate)
    for candidate in guarded:
        _append_unique(exported, candidate)

    all_candidates = sorted(
        core + specialists + meta,
        key=lambda item: item.auc,
        reverse=True,
    )
    best = all_candidates[0]
    for candidate in all_candidates:
        if len(exported) >= MAX_EXPORTED_CANDIDATES:
            break
        if candidate.name in {item.name for item in exported}:
            continue
        correlation = abs(lib.rank_correlation(best.test, candidate.test))
        if correlation <= MAX_DIVERSITY_CORRELATION or candidate.auc >= best.auc:
            _append_unique(exported, candidate)
    for candidate in all_candidates:
        if len(exported) >= MAX_EXPORTED_CANDIDATES:
            break
        _append_unique(exported, candidate)

    exported = exported[:MAX_EXPORTED_CANDIDATES]
    anchor_names = {candidate.name for candidate in anchors if candidate in exported}
    diagnostics = {
        "anchors": [candidate.name for candidate in anchors],
        "guarded_selection": guarded_diagnostics,
        "recipe_candidates": [candidate.name for candidate in recipe_candidates],
    }
    return exported, anchor_names, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--profile", type=Path, default=Path("dataset_profile.json"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--models", choices=["all", "sklearn"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lib.FAILURES.clear()

    profile = _load_profile(args.data_dir, args.profile)
    recipe, planner_diagnostic = _validated_recipe(profile, args.plan)
    train, test, sample, id_column, target_column = lib.load_data(args.data_dir)
    feature_columns = [column for column in test.columns if str(column) != id_column]
    train_x = train[feature_columns].copy()
    test_x = test[feature_columns].copy()
    target, classes = lib.encode_target(train[target_column])
    categorical, numeric = lib.feature_types(train_x)
    minority_count = int(np.bincount(target).min())
    n_splits = min(5, minority_count)
    if n_splits < 2:
        raise ValueError("Not enough minority rows for stratified validation")
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    core: list[lib.PredictionSet] = []
    specialists: list[lib.PredictionSet] = []
    meta: list[lib.PredictionSet] = []
    recipe_candidates: list[lib.PredictionSet] = []
    make_ordinal = lambda: lib.ordinal_preprocessor(categorical, numeric)
    make_linear = lambda: lib.linear_preprocessor(categorical, numeric)

    if args.models == "all":
        _safe_add(
            core,
            lambda: lib.cross_validated_catboost(
                "catboost_d6",
                {
                    "iterations": 750,
                    "depth": 6,
                    "learning_rate": 0.04,
                    "l2_leaf_reg": 6.0,
                    "random_strength": 0.4,
                },
                train_x,
                test_x,
                target,
                categorical,
                numeric,
                folds,
            ),
            "catboost_d6",
        )
        if recipe in {"categorical-native", "ordinal-native", "tiny-safe"}:
            _safe_add(
                core,
                lambda: lib.cross_validated_catboost(
                    "catboost_d8",
                    {
                        "iterations": 550,
                        "depth": 8,
                        "learning_rate": 0.05,
                        "l2_leaf_reg": 8.0,
                        "random_strength": 0.8,
                    },
                    train_x,
                    test_x,
                    target,
                    categorical,
                    numeric,
                    folds,
                ),
                "catboost_d8",
            )
        _safe_add(
            core,
            lambda: lib.cross_validated_lightgbm(
                "lightgbm_native",
                train_x,
                test_x,
                target,
                categorical,
                numeric,
                folds,
            ),
            "lightgbm_native",
        )

    _safe_add(
        core,
        lambda: lib.cross_validated_pipeline(
            "logistic",
            "linear",
            lambda seed: LogisticRegression(
                C=0.5,
                max_iter=1_500,
                solver="liblinear",
                random_state=seed,
            ),
            make_linear,
            train_x,
            test_x,
            target,
            folds,
        ),
        "logistic",
    )
    _safe_add(
        core,
        lambda: lib.cross_validated_pipeline(
            "extra_trees",
            "bagging",
            lambda seed: ExtraTreesClassifier(
                n_estimators=600,
                max_features=0.8,
                min_samples_leaf=2 if len(train_x) >= 2_000 else 1,
                class_weight="balanced",
                random_state=seed,
                n_jobs=2,
            ),
            make_ordinal,
            train_x,
            test_x,
            target,
            folds,
        ),
        "extra_trees",
    )
    _safe_add(
        core,
        lambda: lib.cross_validated_pipeline(
            "hist_gradient",
            "histogram",
            lambda seed: HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=400,
                max_leaf_nodes=31,
                min_samples_leaf=20 if len(train_x) >= 2_000 else 10,
                l2_regularization=2.0,
                early_stopping=True,
                random_state=seed,
            ),
            make_ordinal,
            train_x,
            test_x,
            target,
            folds,
        ),
        "hist_gradient",
    )

    if len(train_x) <= 2_500:
        _safe_add(
            core,
            lambda: lib.cross_validated_pipeline(
                "small_rbf_svm",
                "kernel",
                lambda seed: SVC(C=2.0, gamma="scale", probability=True, random_state=seed),
                make_linear,
                train_x,
                test_x,
                target,
                folds,
            ),
            "small_rbf_svm",
        )

    if recipe in {"numeric-boost", "missing-robust"}:
        before = len(core)
        _safe_add(
            core,
            lambda: lib.cross_validated_pipeline(
                "missing_indicator_hist",
                "missing_histogram",
                lambda seed: HistGradientBoostingClassifier(
                    learning_rate=0.035,
                    max_iter=500,
                    max_leaf_nodes=31,
                    min_samples_leaf=15 if len(train_x) < 2_500 else 30,
                    l2_regularization=3.0,
                    early_stopping=True,
                    random_state=seed,
                ),
                lambda: _missing_indicator_preprocessor(categorical, numeric),
                train_x,
                test_x,
                target,
                folds,
            ),
            "missing_indicator_hist",
        )
        recipe_candidates.extend(core[before:])

    ordinal_like = [
        str(column)
        for column in profile.get("schema", {}).get("ordinal_like_columns", [])
        if str(column) in train_x.columns
    ]
    if args.models == "all" and recipe == "ordinal-native":
        before = len(core)
        _safe_add(
            core,
            lambda: _ordinal_lightgbm(
                train_x,
                test_x,
                target,
                categorical,
                numeric,
                ordinal_like,
                folds,
            ),
            "ordinal_native_lightgbm",
        )
        recipe_candidates.extend(core[before:])

    if args.models == "all" and recipe in {
        "mixed-safe",
        "numeric-boost",
        "categorical-native",
        "ordinal-native",
    }:
        before = len(specialists)
        _safe_add(
            specialists,
            lambda: lib.cross_validated_interaction_lightgbm(
                train_x,
                test_x,
                target,
                categorical,
                numeric,
                folds,
            ),
            "interaction_te",
        )
        recipe_candidates.extend(specialists[before:])

    if len(core) >= 2:
        _safe_add(
            meta,
            lambda: lib.crossfit_logistic_stack(core + specialists, target),
            "stack_logistic",
        )

    exported, anchor_names, diagnostics = _choose_exports(
        core,
        specialists,
        meta,
        recipe_candidates,
        target,
    )
    best = max(exported, key=lambda item: item.auc)
    manifest_candidates: list[dict] = []
    for candidate in exported:
        path = lib.write_submission(sample, target_column, candidate, args.output_dir)
        state_path = args.output_dir / f"candidate_{candidate.name}.npz"
        np.savez_compressed(state_path, oof=candidate.oof, test=candidate.test)
        correlation = (
            1.0
            if candidate.name == best.name
            else abs(lib.rank_correlation(best.test, candidate.test))
        )
        manifest_candidates.append(
            {
                "file": path.name,
                "state_file": state_path.name,
                "name": candidate.name,
                "family": candidate.family,
                "role": "anchor" if candidate.name in anchor_names else "challenger",
                "oof_auc": round(candidate.auc, 8),
                "fold_auc": [round(score, 8) for score in candidate.fold_auc],
                "fold_auc_std": (
                    round(float(np.std(candidate.fold_auc)), 8)
                    if candidate.fold_auc
                    else None
                ),
                "test_rank_correlation_to_best": round(float(correlation), 8),
            }
        )

    manifest = {
        "manifest_version": 1,
        "classes": classes,
        "rows": len(sample),
        "features": len(feature_columns),
        "recipe_id": recipe,
        "planner": planner_diagnostic,
        "initial_candidates": [
            item["name"] for item in manifest_candidates if item["role"] == "anchor"
        ],
        "challenger_candidates": [
            item["name"] for item in manifest_candidates if item["role"] == "challenger"
        ],
        "candidates": manifest_candidates,
        "diagnostics": diagnostics,
        "failures": list(lib.FAILURES),
    }
    manifest_path = args.output_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    candidate_summary = ",".join(
        f"{item['file']}:{item['oof_auc']:.8f}" for item in manifest_candidates
    )
    anchor_summary = ",".join(
        item["file"] for item in manifest_candidates if item["role"] == "anchor"
    )
    challenger_summary = ",".join(
        item["file"] for item in manifest_candidates if item["role"] == "challenger"
    )
    print(f"CANDIDATES {candidate_summary}")
    print(f"ANCHORS {anchor_summary}")
    print(f"CHALLENGERS {challenger_summary}")


if __name__ == "__main__":
    main()
