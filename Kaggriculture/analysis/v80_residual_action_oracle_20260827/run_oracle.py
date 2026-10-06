#!/usr/bin/env python3
"""Bounded V67 residual-action oracle on the consumed Gate8 replays.

This is deliberately not an agent.  Each simulation changes exactly one
post-opening macro event (or moves one market order by one transition), keeps
every other V67 uploaded action byte-for-byte, and keeps the opponent tape
fixed.  The official Rust engine executes the resulting raw action stream.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from kaggriculture_rs import Engine  # noqa: E402

from scripts.evaluate_replay_tape import engine_config, frame_action  # noqa: E402

ANALYSIS_DIR = ROOT / "analysis/v80_residual_action_oracle_20260827"
GATE_PATH = ROOT / "analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json"
REPLAY_DIR = ROOT / "replays/submission_55797143_public_refresh_after22_20260826"
V67_PATH = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
NATIVE_ENGINE_PATH = ROOT / "bindings/python/kaggriculture_rs/_native.abi3.so"
RESULTS_PATH = ANALYSIS_DIR / "results.json"
VARIANT_ROWS_PATH = ANALYSIS_DIR / "variant_results.jsonl"
HASHES_PATH = ANALYSIS_DIR / "HASHES.sha256"

EXPECTED_HASHES = {
    GATE_PATH: "4e9268c02575ea9bb6b2e9aaff8714910bc2fed3d3d886a05355dbb47edc8be2",
    V67_PATH: "300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b",
    NATIVE_ENGINE_PATH: "315f30f12797c64ed1038347af1496d0ba18f9dc5ce3adcd15977a741f84acdd",
}

START_STEP = 120
MARKET_VERBS = {"HIRE", "BUY_LAND", "BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT", "SELL"}
ANIMALS = ("GOOSE", "COW", "SHEEP")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
BUY_PRODUCTS = ("WHEAT", "FERTILIZER")
UNIT_MACROS = {"PLANT", "DIG", "BUILD_COOP", "BUILD_PASTURE", "PLACE", "REMOVE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def command_parts(command: Any) -> tuple[str, str | None, int | None]:
    if not isinstance(command, list) or not command:
        return "INVALID", None, None
    verb = str(command[0])
    item = str(command[1]) if len(command) >= 2 else None
    quantity = int(command[2]) if len(command) >= 3 and isinstance(command[2], int) else None
    return verb, item, quantity


def category_for(verb: str, item: str | None) -> str:
    if verb == "HIRE":
        return "hire"
    if verb == "BUY_LAND":
        return "land"
    if verb in {"BUY_ANIMAL", "BUILD_COOP", "BUILD_PASTURE"}:
        return "animal"
    if verb == "PLACE" and item in ANIMALS:
        return "animal"
    if verb in {"BUY_SEED", "PLANT", "DIG"}:
        return "crop"
    return "market"


@dataclass(frozen=True, slots=True)
class Mutation:
    mutation_id: str
    semantic_key: str
    family: str
    category: str
    descriptor: dict[str, Any]
    overrides: tuple[tuple[int, dict[str, Any]], ...]


def make_mutation(
    *,
    family: str,
    category: str,
    descriptor: dict[str, Any],
    overrides: dict[int, dict[str, Any]],
) -> Mutation:
    semantic = {
        key: value
        for key, value in descriptor.items()
        if key not in {"step", "market_index", "hand_index", "original", "replacement"}
    }
    semantic["family"] = family
    semantic["category"] = category
    semantic_key = canonical(semantic)
    payload = {
        "family": family,
        "category": category,
        "descriptor": descriptor,
        "overrides": sorted((step, action) for step, action in overrides.items()),
    }
    mutation_id = hashlib.sha256(canonical(payload).encode()).hexdigest()[:20]
    return Mutation(
        mutation_id=mutation_id,
        semantic_key=semantic_key,
        family=family,
        category=category,
        descriptor=descriptor,
        overrides=tuple(sorted(overrides.items())),
    )


def replace_market_order(
    actions: list[dict[str, Any]],
    step: int,
    index: int,
    replacement: list[Any] | None,
) -> dict[int, dict[str, Any]]:
    action = copy.deepcopy(actions[step])
    market = list(action.get("market", []))
    if replacement is None:
        market.pop(index)
    else:
        market[index] = replacement
    action["market"] = market
    return {step: action}


def shift_market_order(
    actions: list[dict[str, Any]],
    step: int,
    index: int,
    target_step: int,
) -> dict[int, dict[str, Any]] | None:
    if target_step < START_STEP or target_step >= len(actions):
        return None
    source = copy.deepcopy(actions[step])
    target = copy.deepcopy(actions[target_step])
    source_market = list(source.get("market", []))
    target_market = list(target.get("market", []))
    if len(target_market) >= 10:
        return None
    order = source_market.pop(index)
    target_market.append(order)
    source["market"] = source_market
    target["market"] = target_market
    return {step: source, target_step: target}


def replace_unit_command(
    actions: list[dict[str, Any]],
    step: int,
    actor: str,
    index: int | None,
    replacement: list[Any],
) -> dict[int, dict[str, Any]]:
    action = copy.deepcopy(actions[step])
    if actor == "farmer":
        action["farmer"] = replacement
    else:
        hands = list(action.get("hands", []))
        assert index is not None
        hands[index] = replacement
        action["hands"] = hands
    return {step: action}


def quantity_replacements(quantity: int) -> list[tuple[str, int]]:
    choices = (
        ("half", max(1, (quantity + 1) // 2)),
        ("minus_one", max(1, quantity - 1)),
        ("plus_one", min(100, quantity + 1)),
        ("double", min(100, quantity * 2)),
    )
    seen: set[int] = {quantity}
    result: list[tuple[str, int]] = []
    for mode, value in choices:
        if value not in seen:
            result.append((mode, value))
            seen.add(value)
    return result


def enumerate_mutations(
    actions: list[dict[str, Any]],
) -> tuple[list[Mutation], list[dict[str, Any]]]:
    mutations: list[Mutation] = []
    catalog: list[dict[str, Any]] = []
    occurrence: Counter[tuple[str, str, str]] = Counter()

    for step in range(START_STEP, len(actions)):
        action = actions[step]
        for market_index, order in enumerate(action.get("market", [])):
            verb, item, quantity = command_parts(order)
            if verb not in MARKET_VERBS:
                continue
            category = category_for(verb, item)
            occurrence_key = ("market", verb, item or "-")
            event_occurrence = occurrence[occurrence_key]
            occurrence[occurrence_key] += 1
            base = {
                "surface": "market",
                "verb": verb,
                "item": item,
                "occurrence": event_occurrence,
                "step": step,
                "market_index": market_index,
                "original": order,
            }
            catalog.append({**base, "category": category, "quantity": quantity})

            descriptor = {**base, "operation": "drop"}
            mutations.append(
                make_mutation(
                    family="market_drop",
                    category=category,
                    descriptor=descriptor,
                    overrides=replace_market_order(actions, step, market_index, None),
                )
            )

            if quantity is not None and len(order) >= 3:
                for mode, new_quantity in quantity_replacements(quantity):
                    replacement = [*order[:2], new_quantity]
                    descriptor = {
                        **base,
                        "operation": "quantity",
                        "quantity_mode": mode,
                        "replacement": replacement,
                    }
                    mutations.append(
                        make_mutation(
                            family="market_quantity",
                            category=category,
                            descriptor=descriptor,
                            overrides=replace_market_order(
                                actions, step, market_index, replacement
                            ),
                        )
                    )

            item_choices: tuple[str, ...] = ()
            if verb == "BUY_ANIMAL":
                item_choices = ANIMALS
            elif verb == "BUY_SEED":
                item_choices = CROPS
            elif verb == "BUY_PRODUCT":
                item_choices = BUY_PRODUCTS
            elif verb == "SELL":
                item_choices = PRODUCTS
            if item is not None and quantity is not None:
                for new_item in item_choices:
                    if new_item == item:
                        continue
                    replacement = [verb, new_item, quantity]
                    descriptor = {
                        **base,
                        "operation": "swap_item",
                        "target_item": new_item,
                        "replacement": replacement,
                    }
                    mutations.append(
                        make_mutation(
                            family="market_item_swap",
                            category=category,
                            descriptor=descriptor,
                            overrides=replace_market_order(
                                actions, step, market_index, replacement
                            ),
                        )
                    )

            market_length = len(action.get("market", []))
            for position, target_index in (("front", 0), ("back", market_length - 1)):
                if market_index == target_index:
                    continue
                changed = copy.deepcopy(action)
                market = list(changed.get("market", []))
                moved = market.pop(market_index)
                market.insert(target_index, moved)
                changed["market"] = market
                descriptor = {
                    **base,
                    "operation": "reorder",
                    "target_position": position,
                }
                mutations.append(
                    make_mutation(
                        family="market_reorder",
                        category=category,
                        descriptor=descriptor,
                        overrides={step: changed},
                    )
                )

            for offset in (-1, 1):
                overrides = shift_market_order(actions, step, market_index, step + offset)
                if overrides is None:
                    continue
                descriptor = {
                    **base,
                    "operation": "shift",
                    "step_offset": offset,
                }
                mutations.append(
                    make_mutation(
                        family="market_shift_one",
                        category=category,
                        descriptor=descriptor,
                        overrides=overrides,
                    )
                )

        unit_commands: list[tuple[str, int | None, Any]] = [
            ("farmer", None, action.get("farmer", ["PASS"]))
        ]
        unit_commands.extend(
            ("hands", index, command) for index, command in enumerate(action.get("hands", []))
        )
        for actor, hand_index, command in unit_commands:
            verb, item, quantity = command_parts(command)
            if verb not in UNIT_MACROS:
                continue
            if verb == "PLACE" and item not in ANIMALS:
                continue
            category = category_for(verb, item)
            occurrence_key = (actor, verb, item or "-")
            event_occurrence = occurrence[occurrence_key]
            occurrence[occurrence_key] += 1
            base = {
                "surface": actor,
                "verb": verb,
                "item": item,
                "occurrence": event_occurrence,
                "step": step,
                "hand_index": hand_index,
                "original": command,
            }
            catalog.append({**base, "category": category, "quantity": quantity})
            mutations.append(
                make_mutation(
                    family="unit_macro_drop",
                    category=category,
                    descriptor={**base, "operation": "drop"},
                    overrides=replace_unit_command(actions, step, actor, hand_index, ["PASS"]),
                )
            )

            if verb == "PLANT" and item is not None:
                for new_item in CROPS:
                    if new_item == item:
                        continue
                    replacement = ["PLANT", new_item]
                    mutations.append(
                        make_mutation(
                            family="plant_item_swap",
                            category=category,
                            descriptor={
                                **base,
                                "operation": "swap_item",
                                "target_item": new_item,
                                "replacement": replacement,
                            },
                            overrides=replace_unit_command(
                                actions, step, actor, hand_index, replacement
                            ),
                        )
                    )
            elif verb in {"BUILD_COOP", "BUILD_PASTURE"}:
                replacement = ["BUILD_PASTURE" if verb == "BUILD_COOP" else "BUILD_COOP"]
                mutations.append(
                    make_mutation(
                        family="animal_structure_swap",
                        category=category,
                        descriptor={
                            **base,
                            "operation": "swap_structure",
                            "replacement": replacement,
                        },
                        overrides=replace_unit_command(
                            actions, step, actor, hand_index, replacement
                        ),
                    )
                )
            elif verb == "PLACE" and item in ANIMALS:
                for new_item in ANIMALS:
                    if new_item == item:
                        continue
                    replacement = ["PLACE", new_item, quantity or 1]
                    mutations.append(
                        make_mutation(
                            family="animal_place_swap",
                            category=category,
                            descriptor={
                                **base,
                                "operation": "swap_item",
                                "target_item": new_item,
                                "replacement": replacement,
                            },
                            overrides=replace_unit_command(
                                actions, step, actor, hand_index, replacement
                            ),
                        )
                    )

    unique: dict[str, Mutation] = {}
    for mutation in mutations:
        unique.setdefault(mutation.mutation_id, mutation)
    return list(unique.values()), catalog


def simulate(
    replay: dict[str, Any],
    candidate_seat: int,
    overrides: dict[int, dict[str, Any]],
    *,
    diagnose_all_sanitizer_steps: bool = False,
) -> dict[str, Any]:
    engine = Engine(engine_config(replay))
    sanitizer_checked_steps = 0
    sanitizer_changed_checked_steps = 0
    mutated_step_sanitizer_changes = 0
    for step, frame in enumerate(replay["steps"][1:]):
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        candidate_action = overrides.get(step, actions[candidate_seat])
        if step in overrides:
            actions = list(actions)
            actions[candidate_seat] = candidate_action
        if diagnose_all_sanitizer_steps or step in overrides:
            sanitizer_checked_steps += 1
            sanitized = engine.sanitize_action(candidate_seat, candidate_action)
            changed = int(sanitized != candidate_action)
            sanitizer_changed_checked_steps += changed
            mutated_step_sanitizer_changes += changed if step in overrides else 0
        engine.step(actions)
    money = [float(farm["money"]) for farm in engine.state["farms"]]
    return {
        "margin": money[candidate_seat] - money[1 - candidate_seat],
        "candidate_money": money[candidate_seat],
        "opponent_money": money[1 - candidate_seat],
        "sanitizer_checked_steps": sanitizer_checked_steps,
        "sanitizer_changed_checked_steps": sanitizer_changed_checked_steps,
        "mutated_step_sanitizer_changes": mutated_step_sanitizer_changes,
        "full_sanitizer_diagnostic": diagnose_all_sanitizer_steps,
        "final_step": int(engine.state["step"]),
    }


def compact_result(
    mutation: Mutation,
    simulation: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mutation_id": mutation.mutation_id,
        "semantic_key": mutation.semantic_key,
        "family": mutation.family,
        "category": mutation.category,
        "descriptor": mutation.descriptor,
        "margin": simulation["margin"],
        "delta_margin": simulation["margin"] - baseline["margin"],
        "candidate_money": simulation["candidate_money"],
        "opponent_money": simulation["opponent_money"],
        "mutated_step_sanitizer_changes": simulation["mutated_step_sanitizer_changes"],
    }


def evaluate_episode(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    replay_path = Path(job["replay_path"])
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    candidate_seat = int(row["agent_seat"])
    actions = [copy.deepcopy(frame_action(frame, candidate_seat)) for frame in replay["steps"][1:]]
    baseline = simulate(
        replay,
        candidate_seat,
        {},
        diagnose_all_sanitizer_steps=True,
    )
    expected_margin = float(row["baseline_margin"])
    if baseline["margin"] != expected_margin:
        raise AssertionError(
            f"episode {row['episode_id']} baseline {baseline['margin']} != {expected_margin}"
        )
    if baseline["final_step"] != 719:
        raise AssertionError(f"episode {row['episode_id']} did not finish")

    mutations, catalog = enumerate_mutations(actions)
    if job["profile"] == "core_macro":
        mutations = [
            mutation
            for mutation in mutations
            if mutation.descriptor["surface"] == "market" and mutation.descriptor["verb"] != "SELL"
        ]
    elif job["profile"] == "priority_macro":
        mutations = [
            mutation
            for mutation in mutations
            if mutation.descriptor["surface"] == "market"
            and mutation.descriptor["verb"] != "SELL"
            and mutation.descriptor["step"] <= 270
            and mutation.family in {"market_drop", "market_quantity", "market_item_swap"}
        ]
    variant_results: list[dict[str, Any]] = []
    for mutation in mutations:
        simulation = simulate(
            replay,
            candidate_seat,
            dict(mutation.overrides),
        )
        if simulation["final_step"] != 719:
            raise AssertionError(
                f"episode {row['episode_id']} mutation {mutation.mutation_id} incomplete"
            )
        variant_results.append(compact_result(mutation, simulation, baseline))

    variant_results.sort(
        key=lambda item: (
            -item["margin"],
            item["mutated_step_sanitizer_changes"],
            item["mutation_id"],
        )
    )
    best_by_family: dict[str, dict[str, Any]] = {}
    for result in variant_results:
        best_by_family.setdefault(result["family"], result)
    catalog_counts = Counter(event["category"] for event in catalog)
    verb_counts = Counter(event["verb"] for event in catalog)
    return {
        "episode_id": int(row["episode_id"]),
        "role": job["role"],
        "candidate_seat": candidate_seat,
        "opponent_tape_seat": int(row["opponent_tape_seat"]),
        "opponent": row["opponent"],
        "replay_path": str(replay_path.relative_to(ROOT)),
        "replay_sha256": sha256(replay_path),
        "baseline": baseline,
        "variant_count": len(variant_results),
        "catalog_counts": dict(sorted(catalog_counts.items())),
        "catalog_verb_counts": dict(sorted(verb_counts.items())),
        "catalog": catalog,
        "best": variant_results[0] if variant_results else None,
        "best_by_family": best_by_family,
        "top_variants": variant_results[:25],
        "variant_results": variant_results,
    }


def semantic_rule_summary(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[int, dict[str, dict[str, Any]]] = {}
    roles: dict[int, str] = {}
    for episode in episodes:
        episode_id = episode["episode_id"]
        roles[episode_id] = episode["role"]
        by_episode[episode_id] = {row["semantic_key"]: row for row in episode["variant_results"]}
    common_keys = set.intersection(*(set(rows) for rows in by_episode.values()))
    summaries: list[dict[str, Any]] = []
    for key in common_keys:
        outcomes = []
        for episode_id, rows in by_episode.items():
            row = rows[key]
            outcomes.append(
                {
                    "episode_id": episode_id,
                    "role": roles[episode_id],
                    "margin": row["margin"],
                    "delta_margin": row["delta_margin"],
                    "mutated_step_sanitizer_changes": row["mutated_step_sanitizer_changes"],
                }
            )
        targets = [row for row in outcomes if row["role"] == "target"]
        controls = [row for row in outcomes if row["role"] == "control"]
        first = by_episode[episodes[0]["episode_id"]][key]
        summaries.append(
            {
                "semantic_key": key,
                "family": first["family"],
                "category": first["category"],
                "target_positive_count": sum(row["margin"] > 0 for row in targets),
                "target_worst_margin": min(row["margin"] for row in targets),
                "control_no_regression_count": sum(row["delta_margin"] >= 0 for row in controls),
                "control_worst_delta": min((row["delta_margin"] for row in controls), default=0.0),
                "mean_delta": fmean(row["delta_margin"] for row in outcomes),
                "max_mutated_step_sanitizer_changes": max(
                    row["mutated_step_sanitizer_changes"] for row in outcomes
                ),
                "outcomes": outcomes,
            }
        )
    summaries.sort(
        key=lambda row: (
            -row["target_positive_count"],
            -row["control_no_regression_count"],
            -row["target_worst_margin"],
            -row["control_worst_delta"],
            -row["mean_delta"],
            row["semantic_key"],
        )
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
    )
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--variant-rows", type=Path, default=VARIANT_ROWS_PATH)
    parser.add_argument("--hashes", type=Path, default=HASHES_PATH)
    parser.add_argument("--episode-id", action="append", type=int, default=[])
    parser.add_argument("--include-full-variants", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("priority_macro", "core_macro", "full"),
        default="priority_macro",
        help="core_macro excludes the already-closed sell-only path and unit micro-edits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}: {actual} != {expected}")

    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    for role, key in (("target", "target_losses"), ("control", "regression_controls")):
        for row in gate[key]:
            if args.episode_id and int(row["episode_id"]) not in args.episode_id:
                continue
            replay_path = REPLAY_DIR / f"{row['episode_id']}.json"
            if sha256(replay_path) != row["replay_sha256"]:
                raise RuntimeError(f"replay hash mismatch: {replay_path}")
            jobs.append(
                {
                    "row": row,
                    "role": role,
                    "replay_path": str(replay_path),
                    "profile": args.profile,
                }
            )

    if args.workers == 1:
        episodes = [evaluate_episode(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            episodes = list(pool.map(evaluate_episode, jobs))
    episodes.sort(key=lambda row: row["episode_id"])

    semantic_rules = semantic_rule_summary(episodes)
    hindsight_rows = []
    for episode in episodes:
        baseline_choice = {
            "mutation_id": "ABSTAIN_KEEP_V67",
            "family": "abstain",
            "category": "abstain",
            "semantic_key": "ABSTAIN_KEEP_V67",
            "descriptor": {"operation": "abstain"},
            "margin": episode["baseline"]["margin"],
            "delta_margin": 0.0,
            "mutated_step_sanitizer_changes": 0,
        }
        candidates = [baseline_choice, *episode["variant_results"]]
        best = sorted(
            candidates,
            key=lambda row: (
                -row["margin"],
                row["mutated_step_sanitizer_changes"],
                row["mutation_id"],
            ),
        )[0]
        hindsight_rows.append(
            {
                "episode_id": episode["episode_id"],
                "role": episode["role"],
                "baseline_margin": episode["baseline"]["margin"],
                "oracle_margin": best["margin"],
                "delta_margin": best["delta_margin"],
                "choice": best,
            }
        )

    target_hindsight = [row for row in hindsight_rows if row["role"] == "target"]
    control_hindsight = [row for row in hindsight_rows if row["role"] == "control"]
    strict_common = [
        row
        for row in semantic_rules
        if row["target_positive_count"] == 4 and row["control_no_regression_count"] == 4
    ]

    variant_digest = hashlib.sha256()
    args.variant_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.variant_rows.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            for row in episode["variant_results"]:
                output_row = {"episode_id": episode["episode_id"], "role": episode["role"], **row}
                line = canonical(output_row)
                handle.write(line + "\n")
                variant_digest.update((line + "\n").encode())

    result = {
        "schema_version": 1,
        "analysis": "v80_v67_residual_action_oracle",
        "scope": {
            "gate": "consumed Gate8 only",
            "opening_preserved_through_step": START_STEP - 1,
            "candidate_continuation": (
                "uploaded V67 raw actions preserved byte-for-byte except one local mutation"
            ),
            "opponent_continuation": "uploaded opponent tape preserved byte-for-byte",
            "engine": "official Rust binding",
            "forbidden_sets_used": [],
            "live_proxy_seeds_used": [],
            "candidate_created": False,
            "upload_performed": False,
        },
        "inputs": {
            "gate": {"path": str(GATE_PATH.relative_to(ROOT)), "sha256": sha256(GATE_PATH)},
            "agent": {"path": str(V67_PATH.relative_to(ROOT)), "sha256": sha256(V67_PATH)},
            "native_engine": {
                "path": str(NATIVE_ENGINE_PATH.relative_to(ROOT)),
                "sha256": sha256(NATIVE_ENGINE_PATH),
            },
        },
        "grammar": {
            "families": sorted(
                {row["family"] for episode in episodes for row in episode["variant_results"]}
            ),
            "semantics": "one event mutation; market shifts touch source plus adjacent transition",
            "action_default": "ABSTAIN_KEEP_V67",
            "search_start_step": START_STEP,
            "profile": args.profile,
        },
        "hindsight_oracle": {
            "deployable": False,
            "uses_completed_episode_identity": True,
            "rows": hindsight_rows,
            "target_positive_count": sum(row["oracle_margin"] > 0 for row in target_hindsight),
            "control_no_regression_count": sum(
                row["delta_margin"] >= 0 for row in control_hindsight
            ),
            "target_worst_margin": min(row["oracle_margin"] for row in target_hindsight),
            "control_worst_delta": min(
                (row["delta_margin"] for row in control_hindsight), default=0.0
            ),
        },
        "common_semantic_rule_screen": {
            "definition": (
                "same event-relative semantic mutation applied in all eight episodes; "
                "no observation threshold"
            ),
            "common_rule_count": len(semantic_rules),
            "strict_pass_count": len(strict_common),
            "strict_passes": strict_common[:25],
            "ranked_top": semantic_rules[:50],
        },
        "episodes": [
            (
                episode
                if args.include_full_variants
                else {key: value for key, value in episode.items() if key != "variant_results"}
            )
            for episode in episodes
        ],
        "search": {
            "total_variant_simulations": sum(episode["variant_count"] for episode in episodes),
            "variant_rows_path": str(args.variant_rows.resolve().relative_to(ROOT)),
            "variant_rows_sha256": variant_digest.hexdigest(),
        },
    }
    result["status"] = (
        "HEADROOM_ONLY_HINDSIGHT"
        if result["hindsight_oracle"]["target_positive_count"] == 4
        and result["hindsight_oracle"]["control_no_regression_count"] == 4
        else "STOP_NO_GRAMMAR_HEADROOM"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    hash_lines = []
    for path in (Path(__file__), args.output, args.variant_rows):
        hash_lines.append(f"{sha256(path)}  {path.resolve().relative_to(ROOT)}")
    args.hashes.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "status": result["status"],
                "total_variant_simulations": result["search"]["total_variant_simulations"],
                "target_positive_count": result["hindsight_oracle"]["target_positive_count"],
                "target_worst_margin": result["hindsight_oracle"]["target_worst_margin"],
                "control_no_regression_count": result["hindsight_oracle"][
                    "control_no_regression_count"
                ],
                "strict_common_rule_count": len(strict_common),
            }
        )
    )


if __name__ == "__main__":
    main()
