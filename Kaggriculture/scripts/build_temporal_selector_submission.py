"""Build a standalone Kaggle bundle for a temporal tape-selector checkpoint."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import inspect
import json
import tarfile
import zlib
from pathlib import Path

import numpy as np

from kaggriculture_rl.imitation.spatial_dataset import (
    _scaled,
    encode_spatial_observation,
)
from kaggriculture_rl.learning.dynamic_tape_retriever import (
    MAX_RETRIEVER_UNITS,
    encode_retriever_features,
)
from kaggriculture_rl.learning.tape_selector import (
    TapeSelectorCheckpoint,
    TapeSelectorConfig,
    _load_trace,
    _tile_features,
    encode_selector_features,
    load_selector_checkpoint,
    predict_q,
    select_action,
)
from kaggriculture_rl.policy import primitive_legalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--base-action-id", type=int)
    parser.add_argument(
        "--state-aware-repair",
        action="store_true",
        help="embed weed recovery, terminal liquidation, and exact legalizer",
    )
    parser.add_argument(
        "--semantic-unit-repair",
        action="store_true",
        help="recover rejected tape commands with useful local semantic tasks",
    )
    parser.add_argument(
        "--semantic-unit-operation",
        action="append",
        default=[],
        help="restrict semantic repair to this fallback operation (repeatable)",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help="optional earlier exact-state gate applied before stage two",
    )
    parser.add_argument(
        "--stage1-safe-route",
        action="append",
        default=[],
        metavar="REPRESENTATIVE_EPISODE:ACTION_ID",
        help="route an exact stage-one feature row to a suffix (repeatable)",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=Path,
        help="optional aggregate checkpoint supplying a later exact-state gate",
    )
    parser.add_argument(
        "--stage2-safe-route",
        action="append",
        default=[],
        metavar="REPRESENTATIVE_EPISODE:ACTION_ID",
        help="route an exact stage-two feature row to a suffix (repeatable)",
    )
    parser.add_argument(
        "--stage3-checkpoint",
        type=Path,
        help="optional final exact-state gate applied after stage two",
    )
    parser.add_argument(
        "--stage3-safe-route",
        action="append",
        default=[],
        metavar="REPRESENTATIVE_EPISODE:ACTION_ID",
        help="route an exact stage-three feature row to a suffix (repeatable)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _encoded(value: bytes) -> str:
    return base64.b85encode(value).decode("ascii")


def _literal(name: str, value: str) -> str:
    chunks = [value[index : index + 100] for index in range(0, len(value), 100)]
    body = "\n".join(f'    "{chunk}"' for chunk in chunks)
    return f"{name} = (\n{body}\n)\n"


def _safe_route(value: str) -> tuple[int, int]:
    try:
        representative, action_id = value.split(":", maxsplit=1)
        return int(representative), int(action_id)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "stage-two routes must be REPRESENTATIVE_EPISODE:ACTION_ID"
        ) from error


def _resolve_exact_stage(
    checkpoint_path: Path | None,
    routes: tuple[tuple[int, int], ...],
) -> tuple[
    TapeSelectorCheckpoint | None,
    str,
    tuple[tuple[int, np.ndarray], ...],
]:
    if bool(checkpoint_path) != bool(routes):
        raise ValueError("exact-stage checkpoint and routes must be supplied together")
    if checkpoint_path is None:
        return None, "aggregate", ()

    checkpoint = load_selector_checkpoint(checkpoint_path)
    encoder = str(checkpoint.metadata.get("feature_encoder", "aggregate"))
    if encoder not in {"aggregate", "retriever"}:
        raise ValueError(
            "standalone exact stages support aggregate or retriever features"
        )
    context_ids = tuple(
        int(value)
        for value in checkpoint.metadata.get("context_episode_ids", ())
    )
    if len(context_ids) != len(checkpoint.train_features):
        raise ValueError("exact-stage checkpoint needs one context id per training row")
    row_by_episode = {
        episode_id: checkpoint.train_features[index]
        for index, episode_id in enumerate(context_ids)
    }
    resolved: list[tuple[int, np.ndarray]] = []
    for representative_id, action_id in routes:
        try:
            target = row_by_episode[representative_id].copy()
        except KeyError as error:
            raise ValueError(
                f"unknown exact-stage representative: {representative_id}"
            ) from error
        for existing_action, existing_target in resolved:
            if np.array_equal(target, existing_target):
                if existing_action != action_id:
                    raise ValueError(
                        "conflicting suffixes for the same exact-stage feature row"
                    )
                break
        else:
            resolved.append((action_id, target))
    return checkpoint, encoder, tuple(resolved)


def _stage_constants(
    name: str,
    checkpoint: TapeSelectorCheckpoint | None,
    route_rows: tuple[tuple[int, np.ndarray], ...],
) -> str:
    if checkpoint is None:
        return (
            f"_{name}_BRANCH_STEP = None\n"
            f"_{name}_FEATURE_STEPS = ()\n"
            f"_{name}_FEATURE_MEAN = None\n"
            f"_{name}_FEATURE_SCALE = None\n"
            f"_{name}_ROUTES = ()\n"
        )
    feature_steps = tuple(
        int(value) for value in checkpoint.metadata["feature_steps"]
    )
    routes = "".join(
        "    ("
        f"{int(action_id)!r}, np.array({target.tolist()!r}, dtype=np.float64)"
        "),\n"
        for action_id, target in route_rows
    )
    return (
        f"_{name}_BRANCH_STEP = {checkpoint.config.branch_step!r}\n"
        f"_{name}_FEATURE_STEPS = {feature_steps!r}\n"
        f"_{name}_FEATURE_MEAN = np.array("
        f"{checkpoint.feature_mean.tolist()!r}, dtype=np.float64)\n"
        f"_{name}_FEATURE_SCALE = np.array("
        f"{checkpoint.feature_scale.tolist()!r}, dtype=np.float64)\n"
        f"_{name}_ROUTES = (\n{routes})\n"
    )


def _encode_spatial_selector_features(observation: dict[str, object]) -> np.ndarray:
    spatial, global_features = encode_spatial_observation(observation)
    return np.concatenate((spatial.reshape(-1), global_features))


def _standalone_legalizer_source() -> str:
    module_path = Path(primitive_legalizer.__file__)
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = []
    for node in tree.body:
        if isinstance(
            node,
            ast.Assign | ast.AnnAssign | ast.FunctionDef | ast.ClassDef,
        ):
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                definitions.append(segment)
    return "\n\n".join(definitions)


def _source(
    checkpoint_path: Path,
    population_path: Path,
    base_action_id: int | None,
    *,
    state_aware_repair: bool = False,
    semantic_unit_repair: bool = False,
    semantic_unit_operations: tuple[str, ...] = (),
    stage1_checkpoint_path: Path | None = None,
    stage1_safe_routes: tuple[tuple[int, int], ...] = (),
    stage2_checkpoint_path: Path | None = None,
    stage2_safe_routes: tuple[tuple[int, int], ...] = (),
    stage3_checkpoint_path: Path | None = None,
    stage3_safe_routes: tuple[tuple[int, int], ...] = (),
) -> str:
    if semantic_unit_repair and not state_aware_repair:
        raise ValueError("semantic unit repair requires state-aware repair")
    if semantic_unit_operations and not semantic_unit_repair:
        raise ValueError("semantic unit operations require semantic unit repair")
    unknown_operations = set(semantic_unit_operations) - {
        "DIG",
        "HARVEST",
        "WATER",
        "FERTILIZE",
        "FEED",
        "CARE",
        "COLLECT_FERTILIZER",
        "PLACE",
        "DROP",
    }
    if unknown_operations:
        raise ValueError(f"unknown semantic unit operations: {sorted(unknown_operations)}")
    checkpoint = load_selector_checkpoint(checkpoint_path)
    stage1_checkpoint, stage1_encoder, stage1_route_rows = _resolve_exact_stage(
        stage1_checkpoint_path,
        stage1_safe_routes,
    )
    stage2_checkpoint, stage2_encoder, stage2_route_rows = _resolve_exact_stage(
        stage2_checkpoint_path,
        stage2_safe_routes,
    )
    stage3_checkpoint, stage3_encoder, stage3_route_rows = _resolve_exact_stage(
        stage3_checkpoint_path,
        stage3_safe_routes,
    )
    population = json.loads(population_path.read_text(encoding="utf-8"))
    rows = {int(row["episode_id"]): row for row in population["agents"]}
    selected_base_action = (
        int(base_action_id) if base_action_id is not None else int(checkpoint.action_ids[0])
    )
    required_action_ids = tuple(
        dict.fromkeys(
            (
                *checkpoint.action_ids,
                selected_base_action,
                *(action_id for action_id, _ in stage1_route_rows),
                *(action_id for action_id, _ in stage2_route_rows),
                *(action_id for action_id, _ in stage3_route_rows),
            )
        )
    )
    traces: dict[int, list[dict[str, object]]] = {}
    for index, action_id in enumerate(required_action_ids):
        row = rows[action_id]
        agent_path = Path(row["agent_file"])
        if not agent_path.exists():
            agent_path = population_path.parent / agent_path.name
        traces[action_id] = _load_trace(agent_path, index)

    checkpoint_blob = _encoded(checkpoint_path.read_bytes())
    trace_json = json.dumps(traces, separators=(",", ":"), sort_keys=True).encode()
    trace_blob = _encoded(zlib.compress(trace_json, level=9))
    feature_encoder = str(checkpoint.metadata.get("feature_encoder", "aggregate"))
    feature_definitions = [
        inspect.getsource(_tile_features).strip(),
        inspect.getsource(encode_selector_features).strip(),
    ]
    feature_call = "encode_selector_features(observation)"
    if feature_encoder == "spatial":
        feature_definitions.extend(
            [
                inspect.getsource(_scaled).strip(),
                inspect.getsource(encode_spatial_observation).strip(),
                inspect.getsource(_encode_spatial_selector_features).strip(),
            ]
        )
        feature_call = "_encode_spatial_selector_features(observation)"
    elif feature_encoder != "aggregate":
        raise ValueError(f"standalone builder does not support feature encoder {feature_encoder!r}")
    feature_definitions.append(f"def _encode_features(observation):\n    return {feature_call}")
    stage1_feature_call = "encode_selector_features(observation)"
    stage2_feature_call = "encode_selector_features(observation)"
    stage3_feature_call = "encode_selector_features(observation)"
    if "retriever" in {stage1_encoder, stage2_encoder, stage3_encoder}:
        feature_definitions.append(inspect.getsource(encode_retriever_features).strip())
    if stage1_encoder == "retriever":
        stage1_feature_call = "encode_retriever_features(observation)"
    if stage2_encoder == "retriever":
        stage2_feature_call = "encode_retriever_features(observation)"
    if stage3_encoder == "retriever":
        stage3_feature_call = "encode_retriever_features(observation)"
    feature_definitions.append(
        "def _encode_stage1_features(observation):\n"
        f"    return {stage1_feature_call}"
    )
    feature_definitions.append(
        "def _encode_stage2_features(observation):\n"
        f"    return {stage2_feature_call}"
    )
    feature_definitions.append(
        "def _encode_stage3_features(observation):\n"
        f"    return {stage3_feature_call}"
    )
    definitions = "\n\n".join(
        [
            inspect.getsource(TapeSelectorConfig).strip(),
            inspect.getsource(TapeSelectorCheckpoint).strip(),
            *feature_definitions,
            inspect.getsource(predict_q).strip(),
            inspect.getsource(select_action).strip(),
        ]
    )
    if state_aware_repair:
        definitions += "\n\n" + _standalone_legalizer_source()
    allowed_operations = (
        f"frozenset({tuple(sorted(set(semantic_unit_operations)))!r})"
        if semantic_unit_operations
        else "None"
    )
    stage1_constants = _stage_constants(
        "STAGE1", stage1_checkpoint, stage1_route_rows
    )
    stage2_constants = _stage_constants(
        "STAGE2", stage2_checkpoint, stage2_route_rows
    )
    stage3_constants = _stage_constants(
        "STAGE3", stage3_checkpoint, stage3_route_rows
    )
    agent_definition = (
        f"""
_RECOVERY = WeedRecoveryController()


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    proposed = _POLICY(observation)
    repaired = _RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    legalized = legalize_action(observation, liquidated, configuration)
    return recover_rejected_unit_intents(
        observation,
        liquidated,
        legalized,
        configuration,
        allowed_operations={allowed_operations},
    )
"""
        if semantic_unit_repair
        else """
_RECOVERY = WeedRecoveryController()


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    proposed = _POLICY(observation)
    repaired = _RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    return legalize_action(observation, liquidated, configuration)
"""
        if state_aware_repair
        else """
def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    return _POLICY(observation)
"""
    )
    return (
        '"""Standalone state-adaptive Kaggriculture agent."""\n\n'
        "from __future__ import annotations\n\n"
        "import base64\n"
        "import copy\n"
        "import io\n"
        "import json\n"
        "import math\n"
        "import zlib\n"
        "from collections import Counter\n"
        "from dataclasses import dataclass\n"
        "from typing import Any\n\n"
        "import numpy as np\n\n"
        'CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")\n'
        'ANIMALS = ("GOOSE", "COW", "SHEEP")\n'
        'ITEMS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER", *ANIMALS)\n\n'
        'PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")\n'
        "SHED_ITEMS = (*PRODUCTS, *ANIMALS)\n"
        'SHOPS = ("BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", '
        '"ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", '
        '"YARN_STORE")\n'
        "BOARD_SIZE = 10\n"
        "SPATIAL_CHANNELS = 20\n"
        "GLOBAL_FEATURES = 68\n"
        "MAX_HANDS = 12\n"
        f"MAX_RETRIEVER_UNITS = {MAX_RETRIEVER_UNITS}\n"
        f"_BASE_ACTION_ID = {selected_base_action}\n\n"
        f"{stage1_constants}\n"
        f"{stage2_constants}\n"
        f"{stage3_constants}\n"
        f"{definitions}\n\n"
        + _literal("_CHECKPOINT_BLOB", checkpoint_blob)
        + "\n"
        + _literal("_TRACE_BLOB", trace_blob)
        + """
with np.load(
    io.BytesIO(base64.b85decode(_CHECKPOINT_BLOB.encode("ascii"))),
    allow_pickle=False,
) as _payload:
    _CHECKPOINT = TapeSelectorCheckpoint(
        config=TapeSelectorConfig(**json.loads(str(_payload["config_json"]))),
        feature_mean=_payload["feature_mean"].copy(),
        feature_scale=_payload["feature_scale"].copy(),
        train_features=_payload["train_features"].copy(),
        q_values=_payload["q_values"].copy(),
        global_action=int(_payload["global_action"]),
        action_ids=tuple(int(value) for value in _payload["action_ids"]),
        metadata=json.loads(str(_payload["metadata_json"])),
    )
_TRACES = {
    int(action_id): trace
    for action_id, trace in json.loads(
        zlib.decompress(base64.b85decode(_TRACE_BLOB.encode("ascii")))
    ).items()
}


class _StandalonePolicy:
    def __init__(self) -> None:
        self.checkpoint = _CHECKPOINT
        self.traces = _TRACES
        self.feature_steps = tuple(
            int(step) for step in self.checkpoint.metadata["feature_steps"]
        )
        self._common_action = _BASE_ACTION_ID
        self._selected: dict[int, int] = {}
        self._snapshots: dict[int, dict[int, np.ndarray]] = {}
        self._stage1_selected: dict[int, int] = {}
        self._stage1_snapshots: dict[int, dict[int, np.ndarray]] = {}
        self._stage2_selected: dict[int, int] = {}
        self._stage2_snapshots: dict[int, dict[int, np.ndarray]] = {}
        self._stage3_selected: dict[int, int] = {}
        self._stage3_snapshots: dict[int, dict[int, np.ndarray]] = {}
        self._last_step: dict[int, int] = {}

    def reset(self) -> None:
        self._selected.clear()
        self._snapshots.clear()
        self._stage1_selected.clear()
        self._stage1_snapshots.clear()
        self._stage2_selected.clear()
        self._stage2_snapshots.clear()
        self._stage3_selected.clear()
        self._stage3_snapshots.clear()
        self._last_step.clear()

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self._last_step.get(player, -1)
        if step < previous_step:
            self._selected.pop(player, None)
            self._snapshots.pop(player, None)
            self._stage1_selected.pop(player, None)
            self._stage1_snapshots.pop(player, None)
            self._stage2_selected.pop(player, None)
            self._stage2_snapshots.pop(player, None)
            self._stage3_selected.pop(player, None)
            self._stage3_snapshots.pop(player, None)
        self._last_step[player] = step
        if step in self.feature_steps:
            self._snapshots.setdefault(player, {})[step] = _encode_features(observation)
        if step in _STAGE1_FEATURE_STEPS:
            self._stage1_snapshots.setdefault(player, {})[step] = (
                _encode_stage1_features(observation)
            )
        if step in _STAGE2_FEATURE_STEPS:
            self._stage2_snapshots.setdefault(player, {})[step] = (
                _encode_stage2_features(observation)
            )
        if step in _STAGE3_FEATURE_STEPS:
            self._stage3_snapshots.setdefault(player, {})[step] = (
                _encode_stage3_features(observation)
            )
        if step < self.checkpoint.config.branch_step:
            action_id = self._common_action
        elif player not in self._selected:
            snapshots = self._snapshots.setdefault(player, {})
            current = _encode_features(observation)
            temporal = (
                np.concatenate(
                    [
                        snapshots.get(feature_step, current)
                        for feature_step in self.feature_steps
                    ]
                )
                if self.feature_steps
                else current
            )
            action_id, _, _ = select_action(self.checkpoint, temporal)
            self._selected[player] = action_id
        else:
            action_id = self._selected[player]
        if (
            _STAGE1_BRANCH_STEP is not None
            and step >= _STAGE1_BRANCH_STEP
            and player not in self._stage1_selected
        ):
            stage1_snapshots = self._stage1_snapshots.setdefault(player, {})
            missing = set(_STAGE1_FEATURE_STEPS) - stage1_snapshots.keys()
            routed_action = -1
            if not missing:
                stage1_features = np.concatenate(
                    [stage1_snapshots[value] for value in _STAGE1_FEATURE_STEPS]
                )
                normalized = (
                    stage1_features - _STAGE1_FEATURE_MEAN
                ) / _STAGE1_FEATURE_SCALE
                for candidate_action, target in _STAGE1_ROUTES:
                    if np.array_equal(normalized, target):
                        routed_action = candidate_action
                        break
            self._stage1_selected[player] = routed_action
        routed_action = self._stage1_selected.get(player, -1)
        if routed_action != -1:
            action_id = routed_action
        if (
            _STAGE2_BRANCH_STEP is not None
            and step >= _STAGE2_BRANCH_STEP
            and player not in self._stage2_selected
        ):
            stage2_snapshots = self._stage2_snapshots.setdefault(player, {})
            missing = set(_STAGE2_FEATURE_STEPS) - stage2_snapshots.keys()
            routed_action = -1
            if not missing:
                stage2_features = np.concatenate(
                    [stage2_snapshots[value] for value in _STAGE2_FEATURE_STEPS]
                )
                normalized = (
                    stage2_features - _STAGE2_FEATURE_MEAN
                ) / _STAGE2_FEATURE_SCALE
                for candidate_action, target in _STAGE2_ROUTES:
                    if np.array_equal(normalized, target):
                        routed_action = candidate_action
                        break
            self._stage2_selected[player] = routed_action
        routed_action = self._stage2_selected.get(player, -1)
        if routed_action != -1:
            action_id = routed_action
        if (
            _STAGE3_BRANCH_STEP is not None
            and step >= _STAGE3_BRANCH_STEP
            and player not in self._stage3_selected
        ):
            stage3_snapshots = self._stage3_snapshots.setdefault(player, {})
            missing = set(_STAGE3_FEATURE_STEPS) - stage3_snapshots.keys()
            routed_action = -1
            if not missing:
                stage3_features = np.concatenate(
                    [stage3_snapshots[value] for value in _STAGE3_FEATURE_STEPS]
                )
                normalized = (
                    stage3_features - _STAGE3_FEATURE_MEAN
                ) / _STAGE3_FEATURE_SCALE
                for candidate_action, target in _STAGE3_ROUTES:
                    if np.array_equal(normalized, target):
                        routed_action = candidate_action
                        break
            self._stage3_selected[player] = routed_action
        routed_action = self._stage3_selected.get(player, -1)
        if routed_action != -1:
            action_id = routed_action
        trace = self.traces[action_id]
        return copy.deepcopy(trace[min(step, len(trace) - 1)])


_POLICY = _StandalonePolicy()
"""
        + agent_definition
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.output_dir / "main.py"
    source = _source(
        args.checkpoint,
        args.population,
        args.base_action_id,
        state_aware_repair=args.state_aware_repair,
        semantic_unit_repair=args.semantic_unit_repair,
        semantic_unit_operations=tuple(args.semantic_unit_operation),
        stage1_checkpoint_path=args.stage1_checkpoint,
        stage1_safe_routes=tuple(_safe_route(value) for value in args.stage1_safe_route),
        stage2_checkpoint_path=args.stage2_checkpoint,
        stage2_safe_routes=tuple(_safe_route(value) for value in args.stage2_safe_route),
        stage3_checkpoint_path=args.stage3_checkpoint,
        stage3_safe_routes=tuple(_safe_route(value) for value in args.stage3_safe_route),
    )
    compile(source, str(main_path), "exec")
    main_path.write_text(source, encoding="utf-8")
    archive_path = args.output_dir / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")
    print(
        json.dumps(
            {
                "main": str(main_path),
                "main_bytes": main_path.stat().st_size,
                "main_sha256": hashlib.sha256(main_path.read_bytes()).hexdigest(),
                "archive": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "state_aware_repair": args.state_aware_repair,
                "semantic_unit_repair": args.semantic_unit_repair,
                "semantic_unit_operations": sorted(set(args.semantic_unit_operation)),
                "stage1_checkpoint": (
                    str(args.stage1_checkpoint) if args.stage1_checkpoint else None
                ),
                "stage1_safe_routes": args.stage1_safe_route,
                "stage2_checkpoint": (
                    str(args.stage2_checkpoint) if args.stage2_checkpoint else None
                ),
                "stage2_safe_routes": args.stage2_safe_route,
                "stage3_checkpoint": (
                    str(args.stage3_checkpoint) if args.stage3_checkpoint else None
                ),
                "stage3_safe_routes": args.stage3_safe_route,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
