#!/usr/bin/env python3
"""Measure how portfolio routes rescue losses of one fixed base route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_top20_route_selector import _load_dataset, _summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--base-action-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _row_key(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "team": row["team"],
        "episode_id": int(row["episode_id"]),
        "top1_seat": int(row["top1_seat"]),
    }


def main() -> None:
    args = _parser().parse_args()
    portfolio = json.loads(args.portfolio.read_text(encoding="utf-8"))
    _, q_values, rows, action_ids = _load_dataset(
        portfolio,
        args.replay_root,
        (0,),
        "compact",
    )
    try:
        base_index = action_ids.index(args.base_action_id)
    except ValueError as error:
        raise ValueError(f"base action {args.base_action_id} not in portfolio") from error

    base_margins = q_values[:, base_index]
    base_loss_indices = np.flatnonzero(base_margins <= 0.0)
    base_win_indices = np.flatnonzero(base_margins > 0.0)
    teams = [str(row["team"]) for row in rows]

    coverage: list[dict[str, Any]] = []
    for action_index, action_id in enumerate(action_ids):
        margins = q_values[:, action_index]
        rescued = base_loss_indices[margins[base_loss_indices] > 0.0]
        regressed = base_win_indices[margins[base_win_indices] <= 0.0]
        coverage.append(
            {
                "action_id": int(action_id),
                "action_index": action_index,
                "rescued_base_losses": len(rescued),
                "regressed_base_wins": len(regressed),
                "net_win_change": int(len(rescued) - len(regressed)),
                "rescued_contexts": [_row_key(rows[index]) for index in rescued],
                "regressed_contexts": [_row_key(rows[index]) for index in regressed],
                "summary": _summary(
                    margins.tolist(),
                    [action_index] * len(rows),
                    teams,
                ),
            }
        )
    coverage.sort(
        key=lambda item: (
            item["rescued_base_losses"],
            item["net_win_change"],
            item["summary"]["mean_margin"],
        ),
        reverse=True,
    )

    oracle_actions = np.argmax(q_values, axis=1)
    loss_rows: list[dict[str, Any]] = []
    for row_index in base_loss_indices:
        ranked = np.argsort(q_values[row_index])[::-1]
        loss_rows.append(
            {
                **_row_key(rows[row_index]),
                "base_margin": float(base_margins[row_index]),
                "oracle_action_id": int(action_ids[int(oracle_actions[row_index])]),
                "oracle_margin": float(q_values[row_index, oracle_actions[row_index]]),
                "top_routes": [
                    {
                        "action_id": int(action_ids[int(action_index)]),
                        "margin": float(q_values[row_index, action_index]),
                    }
                    for action_index in ranked[:5]
                ],
            }
        )
    loss_rows.sort(key=lambda item: item["base_margin"])

    remaining = set(int(index) for index in base_loss_indices)
    greedy_cover: list[dict[str, Any]] = []
    while remaining:
        best_action: int | None = None
        best_rescues: set[int] = set()
        for action_index in range(len(action_ids)):
            rescues = {index for index in remaining if q_values[index, action_index] > 0.0}
            if len(rescues) > len(best_rescues):
                best_action = action_index
                best_rescues = rescues
        if best_action is None or not best_rescues:
            break
        greedy_cover.append(
            {
                "action_id": int(action_ids[best_action]),
                "new_rescues": len(best_rescues),
                "remaining_after": len(remaining - best_rescues),
            }
        )
        remaining -= best_rescues

    payload = {
        "portfolio": str(args.portfolio),
        "replay_root": str(args.replay_root),
        "contexts": len(rows),
        "action_ids": action_ids,
        "base_action_id": args.base_action_id,
        "base": _summary(
            base_margins.tolist(),
            [base_index] * len(rows),
            teams,
        ),
        "base_losses": len(base_loss_indices),
        "oracle_rescued_base_losses": int(
            np.sum(np.max(q_values[base_loss_indices], axis=1) > 0.0)
        ),
        "coverage": coverage,
        "greedy_loss_cover": greedy_cover,
        "uncovered_base_losses": [_row_key(rows[index]) for index in sorted(remaining)],
        "losses": loss_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "base": payload["base"],
                "base_losses": payload["base_losses"],
                "oracle_rescued_base_losses": payload["oracle_rescued_base_losses"],
                "coverage": coverage[:6],
                "greedy_loss_cover": greedy_cover,
                "uncovered_base_losses": payload["uncovered_base_losses"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
