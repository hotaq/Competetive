"""Build the self-contained Kaggle targeted local-validation notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "our-local-validation"
OUTPUT_PATH = OUTPUT_DIR / "local-validation-smoke.ipynb"

EMBEDDED_FILES = (
    "attack.py",
    "src/__init__.py",
    "src/analysis/__init__.py",
    "src/analysis/traces.py",
    "src/strategies/__init__.py",
    "src/strategies/archive.py",
    "src/strategies/content_guided.py",
    "src/strategies/static_portfolio.py",
    "configs/targeted-multihop-experiments.json",
)


def embedded_source_cell() -> str:
    sources = {
        relative_path: (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in EMBEDDED_FILES
    }
    serialized = json.dumps(sources, ensure_ascii=False)
    return f"""from pathlib import Path
import json

embedded_sources = json.loads({serialized!r})
for relative_path, source in embedded_sources.items():
    destination = WORK_DIR / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")

print("Wrote embedded attack package:")
for relative_path in embedded_sources:
    print(" -", relative_path)
"""


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            """# AI Agent Security — targeted multi-hop validation

## Goal

Replay a manifest-selected static candidate against the competition-matched Gym
environment, `OptimalGuardrail`, GPT-OSS, and Gemma. The exact message already
produced a scored GPT-OSS finding in version 11; this run tests cross-model
reliability without spending time on discovery."""
        ),
        new_markdown_cell(
            """## Setup

### Key assumptions

- The official competition data source is attached.
- Both GGUF model sources are attached.
- The Gemma model-access agreement has been accepted.
- Internet is enabled only for installing `llama-cpp-python`; final competition
  submissions will keep internet disabled.
- A local score is a debugging and ranking signal, not an exact private-score
  prediction."""
        ),
        new_code_cell(
            """import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

COMP_DIR = Path("/kaggle/input/competitions/ai-agent-security-multi-step-tool-attacks")
GPT_OSS_PATH = Path(
    "/kaggle/input/models/llkh0a/gpt-oss-20b-gguf/pytorch/default/1/"
    "gpt_oss/gpt-oss-20b-Q4_K_M.gguf"
)
GEMMA_PATH = Path(
    "/kaggle/input/models/llkh0a/gemma-4-26b-a4b-it-ud-q4-k-m-gguf/"
    "pytorch/default/1/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
)
WORK_DIR = Path("/kaggle/working")
ARTIFACTS_DIR = WORK_DIR / "validation-artifacts"
ATTACK_PATH = WORK_DIR / "attack.py"

WORK_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
assert COMP_DIR.exists(), f"Missing competition SDK: {COMP_DIR}"
assert GPT_OSS_PATH.exists(), f"Missing GPT-OSS model: {GPT_OSS_PATH}"
assert GEMMA_PATH.exists(), f"Missing Gemma model or access agreement: {GEMMA_PATH}"

sys.path.insert(0, str(COMP_DIR))
sys.path.insert(0, str(WORK_DIR))
os.environ["PYTHONUTF8"] = "1"
os.environ["GPT_OSS_MODEL_PATH"] = str(GPT_OSS_PATH)
os.environ["GEMMA_MODEL_PATH"] = str(GEMMA_PATH)
os.environ["GGML_CUDA_FORCE_MMQ"] = "1"

from kaggle_evaluation.jed_attack_134815 import jed_attack_gateway as gateway

SMOKE_BUDGET_S = 300.0
TARGET_MAX_STEPS = 1
TARGET_MAX_TOOL_HOPS = min(2, gateway.DEFAULT_MAX_TOOL_HOPS)
ATTACK_SEED = gateway.ATTACK_SEED
MODEL_NAMES_TO_RUN = ("gpt_oss", "gemma")
STATIC_PORTFOLIO_SIZE = 4
TARGET_EXPERIMENT_IDS = (
    "direct_exfiltration:sdk_predicate:secret_marker_post",
)

print(json.dumps({
    "smoke_budget_s_per_model": SMOKE_BUDGET_S,
    "competition_budget_s_per_model": gateway.DEFAULT_BUDGET_S,
    "max_tool_hops": TARGET_MAX_TOOL_HOPS,
    "attack_seed": ATTACK_SEED,
    "environment": gateway.ENV_SELECTION,
    "competition_models": list(gateway.MODEL_NAMES),
    "smoke_models": list(MODEL_NAMES_TO_RUN),
    "t4_profile": {
        "n_ctx": 2048,
        "n_gpu_layers": 20,
        "max_new_tokens": 256,
        "n_batch": 128,
        "n_ubatch": 64,
    },
    "targeted_search": {
        "max_steps": TARGET_MAX_STEPS,
        "max_tool_hops": TARGET_MAX_TOOL_HOPS,
        "experiment_ids": list(TARGET_EXPERIMENT_IDS),
    },
}, indent=2))"""
        ),
        new_markdown_cell("### Install the GPU inference runtime"),
        new_code_cell(
            """if importlib.util.find_spec("llama_cpp") is None:
    print("Installing llama-cpp-python CUDA wheel...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--no-cache-dir",
            "llama-cpp-python",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cu124",
        ],
        check=True,
    )

import llama_cpp

print("llama-cpp-python ready:", getattr(llama_cpp, "__version__", "unknown"))"""
        ),
        new_markdown_cell("### Materialize the current attack package"),
        new_code_cell(embedded_source_cell()),
        new_code_cell(
            """TARGET_MANIFEST_PATH = (
    WORK_DIR / "configs" / "targeted-multihop-experiments.json"
)
from src.strategies.static_portfolio import compact_exfiltration_prompts

target_manifest = json.loads(TARGET_MANIFEST_PATH.read_text(encoding="utf-8"))
experiments_by_id = {
    experiment["experiment_id"]: experiment
    for experiment in target_manifest["experiments"]
}
missing_ids = set(TARGET_EXPERIMENT_IDS) - set(experiments_by_id)
assert not missing_ids, f"Missing targeted experiments: {sorted(missing_ids)}"
TARGETED_PROMPTS = compact_exfiltration_prompts(STATIC_PORTFOLIO_SIZE)
assert len(TARGETED_PROMPTS) == STATIC_PORTFOLIO_SIZE

targeted_batch = {
    "experiment_ids": list(TARGET_EXPERIMENT_IDS),
    "portfolio_size": STATIC_PORTFOLIO_SIZE,
    "prompts": list(TARGETED_PROMPTS),
    "max_steps": TARGET_MAX_STEPS,
    "max_tool_hops": TARGET_MAX_TOOL_HOPS,
    "time_budget_s": SMOKE_BUDGET_S,
}
(ARTIFACTS_DIR / "targeted-batch.json").write_text(
    json.dumps(targeted_batch, indent=2),
    encoding="utf-8",
)
print(json.dumps(targeted_batch, indent=2))"""
        ),
        new_code_cell(
            """from aicomp_sdk.attacks import AttackAlgorithmBase

compile(ATTACK_PATH.read_text(encoding="utf-8"), str(ATTACK_PATH), "exec")
attack_spec = importlib.util.spec_from_file_location(
    "attack_package_check",
    str(ATTACK_PATH),
)
attack_module = importlib.util.module_from_spec(attack_spec)
assert attack_spec.loader is not None
attack_spec.loader.exec_module(attack_module)
assert issubclass(attack_module.AttackAlgorithm, AttackAlgorithmBase)
print("attack.py syntax, imports, and base class validated")"""
        ),
        new_markdown_cell(
            """## Steps

Run the official red-team evaluator for both models using only the static candidate."""
        ),
        new_code_cell(
            """from dataclasses import replace

from aicomp_sdk.attacks import AttackCandidate
from aicomp_sdk.core.env.api import EnvSelection
from aicomp_sdk.evaluation.diagnostics import EvaluatorVerbosity, RunDiagnostics
from aicomp_sdk.evaluation.runner import evaluate_redteam
from kaggle_evaluation.jed_attack_134815 import (
    gemma_model_server,
    gpt_oss_model_server,
)
from kaggle_evaluation.jed_attack_134815.gguf_model_server import GgufModelServer


def t4_safe_llama(**kwargs):
    return llama_cpp.Llama(
        **kwargs,
        n_batch=128,
        n_ubatch=64,
        flash_attn=False,
        offload_kqv=False,
    )


def load_attack_class(path):
    spec = importlib.util.spec_from_file_location("competition_attack", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AttackAlgorithm


def build_model(model_name):
    base_spec = (
        gpt_oss_model_server.SPEC
        if model_name == "gpt_oss"
        else gemma_model_server.SPEC
    )
    model_spec = replace(
        base_spec,
        n_ctx=2048,
        n_gpu_layers=20,
        max_new_tokens=256,
    )
    server = GgufModelServer(model_spec, llama_cls=t4_safe_llama)
    started = time.time()
    server.load_model()
    print(f"Loaded {model_name} in {time.time() - started:.1f}s")
    return (lambda: server._load_agent()), server


def unload_model(server):
    try:
        server.unload()
    finally:
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


SubmittedAttackAlgorithm = load_attack_class(ATTACK_PATH)


class StaticReplayAttackAlgorithm(SubmittedAttackAlgorithm):
    def run(self, env, config):
        del env, config
        return [
            AttackCandidate.from_messages([prompt])
            for prompt in TARGETED_PROMPTS
        ]"""
        ),
        new_code_cell(
            """summaries = {}
for model_name in MODEL_NAMES_TO_RUN:
    print(f"\\n{'=' * 24} {model_name} {'=' * 24}")
    transcript = ARTIFACTS_DIR / f"{model_name}-transcript.log"
    framework_events = ARTIFACTS_DIR / f"{model_name}-framework.jsonl"
    agent_debug = ARTIFACTS_DIR / f"{model_name}-agent-debug.jsonl"
    agent_factory, model_server = build_model(model_name)
    started = time.time()
    try:
        with RunDiagnostics(
            EvaluatorVerbosity.DEBUG,
            transcript_file=transcript,
            event_log_file=framework_events,
            agent_debug_file=agent_debug,
        ) as diagnostics:
            execution = evaluate_redteam(
                StaticReplayAttackAlgorithm,
                budget_s=SMOKE_BUDGET_S,
                agent_factory=agent_factory,
                agent_label=f"{model_name}_gguf",
                env_selection=EnvSelection.GYM,
                fixtures_dir=COMP_DIR / "aicomp_sdk" / "fixtures",
                diagnostics=diagnostics,
                attack_env_seed=ATTACK_SEED,
            )
        result = execution.attack
        summaries[model_name] = {
            "score_normalized_0_to_1000": result.score,
            "score_raw": result.score_raw,
            "findings_count": result.findings_count,
            "unique_cells": result.unique_cells,
            "unique_canonical_cells": result.unique_canonical_cells,
            "evaluation_time_s": result.time_taken,
            "wall_time_s": time.time() - started,
            "guardrail_id": result.guardrail_id,
            "guardrail_version": result.guardrail_version,
            "environment": str(execution.env_selection),
        }
        (ARTIFACTS_DIR / f"{model_name}-summary.json").write_text(
            json.dumps(summaries[model_name], indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summaries[model_name], indent=2))
    finally:
        unload_model(model_server)"""
        ),
        new_markdown_cell("## Checks"),
        new_code_cell(
            """assert set(summaries) == set(MODEL_NAMES_TO_RUN)
assert all(summary["evaluation_time_s"] >= 0 for summary in summaries.values())

combined = {
    "models": summaries,
    "local_public_mean": sum(
        summary["score_normalized_0_to_1000"]
        for summary in summaries.values()
    ) / len(summaries),
}
(ARTIFACTS_DIR / "combined-summary.json").write_text(
    json.dumps(combined, indent=2),
    encoding="utf-8",
)
print(json.dumps(combined, indent=2))"""
        ),
        new_markdown_cell(
            """## Next steps

1. Inspect `validation-artifacts/*-transcript.log` and agent-debug JSONL.
2. Export replayable candidates that survive `OptimalGuardrail`.
3. Increase the budget only after this targeted batch produces useful traces.
4. Generate a self-contained static submission from validated chains.

Do not interpret an unexecuted notebook or a zero smoke score as a final
competition result."""
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
