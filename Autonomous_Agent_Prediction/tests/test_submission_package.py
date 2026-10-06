from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIRS = sorted(
    path
    for path in (PROJECT_ROOT / "submissions").glob("*/agent")
    if (path / "agent.yaml").is_file()
)
ADAPTIVE_AGENT_DIR = PROJECT_ROOT / "submissions" / "05_adaptive_scientist" / "agent"
MODELS_PATH = PROJECT_ROOT / "data" / "raw" / "models.yaml"
ALLOWED_TOOLS = {
    "run_command",
    "read_file",
    "write_file",
    "edit_file",
    "submit_predictions",
    "select_submission",
    "get_status",
    "agent_tool",
}


def load_yaml_with_includes(path: Path, agent_dir: Path) -> dict:
    class AgentLoader(yaml.SafeLoader):
        pass

    def include_constructor(loader: AgentLoader, node: yaml.Node) -> str:
        relative_path = Path(loader.construct_scalar(node))
        if relative_path.is_absolute():
            raise ValueError(f"Unsafe include path: {relative_path}")
        include_path = (path.parent / relative_path).resolve()
        include_path.relative_to(agent_dir.resolve())
        return include_path.read_text(encoding="utf-8")

    AgentLoader.add_constructor("!include", include_constructor)
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=AgentLoader)


def load_agent_config(agent_dir: Path) -> dict:
    return load_yaml_with_includes(agent_dir / "agent.yaml", agent_dir)


@pytest.mark.parametrize("agent_dir", AGENT_DIRS)
def test_agent_root_contract(agent_dir: Path) -> None:
    assert (agent_dir / "agent.yaml").is_file()
    assert not any(path.is_symlink() for path in agent_dir.rglob("*"))

    config = load_agent_config(agent_dir)
    assert {"name", "model", "instruction", "tools"} <= config.keys()
    tool_names = {
        tool if isinstance(tool, str) else next(iter(tool))
        for tool in config["tools"]
    }
    assert tool_names <= ALLOWED_TOOLS
    assert "submit_predictions" in tool_names
    assert "select_submission" in tool_names
    assert "get_status" in tool_names

    models = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))["models"]
    assert config["model"] in models

    for tool in config["tools"]:
        if not isinstance(tool, dict) or "agent_tool" not in tool:
            continue
        relative_path = Path(tool["agent_tool"]["config_path"])
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        tool_path = (agent_dir / relative_path).resolve()
        tool_path.relative_to(agent_dir.resolve())
        tool_config = load_yaml_with_includes(tool_path, agent_dir)
        assert tool_config["model"] in models
        assert tool_config["name"]
        assert tool_config["instruction"]


@pytest.mark.parametrize("agent_dir", AGENT_DIRS)
def test_bundled_skills_are_safe_and_well_formed(agent_dir: Path) -> None:
    config = load_agent_config(agent_dir)
    assert len(config.get("skills", [])) <= 10

    for skill_reference in config.get("skills", []):
        relative_path = Path(skill_reference)
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        skill_dir = (agent_dir / relative_path).resolve()
        skill_dir.relative_to(agent_dir.resolve())

        manifest_path = skill_dir / "SKILL.md"
        assert manifest_path.is_file()
        manifest = manifest_path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---", manifest, flags=re.DOTALL)
        assert match is not None
        frontmatter = yaml.safe_load(match.group(1))
        assert frontmatter["name"] == skill_dir.name
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", frontmatter["name"])
        assert frontmatter["description"]


@pytest.mark.parametrize("agent_dir", AGENT_DIRS)
def test_scripts_compile(agent_dir: Path) -> None:
    scripts = sorted(agent_dir.glob("skills/*/scripts/*.py"))
    assert scripts
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        compile(source, str(script), "exec")


@pytest.mark.parametrize("agent_dir", AGENT_DIRS)
def test_fallback_matches_sample_schema(tmp_path: Path, agent_dir: Path) -> None:
    data_dir = PROJECT_ROOT / "data" / "raw" / "data" / "train_01"
    output = tmp_path / "fallback.csv"
    script = next(agent_dir.glob("skills/*/scripts/prepare_fallback.py"))
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    submission = pd.read_csv(output)
    assert submission.columns.tolist() == sample.columns.tolist()
    assert len(submission) == len(sample)
    assert submission.iloc[:, 0].equals(sample.iloc[:, 0])
    assert submission.iloc[:, 0].is_unique
    assert submission.iloc[:, 1].between(0, 1).all()


def test_portfolio_generates_ranked_schema_valid_candidates(tmp_path: Path) -> None:
    data_dir = PROJECT_ROOT / "data" / "raw" / "data" / "train_13"
    agent_dir = AGENT_DIRS[1]
    script = (
        agent_dir
        / "skills"
        / "robust-portfolio"
        / "scripts"
        / "train_portfolio.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--models",
            "sklearn",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.startswith("CANDIDATES ")

    manifest = yaml.safe_load((tmp_path / "candidate_manifest.json").read_text())
    candidates = manifest["candidates"]
    assert 2 <= len(candidates) <= 6
    assert [item["oof_auc"] for item in candidates] == sorted(
        [item["oof_auc"] for item in candidates], reverse=True
    )

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    for candidate in candidates:
        submission = pd.read_csv(tmp_path / candidate["file"])
        assert submission.columns.tolist() == sample.columns.tolist()
        assert len(submission) == len(sample)
        assert submission.iloc[:, 0].equals(sample.iloc[:, 0])
        assert submission.iloc[:, 1].between(0, 1).all()


def test_adaptive_feature_engineer_preserves_alignment(tmp_path: Path) -> None:
    data_dir = PROJECT_ROOT / "data" / "raw" / "data" / "train_13"
    script = (
        AGENT_DIRS[2]
        / "skills"
        / "feature-engineer"
        / "scripts"
        / "generate_features.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--train",
            str(data_dir / "train.csv"),
            "--test",
            str(data_dir / "test.csv"),
            "--target",
            "target",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    original_train = pd.read_csv(data_dir / "train.csv")
    original_test = pd.read_csv(data_dir / "test.csv")
    engineered_train = pd.read_csv(tmp_path / "train_engineered.csv")
    engineered_test = pd.read_csv(tmp_path / "test_engineered.csv")

    assert len(engineered_train) == len(original_train)
    assert len(engineered_test) == len(original_test)
    assert "target" in engineered_train.columns
    assert "target" not in engineered_test.columns
    assert engineered_train["target"].equals(original_train["target"])
    assert engineered_train.drop(columns="target").columns.tolist() == (
        engineered_test.columns.tolist()
    )


def test_native_portfolio_emits_guarded_family_candidates(tmp_path: Path) -> None:
    data_dir = PROJECT_ROOT / "data" / "raw" / "data" / "train_13"
    agent_dir = AGENT_DIRS[3]
    script = (
        agent_dir
        / "skills"
        / "robust-portfolio"
        / "scripts"
        / "train_portfolio.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--models",
            "sklearn",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.startswith("CANDIDATES ")
    manifest = yaml.safe_load(
        (tmp_path / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    candidates = manifest["candidates"]
    assert 2 <= len(candidates) <= 4
    assert all(item["family"] for item in candidates)
    assert [item["oof_auc"] for item in candidates] == sorted(
        [item["oof_auc"] for item in candidates],
        reverse=True,
    )
    assert {"best_core", "blend_gate", "interaction_gate"} <= set(
        manifest["diagnostics"]
    )


def _write_runtime_fixture(root: Path, kind: str) -> Path:
    rng = np.random.default_rng(42)
    rows = 180 if kind == "tiny" else 2_600
    test_rows = 120
    train_features: dict[str, object] = {}
    test_features: dict[str, object] = {}

    if kind == "categorical":
        for index in range(5):
            train_features[f"cat_{index}"] = rng.choice(list("ABCD"), rows)
            test_features[f"cat_{index}"] = rng.choice(list("ABCD"), test_rows)
    elif kind == "ordinal":
        train_features = {
            "ord_0": rng.integers(0, 5, rows),
            "ord_1": rng.integers(0, 8, rows),
            "num_0": rng.normal(size=rows),
            "num_1": rng.normal(size=rows),
            "num_2": rng.normal(size=rows),
        }
        test_features = {
            "ord_0": rng.integers(0, 5, test_rows),
            "ord_1": rng.integers(0, 8, test_rows),
            "num_0": rng.normal(size=test_rows),
            "num_1": rng.normal(size=test_rows),
            "num_2": rng.normal(size=test_rows),
        }
    else:
        for index in range(5):
            shift = 4.0 if kind == "shifted" and index == 0 else 0.0
            train_features[f"num_{index}"] = rng.normal(size=rows)
            test_features[f"num_{index}"] = rng.normal(loc=shift, size=test_rows)
        if kind in {"tiny", "shifted"}:
            train_features["cat"] = rng.choice(list("ABC"), rows)
            test_features["cat"] = rng.choice(list("ABC"), test_rows)

    train = pd.DataFrame(train_features)
    test = pd.DataFrame(test_features)
    if kind == "missing":
        train.loc[rng.random(rows) < 0.65, "num_0"] = np.nan
        test.loc[rng.random(test_rows) < 0.65, "num_0"] = np.nan
    numeric_signal = pd.to_numeric(train.iloc[:, 0], errors="coerce").fillna(0)
    if numeric_signal.nunique() <= 8:
        numeric_signal = pd.Series(pd.factorize(train.iloc[:, 0])[0], index=train.index)
    target = (numeric_signal.to_numpy() + rng.normal(scale=0.8, size=rows) > 0).astype(int)
    if np.unique(target).size != 2:
        target[::2] = 0
        target[1::2] = 1

    train.insert(0, "row_id", np.arange(rows))
    train["target"] = target
    test.insert(0, "row_id", np.arange(100_000, 100_000 + test_rows))
    sample = pd.DataFrame({"row_id": test["row_id"], "target": 0.5})
    root.mkdir(parents=True, exist_ok=True)
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    sample.to_csv(root / "sample_submission.csv", index=False)
    return root


@pytest.mark.parametrize(
    ("kind", "expected_recipe"),
    [
        ("tiny", "tiny-safe"),
        ("numeric", "numeric-boost"),
        ("categorical", "categorical-native"),
        ("ordinal", "ordinal-native"),
        ("missing", "missing-robust"),
        ("shifted", "numeric-boost"),
    ],
)
def test_adaptive_profiler_routes_deterministically(
    tmp_path: Path, kind: str, expected_recipe: str
) -> None:
    data_dir = _write_runtime_fixture(tmp_path / kind, kind)
    script = (
        ADAPTIVE_AGENT_DIR
        / "skills"
        / "adaptive-scientist"
        / "scripts"
        / "profile_dataset.py"
    )
    first = tmp_path / f"{kind}_profile_1.json"
    second = tmp_path / f"{kind}_profile_2.json"
    for output in (first, second):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--data-dir",
                str(data_dir),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    assert first.read_bytes() == second.read_bytes()
    profile = json.loads(first.read_text(encoding="utf-8"))
    assert profile["default_recipe"] == expected_recipe
    assert profile["default_recipe"] in profile["allowed_recipes"]
    assert len(first.read_bytes()) < 64_000
    assert len(profile["schema"]["categorical_columns"]) <= 20
    assert len(profile["schema"]["ordinal_like_columns"]) <= 20
    if kind == "shifted":
        assert profile["flags"]["shifted"] is True


def test_adaptive_workflow_commands_are_declared_and_reachable() -> None:
    config = load_agent_config(ADAPTIVE_AGENT_DIR)
    tool_names = {
        tool if isinstance(tool, str) else next(iter(tool))
        for tool in config["tools"]
    }
    assert {"run_command", "write_file", "submit_predictions", "select_submission", "get_status", "agent_tool"} <= tool_names
    prompt = config["instruction"]
    for command in (
        "prepare_fallback.py",
        "profile_dataset.py",
        "train_adaptive.py",
        "record_feedback.py",
        "select_plan.py --mode challenger",
        "select_plan.py --mode final",
        "select_submission",
    ):
        assert command in prompt
    assert "Call `planner` once" in prompt
    assert "second and\n    final time" in prompt
    assert "write_file" in prompt
    assert "solution.csv" in prompt and "Do not" in prompt


def test_adaptive_invalid_plan_and_missing_optional_models_degrade_safely(
    tmp_path: Path,
) -> None:
    data_dir = _write_runtime_fixture(tmp_path / "runtime", "tiny")
    scripts = ADAPTIVE_AGENT_DIR / "skills" / "adaptive-scientist" / "scripts"
    profile = tmp_path / "dataset_profile.json"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "profile_dataset.py"),
            "--data-dir",
            str(data_dir),
            "--output",
            str(profile),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    invalid_plan = tmp_path / "invalid_plan.json"
    invalid_plan.write_text(
        json.dumps({"recipe_id": "write-python", "code": "not allowed"}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            str(scripts / "train_adaptive.py"),
            "--models",
            "sklearn",
            "--data-dir",
            str(data_dir),
            "--profile",
            str(profile),
            "--plan",
            str(invalid_plan),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.startswith("CANDIDATES ")
    manifest = json.loads((output_dir / "candidate_manifest.json").read_text())
    assert manifest["planner"]["valid"] is False
    assert 2 <= len(manifest["initial_candidates"]) <= 3
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    for candidate in manifest["candidates"]:
        submission = pd.read_csv(output_dir / candidate["file"])
        assert submission.columns.tolist() == sample.columns.tolist()
        assert submission.iloc[:, 0].equals(sample.iloc[:, 0])
        assert np.isfinite(submission.iloc[:, 1]).all()
        assert submission.iloc[:, 1].between(0, 1).all()


def test_feedback_selector_handles_nonfinite_scores_and_hedges_diversity(
    tmp_path: Path,
) -> None:
    rows = 100
    ids = np.arange(rows)
    base = np.linspace(0.01, 0.99, rows)
    inverse = base[::-1]
    for filename, values in (
        ("candidate_tree.csv", base),
        ("candidate_tree_clone.csv", base + np.sin(ids) * 1e-8),
        ("candidate_linear.csv", inverse),
    ):
        pd.DataFrame({"row_id": ids, "target": values}).to_csv(
            tmp_path / filename,
            index=False,
        )
    manifest = {
        "candidates": [
            {"name": "tree", "file": "candidate_tree.csv", "family": "tree", "role": "anchor", "oof_auc": 0.80, "fold_auc_std": 0.02},
            {"name": "tree_clone", "file": "candidate_tree_clone.csv", "family": "boost", "role": "anchor", "oof_auc": 0.81, "fold_auc_std": 0.02},
            {"name": "linear", "file": "candidate_linear.csv", "family": "linear", "role": "anchor", "oof_auc": 0.78, "fold_auc_std": 0.01},
        ]
    }
    ledger = {
        "submissions": [
            {"candidate": "fallback", "file": "fallback_submission.csv", "submission_id": "sub_0", "public_score": "nan", "submission_order": 0},
            {"candidate": "tree", "submission_id": "sub_1", "public_score": 0.82, "submission_order": 1},
            {"candidate": "tree_clone", "submission_id": "sub_2", "public_score": 0.83, "submission_order": 2},
            {"candidate": "linear", "submission_id": "sub_3", "public_score": 0.80, "submission_order": 3},
        ]
    }
    (tmp_path / "candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "feedback_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    selector = (
        ADAPTIVE_AGENT_DIR
        / "skills"
        / "adaptive-scientist"
        / "scripts"
        / "select_plan.py"
    )
    output = tmp_path / "selection_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(selector),
            "--mode",
            "final",
            "--manifest",
            str(tmp_path / "candidate_manifest.json"),
            "--ledger",
            str(tmp_path / "feedback_ledger.json"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    selection = json.loads(output.read_text(encoding="utf-8"))
    assert 1 <= len(selection["submission_ids"]) <= 2
    assert "sub_2" in selection["submission_ids"]
    assert not {"sub_1", "sub_2"} <= set(selection["submission_ids"])
    assert any("non_finite_score" in error for error in selection["errors"])
