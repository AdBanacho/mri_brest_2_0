"""Controller <-> CLI parity.

Every argv the Bash controller builds (all tasks of all stages of every shipped
configuration) must be accepted by ``schema.build_parser(stage)`` and
``schema.resolve``; each stage must emit exactly the fields.tsv keys of its
sections; values must round-trip; and the printed (dry-run) argv must equal
the argv a real task passes to the interpreter.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from mribreast.config import schema

from tests.slurm.bashenv import BASH, ROOT

pytestmark = [pytest.mark.bash, pytest.mark.skipif(BASH is None, reason="bash (>= 4.3) not available")]

CONTROLLER = ROOT / "run_experiment.slurm"
CONFIGS = [ROOT / "experiment.conf", *sorted((ROOT / "configs").glob("*.conf"))]
STAGES = ("prepare", "train", "validate", "summarize")


def run_controller(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, CONTROLLER.as_posix(), *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=600, env=env)


def local_dry_run(stage: str, config: Path) -> list[list[str]]:
    proc = run_controller(f"--{stage}", "--config", config.as_posix(), "--local", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    argvs = []
    for line in proc.stdout.splitlines():
        if line.startswith("Local task "):
            argvs.append(shlex.split(line.split(": ", 1)[1]))
    assert argvs, proc.stdout
    return argvs


def split_argv(argv: list[str]) -> tuple[str, dict[str, list[str]], list[str]]:
    assert argv[1:3] == ["-m", "mribreast"], argv[:4]
    stage, tail = argv[3], argv[4:]
    emitted: dict[str, list[str]] = {}
    key = None
    for word in tail:
        if word.startswith("--") and re.fullmatch(r"--[a-z0-9_]+", word):
            key = word[2:]
            assert key not in emitted, f"duplicate --{key}"
            emitted[key] = []
        else:
            assert key is not None
            emitted[key].append(word)
    return stage, emitted, tail


def test_stage_sections_match_schema() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    match = re.search(r"declare -A STAGE_SECTIONS=\((.*?)\)\n", text)
    assert match
    pairs = dict(re.findall(r"\[(\w+)\]='([^']*)'", match.group(1)))
    assert {k: tuple(v.split()) for k, v in pairs.items()} == schema.STAGE_SECTIONS


@pytest.mark.parametrize("config", CONFIGS, ids=[c.stem for c in CONFIGS])
@pytest.mark.parametrize("stage", STAGES)
def test_every_task_argv_parses_and_round_trips(stage: str, config: Path) -> None:
    expected_keys = {f.key for f in schema.fields_for_stage(stage)}
    for argv in local_dry_run(stage, config):
        argv_stage, emitted, tail = split_argv(argv)
        assert argv_stage == stage
        assert set(emitted) == expected_keys
        ns = schema.build_parser(stage).parse_args(tail)
        resolved = schema.resolve(stage, vars(ns))
        for key, words in emitted.items():
            field = schema.FIELDS[key]
            if field.type == "ints":
                assert len(words) >= 1
                raw = " ".join(words)
            else:
                assert len(words) == 1, f"--{key} must be exactly one argument, got {words}"
                raw = words[0]
            assert schema.parse_value(field, raw) == resolved[key], key
            assert schema.parse_value(field, schema.format_value(field, resolved[key])) == resolved[key], key


def test_legacy_profile_values() -> None:
    argvs = local_dry_run("validate", ROOT / "experiment.conf")
    assert len(argvs) == 24
    _, emitted, tail = split_argv(argvs[-1])
    cfg = schema.resolve("validate", vars(schema.build_parser("validate").parse_args(tail)))
    assert cfg["profile"] == "duke_legacy_3d"
    assert cfg["imaging_patient_id_column"] == "Patient ID"
    assert cfg["target_size"] == (256, 256, 64)
    assert cfg["feature_groups"] == ("clinical", "kinetic", "morphology", "heterogeneity")
    assert cfg["mri_model"] == "resnet18" and cfg["feature_model"] == "mlp" and cfg["sensitivity_lambda"] == 0.2
    assert cfg["use_annotation_boxes"] is True and cfg["include_sensitive"] is False
    assert cfg["validate_num_workers"] == 4 and cfg["fusion_alpha"] == 0.5
    assert cfg["lr"] == 1e-4 and cfg["epoch"] == 30 and cfg["seed"] == 42
    assert emitted["density_file"] == [""]


def test_printed_argv_equals_executed_argv(tmp_path: Path) -> None:
    fake = (ROOT / "tests" / "slurm" / "fakes" / "python").as_posix()
    log = tmp_path / "pylog"
    config = tmp_path / "experiment.conf"
    config.write_text(
        (ROOT / "experiment.conf").read_text(encoding="utf-8")
        + "\nMODULES=()\nAUTO_INSTALL_PYTHON_PACKAGES=false\nPYTHON_VENV_ROOT=''\n"
        + f"PYTHON_BIN='{fake}'\nPATHS[slurm_logs]='{(tmp_path / 'logs').as_posix()}'\n"
        + "VALIDATE_GRID[fusion_alpha]='0.25|0.5|0.75'\n",
        encoding="utf-8", newline="\n",
    )
    printed = local_dry_run("validate", config)[7]
    env = dict(os.environ, FAKE_PY_LOG=log.as_posix())
    proc = run_controller("--validate", "--config", config.as_posix(), "--local", "--task-index", "7", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    recorded = sorted(log.glob("*.argv"))
    assert len(recorded) == 1
    executed = recorded[0].read_bytes().split(b"\0")[:-1]
    assert [w.decode() for w in executed] == printed[1:]
    _, emitted, _ = split_argv(printed)
    assert emitted["fusion_alpha"] == ["0.5"] and emitted["mri_model"] == ["resnet10"]
