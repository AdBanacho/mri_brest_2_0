"""Fast CPU smoke runs of the remaining modes through the real CLI (train -> validate -> summarize).

The two full acceptance runs (3D+tabular+fusion, 2D+masks+tabular+fusion) go through the Bash
controller with configs/synthetic_cpu_{3d,2d}.conf; see docs/acceptance.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from mribreast.cli import main
from mribreast.synthetic.dataset import make_synthetic_dataset

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    info = make_synthetic_dataset(root / "data", n_patients=60, seed=3, shape=(16, 16, 8), size_2d=(16, 16),
                                  images_per_study=3)
    return root, info


def _paths(root: Path, info, tag: str) -> list[str]:
    args = []
    for key in ("clinical_features_file", "metadata_file", "imaging_features_file", "annotation_boxes_file",
                "image_root", "image_manifest_file", "mask_manifest_file"):
        args += [f"--{key}", str(info[key])]
    for key in ("prepared_root", "subtraction_root", "logs_root", "checkpoint_root", "validation_root", "summary_root"):
        args += [f"--{key}", str(root / tag / key)]
    return args


COMMON = ["--num_folds", "3", "--threshold_calibration_folds", "2", "--epoch", "1", "--batch_size", "4",
          "--num_workers", "0", "--target_size", "16", "16", "8", "--target_size_2d", "16", "16",
          "--xgb_n_estimators", "10", "--xgb_n_jobs", "1", "--lasso_n_jobs", "1"]

MODES = {
    # historical image-only, 3 classes, outer fold selects the checkpoint (legacy trainingWorkflow)
    "image_only_3class_historical": ["--branches", "image", "--label_scheme", "oncotype_3class_18_31",
                                     "--selection_split", "outer", "--mri_model", "resnet10"],
    "tabular_only_baseline": ["--branches", "tabular", "--feature_groups", "clinical,kinetic", "--feature_model",
                              "mlp", "--feature_selector", "lasso", "--mlp_max_iter", "50", "--lasso_cv_folds", "2"],
    "subtraction_fcn_separate": ["--branches", "image,tabular", "--subtraction_mode", "post_minus_pre",
                                 "--mri_model", "fcn", "--selection_split", "separate", "--profile", "general",
                                 "--phase_order", "explicit"],
    "ready_2d_multi_image_max": ["--branches", "image", "--profile", "general", "--image_dim", "2d",
                                 "--image_source", "manifest_2d", "--mri_model", "resnet18", "--aggregation",
                                 "max_prob", "--mask_provider", "local_files", "--mask_kind", "predicted",
                                 "--mask_mode", "foreground_only", "--mask_missing_policy", "drop",
                                 "--mask_empty_policy", "full_image"],
}
CONFIGS = Path(__file__).resolve().parents[2] / "configs"
BOX_MODES = ("roi_crop", "box_channel", "dual_stream")


def config_train_args(path: Path) -> list[str]:
    """``--key value`` pairs from the ``declare -A TRAIN=( ... )`` block of a controller config."""
    block = re.search(r"declare -A TRAIN=\((.*?)\n\)", path.read_text(encoding="utf-8"), re.S)
    assert block, path
    args: list[str] = []
    for key, value in re.findall(r"\[(\w+)\]=('[^']*'|\S+)", block.group(1)):
        args += [f"--{key}", *value.strip("'").split()]
    return args


# annotation boxes as a model input (training AND inference): one synthetic config per box_mode
for _mode in BOX_MODES:
    MODES[f"box_{_mode}"] = config_train_args(CONFIGS / f"synthetic_cpu_3d_box_{_mode}.conf")

VALIDATE_EXTRA = {"image_only_3class_historical": ["--threshold_strategy", "fixed_0.5"]}


@pytest.mark.parametrize("mode", list(MODES))
def test_mode_end_to_end(data, mode):
    root, info = data
    # box configs carry their own sizes/epochs; the shared small-run settings are applied after them
    argv = (_paths(root, info, mode) + MODES[mode] + COMMON if mode.startswith("box_")
            else _paths(root, info, mode) + COMMON + MODES[mode])
    assert main(["train", *argv]) == 0
    assert main(["validate", *argv, *VALIDATE_EXTRA.get(mode, []), "--validate_num_workers", "0"]) == 0
    assert main(["summarize", "--validation_root", str(root / mode / "validation_root"),
                 "--summary_root", str(root / mode / "summary_root"), "--expected_folds", "3"]) == 0
    evals = list((root / mode / "validation_root").glob("tr-*/ev-*/COMPLETE"))
    assert len(evals) == 1
    out = evals[0].parent
    manifest = json.loads((out / "evaluation_manifest.json").read_text())
    metrics = pd.read_csv(out / "validation_metrics.csv")
    assert len(metrics) == 3
    branch = manifest["primary_branch"]
    assert metrics[f"{branch}_auc_roc"].notna().any()
    if mode == "image_only_3class_historical":
        assert manifest["num_classes"] == 3 and manifest["decision_rule"] == "argmax"
        assert manifest["selection_uses_outer_eval"] is True
    if mode == "ready_2d_multi_image_max":
        excluded = pd.read_csv(out / "excluded_studies.csv")
        assert manifest["excluded_studies"] == len(excluded)
    if mode.startswith("box_"):
        metas = sorted((root / mode / "checkpoint_root").rglob("image_branch.json"))
        assert len(metas) == 3
        for path in metas:
            meta = json.loads(path.read_text())
            assert meta["box"]["mode"] == mode[len("box_"):] and meta["box"]["requires_box_at_inference"] is True
            audit = pd.read_csv(path.parent / "box_audit.csv")
            assert set(audit["split"]) == {"fit", "selection"} and set(audit["action"]) <= {"use", "full_image"}
        if mode == "box_dual_stream":
            assert meta["network"]["wrapper"] == "dual_stream"
            assert meta["geometry"]["input_shape"]["local"] == [5, 16, 16, 8]
        if mode == "box_box_channel":
            assert meta["channels"][-1] == "annotation_box"
    ranked = pd.read_csv(root / mode / "summary_root" / "ranked_configurations.csv")
    assert len(ranked) == 1
