"""One-call synthetic Duke-like dataset (tables + NIfTI + 2D images + masks) for CPU acceptance runs.

Layout (relied upon by configs/synthetic_cpu_*.conf)::

    <out>/Clinical_and_Other_Features.xlsx
    <out>/nbia_metadata.xlsx
    <out>/Imaging_Features.xlsx
    <out>/Annotation_Boxes.xlsx
    <out>/Breast_Radiologist_Density_Assessments.xlsx
    <out>/nifti/<SeriesInstanceUID>.nii.gz
    <out>/images2d/manifest.csv (+ PNG/TIFF files)
    <out>/masks/manifest.csv     (+ mask files; a few deliberately missing/empty)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

EXPECTED_NAMES = {
    "clinical_features_file": "Clinical_and_Other_Features.xlsx",
    "metadata_file": "nbia_metadata.xlsx",
    "imaging_features_file": "Imaging_Features.xlsx",
    "annotation_boxes_file": "Annotation_Boxes.xlsx",
    "density_file": "Breast_Radiologist_Density_Assessments.xlsx",
}


def make_synthetic_dataset(out: Path, *, n_patients: int = 48, seed: int = 0, shape=(24, 24, 12),
                           size_2d=(32, 32), images_per_study: int = 3) -> dict[str, Any]:
    from mribreast.synthetic.duke import write_duke_tables
    from mribreast.synthetic import imaging as synth_imaging

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = write_duke_tables(out / "_tables", n_patients=n_patients, seed=seed, shape=tuple(shape))
    info: dict[str, Any] = {}
    for key, name in EXPECTED_NAMES.items():
        source = tables.get(key)
        if source is None:
            continue
        target = out / name
        shutil.copyfile(source, target)
        info[key] = target
    studies = tables["studies"]
    boxes = tables.get("boxes")
    if boxes is None and "annotation_boxes_file" in info:
        import pandas as pd

        boxes = pd.read_excel(info["annotation_boxes_file"])
    synth_imaging.write_nifti_series(out / "nifti", studies, shape=tuple(shape), boxes=boxes, seed=seed)
    info["image_root"] = out / "nifti"
    info["image_manifest_file"] = synth_imaging.write_2d_dataset(out / "images2d", studies, size=tuple(size_2d),
                                                                 n_images_per_study=images_per_study, seed=seed)
    info["mask_manifest_file"] = synth_imaging.write_masks(out / "masks", info["image_manifest_file"], seed=seed)
    info["studies"] = studies
    shutil.rmtree(out / "_tables", ignore_errors=True)
    return info
