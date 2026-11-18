# Imaging, masks and the image branch

Packages: `mribreast.imaging` (geometry, volumes, cache, boxes, 2D, datasets),
`mribreast.imaging.masks` (providers, policies, audit, preview), `mribreast.models`
(networks, Lightning classifier, `ImageBranch`), `mribreast.synthetic.imaging`.
Legacy references point to `C:\code\mri_brest` (`mriBreastDuke/...`).

## 1. Geometry conventions

| Item | Convention |
| --- | --- |
| Series file | `<image_root>/<SeriesInstanceUID>.nii.gz`, else `.nii`. UIDs may contain only `[A-Za-z0-9._-]`; anything else is an invalid UID (`ValueError`), a missing file is `SeriesNotFoundError`. |
| Phase order | Order of `series_uids` in the cohort (`|`-joined or tuple); phase 0 = pre-contrast. |
| Axis order | nibabel voxel axes `(i, j, k)` with **no reorientation** (legacy MONAI `LoadImaged` + nibabel reader, `as_closest_canonical=False`). For Duke volumes written by SimpleITK this is `(column, row, slice)`. |
| Affine, units | 4x4 voxel→world (nibabel RAS+, millimetres). Spacing = column norms of the affine; orientation = `nib.aff2axcodes` (stored in `Geometry`). |
| Resize | `torch.nn.functional.interpolate`, trilinear/bilinear, `align_corners=False` (identical to MONAI `ResizeD`, verified numerically). Masks: `nearest-exact`. |
| Affine after resize | `A_out = A_in @ [diag(in/out), 0.5*(in/out - 1)]` (= MONAI MetaTensor affine). **Legacy wrote the stale pre-resize affine** into cached files (`img_meta_dict`); we store the correct one. |
| Padding | Missing phases → zero channels appended at the end; spatial zero padding at the high end of each axis (legacy `pad_collate`). With a fixed `target_size` no spatial padding occurs. |
| 2D slices | Slice along `slice_axis` keeps the remaining axes in increasing order → `[C, H, W]`. `slice_id` = index along `slice_axis` in the **original** grid of the study's first series (the mask grid). |

## 2. Order of operations (verified against legacy `dataLoaders/NiftiDataset.py`)

* `subtraction_mode=none` (`_monai_preprocess_and_save`, L133-155): load → z-score of
  the whole volume (MONAI `NormalizeIntensityd(nonzero=False, channel_wise=True)`:
  population std, **no epsilon**, std==0 → divide by 1) → trilinear resize.
  Normalisation happens **before** resize (test proves the order is observable and
  matches the MONAI pipeline to 1e-5).
* `post_minus_pre` / `consecutive` (`_load_or_create_subtraction`, L241-296):
  verify identical original shape and affine (`allclose rtol=1e-4, atol=1e-3`) →
  resize the raw phases → `minuend - subtrahend` → z-score with population std
  (`unbiased=False`) and `+1e-5`. Pairs (`subtraction.py`): `post_minus_pre` =
  `(p_k, p_0)`, `consecutive` = `(p_k, p_{k-1})`.
* Channels: `phase_count` for `none`, `phase_count - 1` for subtraction (legacy
  `get_input_channels`, 5/4); `+1` for `mask_channel`. Ready 2D images: channels of
  the image files.
* The custom (non-MONAI) legacy path used `std + 1e-5` for phases; it was not the
  default (`use_monai=True` in `NiftiDataModule`) and is not reproduced.

## 3. Deterministic cache (`prepared_root`, `subtraction_root`)

Key = sha256 of `{cache_schema_version, kind (phase|difference), sources [uid, size,
mtime_ns, sha256(first 64 KiB)], target_size, normalisation, resize_backend,
align_corners, dtype}`. Legacy keyed only on `abspath(image_root)` + size, so
preparation variants could share entries. Layout:
`<root>/<kind>/<key[:2]>/<key>.nii.gz` + `<key>.json` (key payload) + `<key>.lock`.
The NIfTI `descrip` holds `mbc1:<sha256 of voxels>`.

Readers take a shared lock (`fcntl.flock` / Windows `LockFileEx`; fallback
`filelock.FileLock`, exclusive). A miss or corrupt entry (load error or checksum
mismatch) takes the exclusive lock, re-checks, builds from the read-only sources,
writes a temp file in the entry directory and publishes with `os.replace`.
Non-corruption errors (e.g. `PermissionError`) are raised, never rebuilt.
Augmentations are never cached. 2D slicing uses the same cache at original size.

## 4. Annotation boxes (`use_annotation_boxes`)

Verified from `dataLoaders/annotation_boxes.py` (docstring L1-7, `load_annotation_boxes`
L23-47) and `tests/test_annotation_boxes.py` (`Breast_MRI_207` → `((358,374,34),(365,387,40))`):
workbook indices are **zero-based and inclusive**; `(Start/End Column, Row, Slice)`
map to array axes `(0, 1, 2)`; stored half-open as `((c0,r0,s0),(c1+1,r1+1,s1+1))`.

* Audit (`annotation_coverage`, statuses as legacy): `eligible`, `missing_box`,
  `ambiguous_patient_studies` (patient ≠ 1 study), `missing_series`,
  `incompatible_series_geometry` (non-3D, shape/affine disagreement between all
  series), `out_of_volume_box` (any series), `unmatched_annotation`; new
  `missing_series_file` (legacy crashed). Duplicate workbook rows / duplicate
  `study_id` / malformed coordinates raise. `write_annotation_coverage` writes
  `annotation_coverage.csv`.
* Resize: start `floor(a*dst/src)`, exclusive end `ceil(b*dst/src)`.
* Erasing (NiftiDataset L324-342): after resize one background cuboid with sides
  `max(1, dim//4)`, shared across channels, up to `annotation_erase_attempts` (8)
  uniform draws; draws intersecting the lesion box on all axes are rejected; if
  none fits the sample is unchanged. RNG = `default_rng([seed, epoch, item])`
  (legacy used the global torch RNG). Only the training dataset of `fit` rows;
  selection/calibration/validation/inference see full images.
* If no fit study is eligible, `fit` raises (legacy behaviour).
* **Open assumption (requires checking on real data):** the workbook names no
  series UID, origin or orientation. The column/row/slice → axis mapping and the
  slice numbering have not been checked visually against real DICOM/NIfTI
  overlays (no reference series in this environment). Until verified, treat
  results with boxes as provisional; shape/affine checks cannot prove it.

## 5. 2D contract

* **Ready images** (`image_source=manifest_2d`): CSV `patient_id, study_id,
  series_id, slice_id, path [, channel|phase]` (+ any extra columns); relative paths
  resolve against the manifest directory; PNG (PIL, always HWC→CHW) and TIFF
  (`tifffile`); rows sharing a key are stacked by `channel`. Reader:
  `read_image_manifest(path)`. Each image channel is z-scored (population std),
  then masks are applied, then the joint resize to `target_size_2d`.
* **Slices of NIfTI** (`image_source=nifti, image_dim=2d`): `slice_axis`,
  `slice_selection` = `center` (contiguous block around `n//2`), `uniform`
  (`int((i+0.5)n/k)`), `mask_max_area` (top-k ROI area, ties → lower index),
  `mask_all` (every slice with ROI); `slices_per_study`. Channels = phases or
  subtraction images in phase order.
* One measurement path for 1..n images per study: every image is an item with a
  `study_index`; the loss unit is the image (class weights from image labels);
  checkpoint metrics and predictions are aggregated per study:
  `mean_prob` (default) or `max_prob` (binary: `p_pos = max_i p_i[pos]`;
  multiclass: per-class max renormalised). Tabular fusion joins at study level.
* Studies whose images are all dropped by a mask policy get NaN probability rows
  (`ImagePredictor.last_audit` lists them); the protocol must exclude them and
  report the denominators.

## 6. Masks

Key: `ImageKey(patient_id, study_id, series_id, slice_id)`; `slice_id=""` = whole
volume. Providers (`contracts.MaskProvider`):

* `LocalFileMaskProvider(manifest, slice_axis=2, source=None)` – CSV with key
  columns + `path` and optional `class_codes` (JSON), `source` (`manual|predicted`),
  `model_id`, `model_version`, `grid_source`, `training_provenance`
  (`unknown|frozen_external|fold_trained:<k>`), `known_overlap`, `fold`. PNG/TIFF
  2D masks; NIfTI masks as whole volumes or slices extracted on demand. `get(key, fold)`
  prefers rows of that fold, else fold-agnostic rows, never another fold's masks.
  Records carry geometry (shape, affine, spacing, orientation), class codes,
  model id/version, sha256 fingerprint and provenance.
* `SyntheticMaskProvider(seed)` – deterministic ellipsoids on the image grid;
  `ImageBranch` binds it to `image_root` / `image_manifest_file`.
* Manual masks, predicted masks and 3D annotation boxes are separate sources:
  `mask_kind` must match the record `source` (`synthetic` accepted), boxes never
  enter the mask path.

Modes: `none`, `roi_crop` (bbox of `mask_roi_classes` ± `roi_margin` × extent per
side, joint crop then joint resize), `mask_channel` (binary ROI appended), and
`foreground_only` (image zeroed outside ROI after the joint resize). Joint
transforms (`joint_crop/flip/resize`) share parameters; image continuous, mask
nearest; every op is logged. `write_mask_preview` writes a PNG (image | overlay)
and a JSON of the applied transforms.

Policies: `mask_missing_policy`, `mask_empty_policy` ∈ `error|full_image|drop`.
`full_image` = whole image is ROI (no crop/zeroing; `mask_channel` gets an all-ones
channel, so the channel count is kept). Invalid masks (key or geometry mismatch,
undeclared class codes, wrong source kind) → error (no config field yet). Mask
geometry must equal the image grid (shape, and the affine for NIfTI). Audit:
`mask_audit` (one row per decision: key, purpose, status, action, provenance,
fingerprint) and `mask_audit_summary` (studies/images in/out = cohort change and
metric denominators). Mask-based slice selection needs masks at inference too; a
missing mask follows the missing policy (`full_image` → centre selection).

### Provenance protocol

* **Frozen external segmenter:** record model id/version, its training data and
  any known overlap with this cohort (`known_overlap`); masks are fold-agnostic.
* **Segmenter trained on cohort data:** masks for outer fold *k* must come from a
  model fitted only on the patients allowed for fold *k* (`fold=k`,
  `training_provenance=fold_trained:k`, `mask_provenance=fold_trained`).
* **Unknown provenance:** allowed, but `describe()["unknown_provenance"]` and
  `limitations` flag it. Report it as a limitation of the result.
* Segmentation quality (Dice etc.) is evaluated separately from the classifier,
  with its own data/artifacts/metrics. Without reference masks, report mask
  coverage (audit) and the geometry checks only.

## 7. Models and training

`make_network(name, spatial_dims, in_channels, num_classes, extra=mri_extra_json)`:
`fcn` (port of `Simple3DFCN`, 2D/3D; legacy hard-coded 1 input channel and failed
when the workflow passed `in_channels`), MONAI `DenseNet121`, `resnet10`, `resnet18`.
`ImageClassifier` reproduces `NiftiClassifier`: Adam (or AdamW), weighted CE with
balanced class weights `total/(K*max(count,1))` and `positive_boost` on the
positive class (`n_fold_cv_run._compute_balanced_class_weights`, L889-903), plus
`sensitivity_lambda*(1 - soft sensitivity)` (`loss=ce` drops it). Logged:
`val_auc_roc` (macro AUROC), `val_sensitivity`, `val_balanced_accuracy`,
`val_specificity`. `ModelCheckpoint(monitor=val_auc_roc, max, save_top_k=1,
save_last=True)`; early stopping off by default (legacy declared it but left it
commented out, `n_fold_cv_run.py` L560-578); scheduler `none|cosine`; CSV logger.
`trainer_extra_json` cannot override `callbacks, logger, default_root_dir,
enable_checkpointing`. The training loader is not shuffled (legacy).

`ImageBranch.fit` writes `best.ckpt, last.ckpt, image_branch.json, history.csv`
(+ `annotation_coverage.csv`, `mask_audit_fit.csv`). `load` rejects artifacts
trained with different geometry/channel/mask settings or class count.

## 8. Synthetic data

`write_nifti_series(image_root, studies, shape, boxes=None, seed=0)`,
`make_annotation_boxes(studies, shape, seed)` (Duke format, label-independent),
`write_2d_dataset(root, studies, size, n_images_per_study, seed, vary_count=False)`
(even rows PNG, odd rows TIFF; `series_id = <study_id>.2d`),
`write_masks(root, manifest_or_studies, seed=0, n_missing=1, n_empty=1,
image_root=None, boxes=None, folds=None, nifti_studies=None)`. To make one mask
manifest that covers both ready 2D images and NIfTI slices, call
`write_masks(out/"masks", image_manifest, seed=seed, nifti_studies=studies,
image_root=out/"nifti", boxes=boxes)`.
