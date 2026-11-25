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

## 4b. Annotation boxes as a model input (`box_mode`)

New (not in legacy). In these modes the Duke box is part of the model input, so it is needed at
**training and at inference**: for fit, selection, calibration, validation and prediction frames. That is
a research setting. Deployment would need a radiologist box or a lesion detector for every test study.
`image_branch.json` records it as `box.requires_box_at_inference: true`.

| `box_mode` | Input | Network |
|---|---|---|
| `none` | unchanged | `mri_model` |
| `roi_crop` | box ± `box_margin` cropped on the **native** grid, resized to `target_size`; the only image | `mri_model` |
| `box_channel` | full volume at `target_size` + one binary channel = box mapped with `resize_box` (outward rounding, no lesion voxel lost) | `mri_model`, `in_channels + 1` |
| `dual_stream` | global = full volume at `target_size`; local = box ± `box_margin` cropped on the native grid, resized to `box_local_size` | `DualStreamNetwork`: two independent `mri_model` encoders (output `dual_stream_embed_dim`), concat → ReLU → Dropout(0.2) → Linear |

Fields: `box_mode`, `box_margin` (fraction of box extent per side, `ceil`, clipped to the
volume, [0, 2], default 0.1; same rule as mask `roi_margin`), `box_local_size` (default `64 64 32`),
`dual_stream_embed_dim` (default 128), `box_missing_policy`.

* Code: `imaging/boxes/` (`policies.py` decisions + audit, `common.py` `BoxInput`/`expand_box`,
  `roi_crop.py`, `channel.py`, `dual.py`), `models/dual_stream.py`.
* **Missing box** = any `annotation_coverage` status other than `eligible`. `box_missing_policy`:
  `error` (default) fails; `full_image` keeps the study (`roi_crop` → full volume, `box_channel` →
  all-zero channel, `dual_stream` → local = full native volume resized); `drop` gives the study no
  items, so prediction returns a NaN row (same as mask `drop`). Decisions are written to
  `box_audit.csv` (`split` = fit/selection) and to `last_audit` at prediction.
* Rules: 3D NIfTI + image branch + `annotation_boxes_file` required; mutually exclusive with
  `mask_mode` and mask-based slice selection (one lesion-location source). `use_annotation_boxes`
  (erasing) is allowed with `box_channel` and `dual_stream` (only the global volume is erased; the
  local crop comes from the unerased native volume) and rejected with `roi_crop`.
* Native grid = the first series' loaded array without resize. A box whose `source_shape`
  differs from the loaded native volume raises instead of being silently mis-placed.
* `roi_crop` and `dual_stream` resize the crop to a fixed size, so the aspect ratio is not preserved
  (same as mask `roi_crop`).
* `load()` refuses a saved branch whose `box_mode`, `box_margin`, `box_local_size` or
  `dual_stream_embed_dim` differs from the current config.
* **Risk:** the open axis-convention assumption of §4 matters more here than for erasing. A wrong
  mapping makes erasing hit some other background region, but it makes these modes crop or mark
  the wrong region at training and inference. Verify overlays on real data before trusting results.
* Smoke configs: `configs/synthetic_cpu_3d_box_{roi_crop,box_channel,dual_stream}.conf`
  (parametrised into `tests/e2e/test_modes_smoke.py`).

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
* Optional manifest column `prompt_source` (read-time default `""`, written only by tools that use it,
  so older manifests and their fingerprints are unchanged). `annotation_box` marks masks from a
  segmenter prompted with the Duke boxes (`prepare-data segment-mri-core`,
  `docs/segmentation-mri-core.md`). Such masks exist only where a box exists. `MaskRecord.prompt_source`
  is copied into `mask_audit*.csv`, `describe()` reports `prompt_sources` and
  `requires_box_at_inference` plus a limitation, and `image_branch.json` sets
  `mask.requires_box_at_inference` when the branch uses masks.

Modes: `none`, `roi_crop` (bbox of `mask_roi_classes` ± `roi_margin` × extent per
side, joint crop then joint resize), `mask_channel` (binary ROI appended), and
`foreground_only` (image zeroed outside ROI after the joint resize).

ROI geometry is always computed on the **native mask grid** (2D and 3D):

* `roi_crop`: bbox on the native grid; image and mask are cropped at native
  resolution and then resized to the target (3D uses an uncached-size native
  volume, i.e. phases z-scored / differences built on the original grid, then
  crop+resize; cannot be combined with `use_annotation_boxes`).
* `mask_channel` / `foreground_only`: the binary ROI is resampled to the target
  grid with the image's own interpolation (bilinear/trilinear, `align_corners=False`)
  and thresholded at `> 0` (rule `linear>0`, so ROI and interpolated image content
  coincide). If that is empty although the native ROI is not (a lesion smaller than
  the sampling stride), the `any_overlap` rule is used (output voxel is ROI iff its
  floor/ceil footprint overlaps any ROI voxel), so an accepted mask never yields an
  empty ROI. The rule used is written to the transform log (`roi_resample`).
  Emptiness is decided once on the native grid by `mask_empty_policy` (audited).

Reference grid: the grid of a study is the header geometry of its **first series**
(the key's `series_id`). Masks are validated against it with `Geometry.matches`
(shape equality, affine `atol=1e-4`) both when items are built and when an item is
loaded; for subtraction, `StudyVolume.affine` is also reported on that grid (phases
must agree within the legacy `same_grid` tolerance, `rtol=1e-4, atol=1e-3`). Joint
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
metric denominators). The cohort audit (`ImageBranch.audit`) never raises: a
policy `error` is reported as action `error`; with fold-trained masks it is
computed once per fold present in the mask manifest (`fold` column; fold-agnostic
masks give `fold=""`). Mask-based slice selection needs masks at inference too; a
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

Probabilities are computed as a float64 softmax of the logits (float32 saturates to
exactly 1.0 for confident studies and creates artificial ROC-AUC ties), both for
validation metrics and `predict_proba`, before study aggregation.

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
