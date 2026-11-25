# Lesion segmentation with MRI-CORE, prompted by Duke annotation boxes

Status: **implemented (zero-shot path); real-data quality open**. Findings marked *verified* were
checked by running code in this environment (Windows, CPU only, no Duke images). Everything that
needs real Duke data or a GPU is marked *open*.

## 1. Goal and boundaries

* A separate preparation step, `python -m mribreast prepare-data segment-mri-core`, segments the lesion in
  the Duke DCE series. It writes per-study NIfTI masks (or 2D PNG masks for ready 2D images) and a
  **mask manifest** in the existing `LocalFileMaskProvider` format.
* The classifier consumes the result **only** through the existing external-mask path
  (`mask_provider=local_files`, `mask_manifest_file`, `mask_kind=predicted`,
  `mask_mode=roi_crop|mask_channel|foreground_only`, mask-based slice selection). No parallel mechanism
  is added, and `train`/`validate` never import the segmentation code or its dependencies.
* `C:\code\mri_foundation` (MRI-CORE, mazurowski-lab, SHA `0f4f452614013873f8bb58a1fd613dbe36b58aba`) is
  third-party and **read-only**. We import its SAM model classes and load weights with our own loader
  (§2.4). The repo's SHA and the weight files' sha256 go into every manifest row and the run manifest.
* Optional dependencies: `.[segment]` = `einops`, `torchvision` (needed by `models/sam/modeling/vit.py`).

## 2. Findings: mri_foundation

### 2.1 What the code does

| Item | Finding (file) |
| --- | --- |
| Architecture | SAM ViT-B (`models/sam/build_sam.py::_build_sam`): image encoder, prompt encoder, two-way-transformer mask decoder. `use_rel_pos=False`, windowed attention (window 14) with global attention at blocks 2, 5, 8, 11. |
| Input | 3-channel image, `image_size=1024`, ImageNet normalisation (`normalize_type` `sam` or `ours`; `medsam` = per-image min-max). `utils/dataset.py::__getitem__` converts the slice PNG to RGB, resizes it to 1024×1024 (not aspect-preserving), min-max scales it to [0, 1], quantises to uint8 and then applies `ToTensor` + `Normalize`. |
| Output | Low-resolution logits at 256×256 (`out_size`). The mask is `logit > 0` (`main.py::binarize_pred`). `multimask_output=True` drops mask token 0 and returns `num_cls` masks. |
| Prompts | `prompt_encoder` accepts `points`, `boxes` (B×4, `x0,y0,x1,y1` in the 1024 frame) and `masks`. **`main.py` always calls it with `boxes=None`**: the published fine-tuning is prompt-free. `utils/dataset.py` can build box prompts from masks (`if_prompt=True, prompt_type='box'`), but it calls `get_top_boxes`, which is not defined anywhere, and `main.py` sets `if_prompt=False`. |
| Fine-tuning | `main.py`: few-shot (5 patients, one random slice each), adapters (`finetune_type=adapter`, encoder blocks 0, 1, 10, 11 + decoder), `DiceCELoss`, 3000 epochs forced, the checkpoint path to "ours" is hard-coded to the authors' server (`/data/humanBodyProject/...`), and `num_cls` is forced to 1. It is a research script, not a reusable training entry point. |
| Loading quirks | `torch.load(..., map_location='cuda:0')` is hard-coded, so it fails on CPU. `pretrained_sam=True` loads `pretrained_weights/sam_vit_b_01ec64.pth` relative to the **current working directory**. A missing SAM file triggers an interactive download prompt (`input()`). |

### 2.2 What the weight files contain (*verified*)

* `pretrained_weights/mri_foundation.pth` (435,830,190 bytes) = a DINOv2 checkpoint: `teacher.backbone.*`
  (ViT-B/16, 12 blocks stored in DINOv2 chunks `blocks.<chunk>.<i>`, `patch_embed`, `pos_embed`
  `(1, 197, 768)`, i.e. trained at 224 px, `cls_token`, `mask_token`, `norm`) plus `dino_head.*`. It has **no
  neck, no prompt encoder and no mask decoder**. This is the "MRI-CORE" checkpoint referred to in the
  README as `MRI_CORE_vitb.pth`.
* `pretrained_weights/sam_vit_b_01ec64.pth` (375,042,383 bytes) = original SAM ViT-B.
* Building "ours" as the repo does (`pretrained_sam=True`) gives this mix. It was traced weight by weight
  (`tests/unit/test_segmentation_mri_core.py::test_loader_matches_the_repos_own_builder` now checks our
  loader against the repo's builder tensor by tensor):

  | Component | Source |
  | --- | --- |
  | `image_encoder.patch_embed`, all 144 block tensors | MRI-CORE |
  | `image_encoder.pos_embed` | **SAM**. The MRI-CORE `pos_embed` is interpolated and then stored under the wrong key (`backbone.pos_embed`) and silently dropped by `strict=False`. |
  | `image_encoder.neck` | SAM (MRI-CORE has none) |
  | `prompt_encoder` | SAM |
  | `mask_decoder` with `num_cls=3` | SAM, entirely |
  | `mask_decoder` with `num_cls=1` (what `main.py` forces) | SAM except **`mask_tokens` and part of `iou_prediction_head`, which are random**. The shape mismatch sends loading to `load_from`, which drops those keys. |

  So MRI-CORE in this repo is "MRI-pretrained encoder + SAM's neck, prompt encoder and decoder". The
  decoder was never trained on MRI-CORE features, so the pairing is only coherent after fine-tuning,
  unless box prompts carry the segmentation (§2.3).

### 2.3 Do box prompts work zero-shot? (*verified on toy images only*)

One-off CPU experiment (2026-11-24). The 256² bright case is kept as
`test_segmentation_mri_core.py::test_zero_shot_box_prompt_on_a_toy_lesion`. Synthetic 2D "lesions" (bright,
low-contrast, irregular, rim-enhancing) on a smooth background with a bright band and noise. Box = tight
box + 10% margin. Preprocessing follows the repo's dataset path. Dice against the drawn lesion:

| Model | 60×44 px lesion in 256² | 8×14 px lesion in 512² | 16×24 px lesion in 512² |
| --- | --- | --- | --- |
| SAM ViT-B, box | 0.96–0.98 | 0.77–0.83 | 0.73–0.93 |
| MRI-CORE, `num_cls=1` (as `main.py`), box | **0.00–0.52** (random mask token) | not run | not run |
| MRI-CORE, `num_cls=3` (SAM decoder), box | 0.92–0.97 | 0.81–0.86 | 0.69–0.90 |
| Any model, no prompt | ≈0 (whole image or empty) | | |

Conclusions:

1. Zero-shot box prompting is **mechanically sound** only with the 3-output SAM decoder (`num_cls=3`, best mask
   picked by predicted IoU). The repo's `num_cls=1` configuration must not be used without fine-tuning.
2. Box prompts constrain the output well: 93–100% of the mask lies inside the box.
3. Small lesions lose accuracy because the decoder predicts at 256² for the whole frame. Hence the
   context crop in §4.2.
4. **Open:** toy blobs are much easier than real DCE-MRI (parenchymal enhancement, vessels, skin, motion).
   Whether the quality is acceptable on Duke can only be decided on real slices against reference masks
   (§6). The pipeline is the same for zero-shot and fine-tuned weights, so implementation does not wait for
   this decision.

### 2.4 Environment and loading

* The README pins Python 3.12, torch 2.6 and CUDA 12.4. The code ran here on Python 3.11.5 + torch 2.9.1 (CPU) +
  torchvision 0.24.1 + einops 0.8.2. Nothing in the inference path is version-specific. On Helios the
  `ML-bundle` torch is used and `torchvision` must match it: `segment_mri_core.slurm` builds its own
  `segment-<hash>` venv (the config's packages + `einops` + `torchvision`, numpy/torch pinned to the module,
  as in `run_experiment.slurm`), so pip resolves the matching `torchvision`. `timm` is not needed (`tiny_vit_sam` is never imported).
* Our loader (`mribreast.prepare.segmentation.mri_core`) does not call `sam_model_registry`'s loading
  path:
  1. It imports `models/sam` from the configured repo path under a private module name, so no generic
     top-level `models` package is registered and nothing is written to the repo (bytecode off during import).
  2. It builds `_build_sam(..., num_classes=3, checkpoint=None)` with adapters off.
  3. It loads SAM ViT-B strictly, except for the `rel_pos` keys that `use_rel_pos=False` does not use.
  4. For `--weights mri_core` it overlays `patch_embed` and the 144 block tensors from the DINOv2 teacher
     (`backbone.blocks.<c>.<i>.X → image_encoder.blocks.<i>.X`, `fc → lin`) and asserts that every one was
     replaced.
  5. `pos_embed` stays SAM's, which reproduces what the authors' code actually ran. This is recorded as
     `pos_embed_source=sam`.

  A parity test compares this loader with the repo's own builder when the weights are present.
* `--weights sam` (plain SAM ViT-B) is available as a baseline. `--weights state_dict --checkpoint <file>`
  loads a full fine-tuned state dict (for example from `main.py` or a future fold-trained model).

## 3. Findings: mri_brest_2_0

* **Axes:** NIfTI is read with nibabel without reorientation. For Duke, array axes `(0, 1, 2)` = DICOM
  `(column, row, slice)`. The Duke box is half-open `((c0, r0, s0), (c1, r1, s1))` on that grid
  (`imaging/annotations.py`). The open assumption in `docs/imaging.md` §4 (mapping never checked against
  overlays) applies here even more strongly. A wrong mapping prompts SAM at the wrong place.
* **2D slices:** slicing along `slice_axis=2` keeps the remaining axes in increasing order, so a slice is
  `[H = axis 0 (column), W = axis 1 (row)]`. In SAM's `x, y` convention (`x` = width):
  `box_xyxy = (r0, c0, r1, c1)`. **Breast_MRI_207** `((358, 374, 34), (365, 387, 40))` → slices 34–39,
  `box_xyxy = (374, 358, 387, 365)`. This is a 13×7 px box on a 512² slice, the small-lesion case of §2.3.
* **Mask manifest:** key columns `patient_id, study_id, series_id, slice_id` + `path` (+ `class_codes`,
  `source`, `model_id`, `model_version`, `grid_source`, `training_provenance`, `known_overlap`, `fold`).
  `slice_id=""` = whole-volume NIfTI mask, from which the provider extracts 2D slices on demand. So **one NIfTI
  per study serves both the 3D path and 2D slicing of NIfTI**. The mask grid must equal the first series'
  header geometry (shape + affine, `atol=1e-4`).
* `mask_manifest_file` is an identity path, so its content fingerprint enters `train_id`. A new segmentation run
  therefore produces a new `train_id`.
* `MaskRecord`/`mask_audit` carry model id/version and provenance. There was no field saying that a
  mask was produced from an annotation box.
* `prepare/mama_mia.py` only maps MAMA-MIA series UIDs to Duke. It does **not** import MAMA-MIA's expert
  segmentations yet.

## 4. Design

### 4.1 Inputs

| Input | Source |
| --- | --- |
| Studies | `--clinical_features_file` + `--metadata_file` (+ `--cohort_option KEY=VALUE`, e.g. `profile=general`): the cohort is built by the stages' own `DukeCohortSource`, with `branches=image` so tabular availability does not filter studies. This is needed because `mribreast prepare` with `mask_provider=local_files` already requires the mask manifest this step produces. Alternatively `--cohort_csv` (`patient_id, study_id, series_uids`), or `--image_manifest_file` for ready 2D images |
| Boxes | `--annotation_boxes_file` (Duke format) → `annotation_coverage` (same eligibility statuses as `box_mode`); for ready 2D images, a `lesion_box` column `[h0, h1, w0, w1]` (half-open, the synthetic 2D convention) |
| Image to segment | `--input post_minus_pre:1` (default: first post-contrast minus pre-contrast, where enhancing lesions are most conspicuous) or `phase:<k>` |
| Model | `--mri_foundation_repo`, `--weights mri_core|sam|state_dict`, `--sam_checkpoint`, `--mri_core_checkpoint`, `--checkpoint` |

### 4.2 Box → prompt, per slice

1. Slices: `s0 − slice_margin … s1 + slice_margin − 1` (clipped). Slices outside that range are empty in the mask.
2. 2D box on the slice: `(h0, h1, w0, w1) = (c0, c1, r0, r1)` (axis 0 → H, axis 1 → W).
3. **Context window:** a square window centred on the box, with side `max(ceil(context × max(h, w)), min_window)`
   (defaults `context=3`, `min_window=64`), shifted to stay inside the slice and zero-padded only when the slice
   is smaller than the window. This makes small lesions larger in the 1024 frame (§2.3).
4. Intensity: the input volume is clipped to its [0.5, 99.5] percentiles (per study). The window is min-max
   scaled to [0, 1], quantised to uint8 like the repo, replicated to 3 channels, bilinearly resized to 1024²
   (`align_corners=False`) and ImageNet-normalised.
5. Prompt: `box_xyxy` relative to the window × `1024 / side`.
6. Decoder: `multimask_output=True` (3 masks). Choose `--mask_output best_iou` (default) or a fixed index.
   The 256² logits are bilinearly upsampled to the window, then `logit > threshold` (default 0.0).
7. Pasting: the window mask goes back into the native slice. With `--clip_to_box` (default on), voxels outside
   the box ± `box_margin` (default 0.1, same `expand_box` rule as `box_mode`) are removed. The fraction
   removed is recorded as QC.

### 4.3 Outputs

```
<output_dir>/
  masks/<study_id>.nii.gz      uint8 {0: background, 1: lesion}, shape + affine of the study's first series
  masks/<study_id>.json        per-study sidecar: settings fingerprint, QC, slices, timing
  mask_manifest.csv            LocalFileMaskProvider format (+ prompt_source column)
  segmentation_qc.csv          one row per study (see §6)
  annotation_coverage.csv      box eligibility (same statuses as box_mode)
  segmentation_cohort.csv      the studies considered (patient_id, study_id, series_uids)
  segmentation_manifest.json   settings, settings_fingerprint, mri_foundation SHA, weight sha256s,
                               code/env versions, counts, limitations
  previews/<study_id>.png/json slice with the largest mask: image | overlay (first --previews studies)
  COMPLETE
```

Manifest rows: `source=predicted`, `model_id=mri_core_sam_vitb` (or `sam_vitb`, `state_dict:<sha12>`),
`model_version=<weights sha12>+mf<repo sha7>`, `grid_source=<first series uid>`,
`training_provenance=frozen_external` (zero-shot), `known_overlap` (§5), `prompt_source=annotation_box`,
`class_codes={"0": "background", "1": "lesion"}`, `fold=""`.

Ready 2D images produce `masks/<key>.png` rows with `slice_id` set.

### 4.4 Determinism and resume

* Inference runs under `torch.no_grad`, in eval mode, with fixed preprocessing. CPU runs are bit-reproducible.
  CUDA may differ in the last bits of the logits, which can flip only voxels with `|logit| ≈ 0`.
* `settings_fingerprint` = sha256 over every setting that changes masks (model, weight sha256s, repo SHA,
  input, context, thresholds, clipping, slice margin, preprocessing version). A study is skipped when its
  NIfTI and sidecar exist and the sidecar's fingerprint matches. Each file is written to a temp file in the
  same directory and published with `os.replace`. The manifest, QC table and `COMPLETE` are written last.
  An output directory holding a different fingerprint is refused unless you pass `--overwrite`.

### 4.5 Box needed at inference

These masks exist only where a Duke box exists. A classifier that uses them needs the box at test time,
exactly like `box_mode`. Changes in mri_brest_2_0:

* the mask manifest gains an optional `prompt_source` column (`""` | `annotation_box` | free text);
* `MaskRecord.prompt_source` → `mask_audit.csv` column `prompt_source`;
* `LocalFileMaskProvider.describe()` → `requires_box_at_inference: true` plus a limitation. This flows into
  `image_branch.json` under `mask.provider` and, explicitly, `mask.requires_box_at_inference`.

Studies without an eligible box get **no manifest row**, so the classifier's `mask_missing_policy` decides
(`error` / `full_image` / `drop`), and the decision is audited as before.

## 5. Leakage and provenance

* Zero-shot segmentation uses no Oncotype labels and no cohort-derived training. The masks are fold-agnostic
  (`mask_provenance=frozen_external`).
* **Open: pre-training overlap.** MRI-CORE was pre-trained (self-supervised, DINOv2) on a large MRI collection
  assembled by the same lab (Duke). Whether it contains Duke-Breast-Cancer-MRI volumes has not been checked here.
  The default `known_overlap` is `"unknown: MRI-CORE pretraining data may include Duke breast MRI (self-supervised, no labels)"`.
  Self-supervised exposure involves no labels, so it is not label leakage, but it must be stated.
* **Fine-tuning (only if zero-shot fails on real data):** the reference would be expert segmentations, for example
  MAMA-MIA (Duke subset). Two options:
  1. Fine-tune on patients outside the Oncotype cohort. The masks stay `frozen_external` with `known_overlap=none`.
  2. Fine-tune per outer fold on fit patients only. Masks get `fold=k`, `training_provenance=fold_trained:k`,
     `mask_provenance=fold_trained`, which `LocalFileMaskProvider` already supports.

  Segmentation labels are not Oncotype labels, so using the same patients does not leak the outcome. Option 2 is
  still the conservative choice when the evaluation patients' masks would otherwise come from a model that saw
  those patients' segmentations. Either way, the patients used must be listed in the manifest. Fine-tuning is
  not implemented in this stage. The `state_dict` weights path accepts its result.

## 6. Quality checks

`segmentation_qc.csv` (one row per study): `status` (`segmented`, a coverage status, or `error`),
`n_slices_prompted`, `n_slices_nonempty`, `mask_voxels`, `box_voxels`, `mask_box_fraction`
(= mask voxels / box voxels), `frac_removed_by_clip`, `mean_pred_iou`, `empty_mask`, and with
`--reference_mask_manifest` (any mask manifest on the same grid, for example MAMA-MIA once imported):
`dice`, `iou`, `reference_voxels`. A summary goes into `segmentation_manifest.json`
(median Dice, empty rate, mask/box fraction quantiles). Preview PNGs show the slice with the largest mask.

Acceptance on real data (*open*, needs Duke NIfTI and preferably a GPU):

1. Overlay previews for ≥10 studies, including Breast_MRI_207. This also checks the axis convention of §3.
2. Dice against MAMA-MIA where available. Proposed bar: median ≥ 0.7, otherwise go to fine-tuning (§5).
3. Empty-mask rate below 5%, and `frac_removed_by_clip` mostly below 0.1. A large value means the prompt
   frame is wrong.

## 7. Running it

```bash
python -m pip install -e ".[segment]"
python -m mribreast prepare-data segment-mri-core \
  --clinical_features_file data/duke/Clinical_and_Other_Features.xlsx \
  --metadata_file data/duke/Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx \
  --cohort_option profile=general --cohort_option phase_order=explicit \
  --annotation_boxes_file data/duke/Annotation_Boxes.xlsx --image_root /net/storage/pr3/plgrid/plggsiwomipipan/aBanacho_duke_oncotype/tciaNifti \
  --mri_foundation_repo ../mri_foundation --weights mri_core \
  --output_dir /net/scratch/hscra/plgrid/plgabanacho/mri_brest_2_0/segmentation/mri_core_zeroshot --device cuda --previews 20
# then in experiment.conf: [mask_manifest_file]=/net/scratch/hscra/plgrid/plgabanacho/mri_brest_2_0/segmentation/mri_core_zeroshot/mask_manifest.csv
#   TRAIN: [mask_provider]=local_files [mask_kind]=predicted [mask_provenance]=frozen_external
```

Exit code 0 means every eligible study was segmented (`COMPLETE` written). Exit code 1 means some failed:
they are listed in `segmentation_qc.csv`, and a rerun retries them while reusing the rest. Useful flags:
`--limit N` (smoke run), `--weights sam` (baseline), `--reference_mask_manifest` (Dice/IoU),
`--device cpu|cuda`, `--batch_size`.

`configs/duke_2d_mri_core_mask.conf` is the full example (2D slices selected by mask area,
`mask_mode` ∈ {roi_crop, mask_channel}). `docs/slurm.md` §"Lesion segmentation" describes the Helios GPU job.

## 8. Implementation map and tests

| Piece | Code | Tests |
| --- | --- | --- |
| Box/slice/window geometry (axis mapping, Breast_MRI_207, padding, clipping) | `prepare/segmentation/geometry.py` | `tests/unit/test_segmentation.py`: the lesion is recovered on every `slice_axis` with an asymmetric box, so a transposition fails |
| Pipeline: inputs, per-study atomic outputs, resume, refusal of mixed settings, QC, Dice, 2D images | `prepare/segmentation/pipeline.py` | `tests/unit/test_segmentation.py` (fake segmenter in `tests/fakes.py`) |
| MRI-CORE/SAM loader + segmenter | `prepare/segmentation/mri_core.py` | `tests/unit/test_segmentation_mri_core.py`: real weights; skipped without them or the `.[segment]` extras; asserts the mri_foundation checkout is unchanged |
| CLI `prepare-data segment-mri-core` | `prepare/cli.py` | `test_segmentation.py::test_cli_runs_with_injected_segmenter` |
| `prompt_source` → `MaskRecord`, `mask_audit*.csv`, `describe()`, `image_branch.json` | `contracts.py`, `imaging/masks/{local,policies}.py`, `models/image_branch.py` | `test_segmentation.py`, `tests/e2e/test_segmentation_e2e.py` |
| End to end: segment (fake) → train → validate → summarize, 3D `roi_crop` and 2D `mask_max_area` + `mask_channel` | | `tests/e2e/test_segmentation_e2e.py` |

Not implemented: fine-tuning (§5) and importing MAMA-MIA masks as a reference mask manifest. The MAMA-MIA
masks' grid must first be checked against our NIfTI conversion.
