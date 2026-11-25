# Screening campaign: 30 training configurations

Design: one baseline (B0) + one-factor-at-a-time variations + a few targeted interactions,
because the full configuration space (~10^11 variants) cannot be searched with 257 studies.
Tabular-only runs (cheap) get a near-full factorial. Every run is validated at
`fusion_alpha` 0, 0.25, 0.5, 0.75, 1 without retraining (1 = image only, 0 = tabular only).

```bash
bash configs/campaign/submit.sh --dry-run    # what would be submitted
bash configs/campaign/submit.sh              # everything, with dependencies
bash configs/campaign/submit.sh B01 A01      # a subset (prefix match)
```

`submit.sh` first submits the MRI-CORE embedding job (needed by A03) and the MRI-CORE
segmentation (needed by C02/C03) unless their outputs exist, then train → validate per run file,
then one `summarize` over the whole campaign (`runs/campaign/summary`).

## Baseline B0 (`base.conf`)
Cohort `general` / `explicit` / `phase_count=4` (257 studies, 159/98). 3D `densenet121`,
`256 256 64`, no subtraction, background erasing outside the Duke box (`use_annotation_boxes`),
`ce_soft_sensitivity` λ=0.05, adam, lr 1e-4, batch 4, 30 epochs, 5 outer folds,
`selection_split=shared`. Tabular: clinical + kinetic + morphology + heterogeneity, xgboost, lasso.

## Runs
| ID | File | Change from B0 | Question |
|---|---|---|---|
| A1–A2 | `A01_tabular_clinical` | tabular only; clinical; no selector; xgboost / mlp | clinical-only reference |
| A3–A4 | `A03_tabular_rich` | tabular only; + imaging features; xgboost / mlp | value of radiomic features |
| A5–A6 | `A03_tabular_rich` | tabular only; clinical + mricore; xgboost / mlp | value of MRI-CORE embeddings |
| A7–A8 | `A03_tabular_rich` | tabular only; imaging + mricore; xgboost / mlp | complementary? |
| B1 | `B01_models` | — (baseline) | reference |
| B2–B3 | `B01_models` | resnet10 / resnet18 | architecture |
| B4–B5 | `B04_subtraction` | post_minus_pre / consecutive | enhancement information |
| B6 | `B06_no_erasing` | no erasing, full volume | does erasing help |
| B7 | `B06_no_erasing` | box_mode=roi_crop (no erasing) | lesion-centred crop |
| B8–B9 | `B08_box_inputs` | box_mode=box_channel / dual_stream | box as model input |
| B10 | `B10_loss_ce` | loss=ce | sensitivity loss |
| B11 | `B11_lambda` | sensitivity_lambda=0.2 | its strength |
| B12 | `B12_recipe` | adamw + cosine + early stopping | training recipe |
| B13 | `B13_selection_separate` | selection_split=separate | threshold honesty |
| B14 | `B14_roi_crop_subtraction` | roi_crop + post_minus_pre | best-guess combination |
| B15 | `B15_dual_stream_subtraction` | dual_stream + post_minus_pre | combination |
| B16 | `B16_box_channel_resnet18` | box_channel + resnet18 | interaction |
| B17 | `B17_roi_crop_resnet10` | roi_crop + resnet10 | small crop × small model |
| C1 | `C01_2d_center` | 2D, centre slice, no masks | 2D reference |
| C2 | `C02_2d_mricore_roi_crop` | 2D, MRI-CORE masks, roi_crop, largest-lesion slice | predicted masks |
| C3 | `C03_2d_mricore_channel` | 2D, MRI-CORE masks as channel, all lesion slices | mask as channel |
| D1 | `D01_legacy_order` | legacy cohort (row order, 5 series; 261 studies) | size of the phase-order bug |
| D2 | `D02_three_class` | 3 classes (159/76/22), argmax | exploratory |

## Reading the results
* 98 positives: an AUC has a standard error of about 0.035, so single-run differences below
  ~0.05 are noise. **Phase 2:** rerun the 3 best B/C runs with `TRAIN[seed]=1` and `=2`
  (copy the run file, change only the seed; a new seed is a new `train_id`).
* Compare a variation with B1 at the same `fusion_alpha`; alpha 1 isolates the image model.
* Do not judge D2 by AUC alone: class 2 has 22 studies.

## Rules
* Adding a run: new file, `source` the base, override keys. A key moved into `TRAIN_GRID`
  must be `unset` from `TRAIN` first. Never define the same row in two files.
* `CLEAN_BEFORE_*` stays `false` (all runs share the roots).
* Checkpoints and logs are on scratch (purged periodically); validation/summary in `runs/campaign/`.
* Not verified on GPU: runtime per run is unknown; the 24 h train limit is a guess. Check B1 first.
