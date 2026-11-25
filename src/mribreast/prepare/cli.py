"""``python -m mribreast.prepare <command>`` - explicit data-preparation jobs (own optional deps).

Commands (each reads inputs read-only and writes only below its output path):

* ``nbia-download``  - NBIA/TCIA series for ``--cohort all|outcome`` (legacy ``downloadAll.slurm`` /
  ``downloadDicom.download_all`` / ``download_oncotype``)
* ``dicom-to-nifti`` - verified ``<SeriesInstanceUID>.nii.gz`` (legacy ``transfer_dcm_to_nifti``)
* ``nifti-info``     - geometry listing of converted files (legacy ``read_nifti_metadata_images``)
* ``mama-mia-download`` - Synapse resource (legacy ``mama_mia/downloadMamamia.py``)
* ``mama-mia-map``   - MAMA-MIA x Duke series mapping (legacy ``mama_mia/readFromXlmx.py``)
* ``segment-mri-core`` - box-prompted lesion masks (MRI-CORE/SAM) + mask manifest (new;
  docs/segmentation-mri-core.md; needs ``.[segment]`` and the read-only ``mri_foundation`` checkout)
* ``mricore-embed``  - frozen MRI-CORE encoder embeddings per study (feature group ``mricore``)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


def _nbia_download(args: argparse.Namespace) -> int:
    from mribreast.prepare import nbia

    metadata = nbia.read_nbia_metadata(args.metadata_file)
    outcome = nbia.read_outcome_patients(args.clinical_features_file) if args.cohort == "outcome" else None
    studies = nbia.select_study_uids(metadata, args.cohort, outcome)
    if args.limit:
        studies = studies[: args.limit]
    print(f"nbia-download: {len(studies)} studies (cohort={args.cohort}) -> {args.dest}")
    if args.dry_run:
        return 0
    client = nbia.RestNbiaClient(base_url=args.base_url)
    manifest = nbia.download_studies(client, studies, args.dest, collection=args.collection, overwrite=args.overwrite,
                                     progress=print)
    path = nbia.write_manifest(manifest, Path(args.dest) / "download_manifest.csv")
    print(manifest["status"].value_counts().to_string())
    print(f"manifest: {path}")
    return int((manifest["status"].str.startswith("failed")).any())


def _dicom_to_nifti(args: argparse.Namespace) -> int:
    from mribreast.prepare import dicom_to_nifti

    report = dicom_to_nifti.convert_tree(args.download_root, args.image_root, backend=args.backend,
                                         overwrite=args.overwrite)
    frame = pd.DataFrame(report)
    out = Path(args.image_root) / dicom_to_nifti.SIDE_CAR_DIR / "conversion_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame["status"].value_counts().to_string() if len(frame) else "no series directories")
    print(f"report: {out}")
    return int(len(frame) and (~frame["status"].isin(["converted", "skipped_existing"])).any())


def _nifti_info(args: argparse.Namespace) -> int:
    import nibabel as nib

    rows = []
    for path in sorted(Path(args.image_root).glob("*.nii.gz")):
        image = nib.load(str(path))
        rows.append({"series_uid": path.name[: -len(".nii.gz")], "shape": " ".join(map(str, image.shape)),
                     "spacing": " ".join(f"{v:.6g}" for v in image.header.get_zooms()[:3]),
                     "orientation": "".join(nib.aff2axcodes(image.affine)), "dtype": str(image.get_data_dtype())})
    frame = pd.DataFrame(rows)
    if args.output:
        frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False) if len(frame) else "no NIfTI files")
    return 0


def _mama_download(args: argparse.Namespace) -> int:  # pragma: no cover - network
    from mribreast.prepare import mama_mia

    path = mama_mia.download_resource(mama_mia.make_synapse_client(args.token_env), args.dest, args.entity_id)
    print(f"mama-mia-download: {path}")
    return 0


def _mama_map(args: argparse.Namespace) -> int:
    from mribreast.prepare import mama_mia, nbia

    mama = mama_mia.read_mama_mia_table(args.table, sheet=args.sheet, uid_column=args.uid_column)
    duke = nbia.read_nbia_metadata(args.metadata_file)
    outcome = nbia.read_outcome_patients(args.clinical_features_file) if args.clinical_features_file else None
    mapping, summary = mama_mia.map_to_duke(mama, duke, outcome)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out / "mama_mia_duke_mapping.csv", index=False)
    (out / "mama_mia_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def make_segmenter(args: argparse.Namespace):
    """Build the MRI-CORE/SAM segmenter (tests replace this with a fake)."""
    from mribreast.prepare.segmentation.mri_core import MriCoreSegmenter

    repo = Path(args.mri_foundation_repo)
    weights_dir = repo / "pretrained_weights"
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return MriCoreSegmenter(repo, args.weights, sam_checkpoint=args.sam_checkpoint or weights_dir / "sam_vit_b_01ec64.pth",
                            mri_core_checkpoint=args.mri_core_checkpoint or weights_dir / "mri_foundation.pth",
                            checkpoint=args.checkpoint, device=device, batch_size=args.batch_size,
                            mask_output=args.mask_output)


def _cohort_from_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Studies built by the same cohort code as prepare/train (image branch only: no tabular filtering).

    Needed because ``mribreast prepare`` with ``mask_provider=local_files`` already requires the mask manifest
    that this command produces.
    """
    from mribreast.cli import build_cohort
    from mribreast.config import schema

    if not (args.clinical_features_file and args.metadata_file):
        raise SystemExit("segment-mri-core: the cohort needs both --clinical_features_file and --metadata_file")
    options = {}
    for item in args.cohort_option:
        key, sep, value = item.partition("=")
        if not sep or key not in schema.FIELDS:
            raise SystemExit(f"segment-mri-core: --cohort_option expects KEY=VALUE with a config key, got {item!r}")
        options[key] = value
    values = {"branches": "image", **options, "clinical_features_file": args.clinical_features_file,
              "metadata_file": args.metadata_file, "image_root": args.image_root}
    cfg = schema.resolve("prepare", values)
    frame = build_cohort(cfg).frame
    return frame, {"cohort_from": {"clinical_features_file": args.clinical_features_file,
                                   "metadata_file": args.metadata_file, "options": options,
                                   "profile": cfg["profile"], "n_studies": int(len(frame))}}


def _segment(args: argparse.Namespace) -> int:
    from mribreast.prepare.segmentation import pipeline

    settings = pipeline.SegmentSettings(
        input=args.input, input_channel=args.input_channel, slice_axis=args.slice_axis, slice_margin=args.slice_margin,
        context=args.context, min_window=args.min_window, clip_percentiles=tuple(args.clip_percentiles),
        threshold=args.threshold, clip_to_box=not args.no_clip_to_box, box_margin=args.box_margin,
        training_provenance=args.training_provenance, known_overlap=args.known_overlap)
    from_tables = bool(args.clinical_features_file or args.metadata_file)
    if sum((bool(args.cohort_csv), from_tables, bool(args.image_manifest_file))) != 1:
        raise SystemExit("segment-mri-core: give exactly one study source: --cohort_csv, "
                         "--clinical_features_file + --metadata_file (cohort built as in the stages), "
                         "or --image_manifest_file (ready 2D images)")
    reference = None
    if args.reference_mask_manifest:
        from mribreast.imaging.masks.local import LocalFileMaskProvider

        reference = LocalFileMaskProvider(args.reference_mask_manifest, slice_axis=args.slice_axis)
    segmenter = make_segmenter(args)
    common = dict(previews=args.previews, reference=reference, overwrite=args.overwrite)
    if args.cohort_csv or from_tables:
        if not (args.annotation_boxes_file and args.image_root):
            raise SystemExit("segment-mri-core: NIfTI studies need --annotation_boxes_file and --image_root")
        from mribreast.imaging.annotations import load_annotation_boxes

        if args.cohort_csv:
            studies = pd.read_csv(args.cohort_csv, dtype=str, keep_default_na=False)
            source = {"cohort_csv": args.cohort_csv}
        else:
            studies, source = _cohort_from_tables(args)
        if args.limit:
            studies = studies.head(args.limit)
        inputs = {**source, "annotation_boxes_file": args.annotation_boxes_file, "limit": args.limit,
                  "reference_mask_manifest": args.reference_mask_manifest}
        result = pipeline.run_nifti(studies, load_annotation_boxes(args.annotation_boxes_file), args.image_root,
                                    segmenter, args.output_dir, settings, inputs=inputs, **common)
    else:
        result = pipeline.run_images_2d(args.image_manifest_file, segmenter, args.output_dir, settings,
                                        inputs={"reference_mask_manifest": args.reference_mask_manifest}, **common)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"segment-mri-core: mask manifest {Path(args.output_dir) / 'mask_manifest.csv'}")
    return 1 if result["summary"]["n_errors"] else 0


def build_parser() -> argparse.ArgumentParser:
    from mribreast.prepare import mama_mia, nbia

    parser = argparse.ArgumentParser(prog="python -m mribreast.prepare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("nbia-download", help="download NBIA/TCIA DICOM series")
    p.add_argument("--metadata_file", required=True, help="NBIA digest (sheet Metadata) or CSV")
    p.add_argument("--clinical_features_file", default=None, help="needed for --cohort outcome")
    p.add_argument("--cohort", choices=nbia.COHORTS, default="outcome")
    p.add_argument("--dest", required=True, help="download root; series land in <dest>/<SeriesInstanceUID>/")
    p.add_argument("--collection", default=nbia.DUKE_COLLECTION)
    p.add_argument("--base_url", default=nbia.DEFAULT_BASE_URL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.set_defaults(func=_nbia_download)
    p = sub.add_parser("dicom-to-nifti", help="convert verified series to <SeriesInstanceUID>.nii.gz")
    p.add_argument("--download_root", required=True)
    p.add_argument("--image_root", required=True, help="output directory (pipeline image_root)")
    p.add_argument("--backend", choices=("nibabel", "sitk"), default="nibabel")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_dicom_to_nifti)
    p = sub.add_parser("nifti-info", help="list geometry of converted NIfTI files")
    p.add_argument("--image_root", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_nifti_info)
    p = sub.add_parser("mama-mia-download", help="download the MAMA-MIA Synapse resource")
    p.add_argument("--dest", required=True)
    p.add_argument("--entity_id", default=mama_mia.LEGACY_ENTITY_ID)
    p.add_argument("--token_env", default=mama_mia.TOKEN_ENV)
    p.set_defaults(func=_mama_download)
    p = sub.add_parser("mama-mia-map", help="map MAMA-MIA series UIDs onto Duke series/patients")
    p.add_argument("--table", required=True, help=f"MAMA-MIA table ({mama_mia.LEGACY_TABLE_NAME}) or CSV")
    p.add_argument("--sheet", default=mama_mia.LEGACY_SHEET)
    p.add_argument("--uid_column", default=mama_mia.LEGACY_UID_COLUMN)
    p.add_argument("--metadata_file", required=True)
    p.add_argument("--clinical_features_file", default=None)
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=_mama_map)

    from mribreast.prepare.segmentation import pipeline as seg

    d = seg.SegmentSettings()
    p = sub.add_parser("segment-mri-core", help="box-prompted lesion masks (MRI-CORE/SAM) -> mask manifest",
                       description="See docs/segmentation-mri-core.md. Output feeds mask_provider=local_files.")
    p.add_argument("--cohort_csv", default=None, help="studies: patient_id, study_id, series_uids (cohort.csv)")
    p.add_argument("--annotation_boxes_file", default=None, help="Duke Annotation_Boxes.xlsx (or CSV)")
    p.add_argument("--image_root", default=None, help="directory of <SeriesInstanceUID>.nii.gz")
    p.add_argument("--clinical_features_file", default=None, help="build the cohort as the stages do (with --metadata_file)")
    p.add_argument("--metadata_file", default=None, help="NBIA digest (with --clinical_features_file)")
    p.add_argument("--cohort_option", action="append", default=[], metavar="KEY=VALUE",
                   help="cohort-defining config value, e.g. profile=general phase_order=explicit (repeatable)")
    p.add_argument("--image_manifest_file", default=None, help="ready 2D images with a lesion_box column")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mri_foundation_repo", required=True, help="read-only mri_foundation checkout")
    p.add_argument("--weights", choices=("mri_core", "sam", "state_dict"), default="mri_core")
    p.add_argument("--sam_checkpoint", default=None, help="default <repo>/pretrained_weights/sam_vit_b_01ec64.pth")
    p.add_argument("--mri_core_checkpoint", default=None, help="default <repo>/pretrained_weights/mri_foundation.pth")
    p.add_argument("--checkpoint", default=None, help="full fine-tuned state dict (--weights state_dict)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:<n>")
    p.add_argument("--batch_size", type=int, default=4, help="slices per encoder batch")
    p.add_argument("--mask_output", choices=("best_iou", "0", "1", "2"), default="best_iou")
    p.add_argument("--input", default=d.input, help="post_minus_pre:<k> | phase:<k> (phase 0 = pre-contrast)")
    p.add_argument("--input_channel", type=int, default=d.input_channel, help="ready 2D images: channel to segment")
    p.add_argument("--slice_axis", type=int, default=d.slice_axis)
    p.add_argument("--slice_margin", type=int, default=d.slice_margin, help="extra slices beyond the box per side")
    p.add_argument("--context", type=float, default=d.context, help="window side / longest box side")
    p.add_argument("--min_window", type=int, default=d.min_window, help="minimum window side (pixels)")
    p.add_argument("--clip_percentiles", type=float, nargs=2, default=list(d.clip_percentiles))
    p.add_argument("--threshold", type=float, default=d.threshold, help="logit threshold")
    p.add_argument("--no_clip_to_box", action="store_true", help="keep mask voxels outside box +- box_margin")
    p.add_argument("--box_margin", type=float, default=d.box_margin)
    p.add_argument("--training_provenance", choices=("frozen_external", "unknown"), default=d.training_provenance)
    p.add_argument("--known_overlap", default=d.known_overlap)
    p.add_argument("--reference_mask_manifest", default=None, help="mask manifest of reference masks (Dice/IoU QC)")
    p.add_argument("--previews", type=int, default=0, help="overlay PNGs for the first N segmented items")
    p.add_argument("--limit", type=int, default=0, help="first N cohort rows only (smoke runs)")
    p.add_argument("--overwrite", action="store_true", help="replace outputs made with different settings")
    p.set_defaults(func=_segment)

    from mribreast.prepare import mricore_embed

    mricore_embed.add_parser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "nbia-download" and args.cohort == "outcome" and not args.clinical_features_file:
        print("error: --cohort outcome needs --clinical_features_file", file=sys.stderr)
        return 2
    return int(args.func(args))
