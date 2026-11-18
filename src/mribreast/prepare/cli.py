"""``python -m mribreast.prepare <command>`` - explicit data-preparation jobs (own optional deps).

Commands (each reads inputs read-only and writes only below its output path):

* ``nbia-download``  - NBIA/TCIA series for ``--cohort all|outcome`` (legacy ``downloadAll.slurm`` /
  ``downloadDicom.download_all`` / ``download_oncotype``)
* ``dicom-to-nifti`` - verified ``<SeriesInstanceUID>.nii.gz`` (legacy ``transfer_dcm_to_nifti``)
* ``nifti-info``     - geometry listing of converted files (legacy ``read_nifti_metadata_images``)
* ``mama-mia-download`` - Synapse resource (legacy ``mama_mia/downloadMamamia.py``)
* ``mama-mia-map``   - MAMA-MIA x Duke series mapping (legacy ``mama_mia/readFromXlmx.py``)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "nbia-download" and args.cohort == "outcome" and not args.clinical_features_file:
        print("error: --cohort outcome needs --clinical_features_file", file=sys.stderr)
        return 2
    return int(args.func(args))
