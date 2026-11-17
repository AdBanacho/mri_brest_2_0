"""Artifact layout, manifests, fingerprints and atomic publication.

Layout (see docs/architecture.md, "Artifacts")::

    <checkpoint_root>/<train_id>/
        train_manifest.json          # schema_version, ids, resolved config, versions, label def, data fingerprint
        cohort.csv                   # patient_id, study_id, label (no raw identifiers beyond dataset ids)
        split_manifest.csv           # fold, patient_id, study_id, label, role
        fold_<k>/                    # published atomically; contains COMPLETE when finished
            fold_manifest.json
            image/ ...               # owned by the image branch
            tabular/ ...             # owned by the tabular branch
            COMPLETE
        COMPLETE                     # all folds complete
    <validation_root>/<train_id>/<eval_id>/   # published atomically with COMPLETE
"""

from mribreast.artifacts.core import (  # noqa: F401
    COMPLETE_FLAG,
    ArtifactMismatch,
    atomic_dir,
    environment_versions,
    file_fingerprint,
    code_version,
    is_complete,
    mark_complete,
    paths_fingerprint,
    read_json,
    write_json_atomic,
    write_text_atomic,
)
