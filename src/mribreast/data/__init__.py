"""mribreast.data — Duke sources, cohort building, label encoding, allowed-predictor policy.

Import-time side-effect free: no file is read until a function is called.
"""

from mribreast.data.duke import (
    CLINICAL_PREDICTOR_COLUMNS,
    FEATURE_GROUPS,
    SENSITIVE_CLINICAL_PREDICTOR_COLUMNS,
    CohortError,
    DukeCohortSource,
    SeriesRecord,
    StudyRecord,
    cohort_from_csv_frame,
    cohort_to_csv_frame,
    parse_phase_index,
    read_clinical_workbook,
    read_metadata_digest,
)
from mribreast.data.predictor_policy import PredictorPolicy, PredictorPolicyError, load_policy, policy_from_cfg

__all__ = [
    "CLINICAL_PREDICTOR_COLUMNS", "FEATURE_GROUPS", "SENSITIVE_CLINICAL_PREDICTOR_COLUMNS", "CohortError",
    "DukeCohortSource", "SeriesRecord", "StudyRecord", "cohort_from_csv_frame", "cohort_to_csv_frame",
    "parse_phase_index", "read_clinical_workbook", "read_metadata_digest", "PredictorPolicy",
    "PredictorPolicyError", "load_policy", "policy_from_cfg",
]
