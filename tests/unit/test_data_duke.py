"""Duke cohort building: labels, eligibility, ordering, groups, sensitive opt-in, policy, density."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mribreast.contracts import LABEL, PATIENT_ID, SERIES_UIDS, STUDY_ID, label_definition
from mribreast.data.density import density_features, merge_density, read_density_assessments
from mribreast.data.duke import (
    CohortError, DukeCohortSource, parse_phase_index, read_clinical_workbook, read_metadata_digest,
    select_studies_general, select_studies_legacy, study_records,
)
from mribreast.data.imaging_features import ImagingFeatureError, merge_imaging_features
from mribreast.data.predictor_policy import PredictorPolicyError, check_label_leakage, load_policy
from mribreast.synthetic.duke import DENSITY_LEGEND, write_duke_tables


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return write_duke_tables(tmp_path_factory.mktemp("duke"), n_patients=40, seed=3)


def _cfg(tables, **extra):
    cfg = {"clinical_features_file": str(tables["clinical_features_file"]), "metadata_file": str(tables["metadata_file"]),
           "imaging_features_file": str(tables["imaging_features_file"]), "density_file": str(tables["density_file"]),
           "profile": "duke_legacy_3d", "phase_order": "metadata", "phase_count": 5, "label_scheme": "oncotype_binary_18",
           "positive_class": 1, "branches": ("image", "tabular"), "feature_groups": ("clinical",),
           "imaging_patient_id_column": "Patient ID", "allow_missing_imaging_features": False,
           "include_sensitive": False, "prediction_time": "mri", "allowed_predictors_file": ""}
    cfg.update(extra)
    return cfg


# --------------------------------------------------------------------------- labels
@pytest.mark.parametrize("score,binary,three", [(0, 0, 0), (18, 0, 0), (18.0001, 1, 1), (31, 1, 1), (31.0001, 1, 2), (76, 1, 2)])
def test_label_boundaries(score, binary, three):
    assert label_definition("oncotype_binary_18").encode(score) == binary
    assert label_definition("oncotype_3class_18_31").encode(score) == three


def _metadata(rows):
    return pd.DataFrame(rows, columns=["row_index", PATIENT_ID, STUDY_ID, "series_uid", "series_description"])


def _clinical(scores):
    return pd.DataFrame({"patient_id": list(scores), "oncotype_score": list(scores.values())})


def test_legacy_eligibility_and_first_five_in_file_order():
    desc = ["ax dyn 2nd pass", "ax dyn pre", "ax dyn 1st pass", "ax dyn 3rd pass", "ax dyn 4th pass", "ax t1"]
    rows = [(i, "P1", "S1", f"u{i}", d) for i, d in enumerate(desc)]                 # 6 series, unordered
    rows += [(6, "P2", "S2", "v0", "ax dyn pre")]                                      # 1 series -> excluded
    rows += [(7, "P3", "S3", "w0", "pre"), (8, "P3", "S3", "w0", "pre"), (9, "P3", "S3", "w1", "1st pass")]
    rows += [(10, "P4", "S4", "x0", "pre"), (11, "P4", "S4", "x1", "1st pass")]        # no score
    meta = _metadata(rows)
    clinical = _clinical({"P1": 25.0, "P2": 10.0, "P3": 18.0, "P4": np.nan})
    studies, audit = select_studies_legacy(meta, clinical, label_definition("oncotype_binary_18"), 5)
    assert list(studies[STUDY_ID]) == ["S1", "S3"]
    assert studies.loc[0, SERIES_UIDS] == ("u0", "u1", "u2", "u3", "u4")              # row order, not phase order
    assert studies.loc[1, SERIES_UIDS] == ("w0", "w0", "w1")                           # legacy list(x)[:5] keeps duplicates
    assert list(studies[LABEL]) == [1, 0]
    reasons = dict(zip(audit[STUDY_ID], audit["reason"]))
    assert reasons == {"S1": "included", "S2": "fewer_than_2_unique_series", "S3": "included", "S4": "missing_oncotype_score"}
    assert "truncated_6_to_5" in audit.set_index(STUDY_ID).at["S1", "notes"]
    assert "duplicate_series_rows_selected" in audit.set_index(STUDY_ID).at["S3", "notes"]


def test_general_profile_orders_phases_and_reports_missing_pre_post():
    desc = ["ax dyn 2nd pass", "ax dyn pre", "ax dyn 1st pass", "ax dyn 3rd pass", "ax dyn 4th pass", "ax t1"]
    rows = [(i, "P1", "S1", f"u{i}", d) for i, d in enumerate(desc)]
    rows += [(10 + i, "P2", "S2", f"v{i}", d) for i, d in enumerate(["1st pass", "2nd pass", "3rd pass", "4th pass", "5th pass"])]
    rows += [(20 + i, "P3", "S3", f"w{i}", d) for i, d in enumerate(["", "", ""])]
    rows += [(30 + i, "P4", "S4", f"x{i}", d) for i, d in enumerate(["pre", "1st pass", "1st pass", "2nd pass", "3rd pass", "4th pass"])]
    rows += [(40 + i, "P5", "S5", f"y{i}", d) for i, d in enumerate(["pre", "1st pass", "2nd pass"])]
    clinical = _clinical({p: 20.0 for p in ("P1", "P2", "P3", "P4", "P5")})
    studies, audit = select_studies_general(_metadata(rows), clinical, label_definition("oncotype_binary_18"), 5)
    assert list(studies[STUDY_ID]) == ["S1"]
    assert studies.loc[0, SERIES_UIDS] == ("u1", "u2", "u0", "u3", "u4")
    reasons = dict(zip(audit[STUDY_ID], audit["reason"]))
    assert reasons["S2"] == "missing_pre_contrast"
    assert reasons["S3"] == "phase_unparseable"
    assert reasons["S4"] == "ambiguous_phase_index"
    assert reasons["S5"] == "insufficient_phases"
    # row-order variant rejects the unordered study instead of silently re-ordering
    studies_m, audit_m = select_studies_general(_metadata(rows), clinical, label_definition("oncotype_binary_18"), 5, "metadata")
    assert studies_m.empty and dict(zip(audit_m[STUDY_ID], audit_m["reason"]))["S1"] == "non_contiguous_or_unordered_phases"


def test_general_requires_phase_information():
    meta = _metadata([(0, "P1", "S1", "u0", "pre")]).drop(columns="series_description")
    with pytest.raises(CohortError, match="Series Description"):
        select_studies_general(meta, _clinical({"P1": 3.0}), label_definition("oncotype_binary_18"), 5)


@pytest.mark.parametrize("text,index", [("ax dyn pre", 0), ("Ax 3D Dyn 1st pass", 1), ("ax dyn 4th pass", 4),
                                        ("post2", 2), ("Ph3/ax 3d dyn", 3), ("ax t1 tse", None), ("", None)])
def test_parse_phase_index(text, index):
    assert parse_phase_index(text) == index


def test_cohort_source_matches_synthetic_and_legacy_row_order(tables):
    cohort = DukeCohortSource().load(_cfg(tables))
    expected = tables["studies"]
    assert list(cohort.frame.columns[:4]) == [PATIENT_ID, STUDY_ID, LABEL, SERIES_UIDS]
    assert cohort.frame[[PATIENT_ID, STUDY_ID, SERIES_UIDS]].equals(expected[[PATIENT_ID, STUDY_ID, SERIES_UIDS]])
    assert cohort.frame[LABEL].tolist() == expected[LABEL].tolist()
    assert "oncotype_score" not in cohort.frame.columns
    assert cohort.frame[STUDY_ID].is_monotonic_increasing                         # legacy groupby order
    assert set(cohort.audit["reason"]) >= {"included", "missing_oncotype_score", "fewer_than_2_unique_series"}
    assert cohort.frame[PATIENT_ID].duplicated().any()                          # patients with two studies
    assert any("row order" in n for n in cohort.notes)
    assert any("duke_oncotype_allowed_predictors" in n for n in cohort.notes)
    records = study_records(cohort.frame, read_metadata_digest(tables["metadata_file"]))
    assert all(len(r.series) == 5 and r.series[0].num_slices == 16 for r in records)


def test_three_class_labels(tables):
    cohort = DukeCohortSource().load(_cfg(tables, label_scheme="oncotype_3class_18_31"))
    assert cohort.frame[LABEL].tolist() == tables["studies"]["label_3class"].tolist()
    assert cohort.label.num_classes == 3


def test_general_profile_on_synthetic(tables):
    cohort = DukeCohortSource().load(_cfg(tables, profile="general", phase_order="explicit"))
    legacy = DukeCohortSource().load(_cfg(tables))
    assert set(cohort.frame[STUDY_ID]) < set(legacy.frame[STUDY_ID])
    assert "phase_unparseable" in set(cohort.audit["reason"])
    for uids in cohort.frame[SERIES_UIDS]:
        assert len(uids) == 5
    meta = read_metadata_digest(tables["metadata_file"]).set_index("series_uid")
    for uids in cohort.frame[SERIES_UIDS]:
        assert [parse_phase_index(meta.at[u, "series_description"]) for u in uids] == [0, 1, 2, 3, 4]
    with pytest.raises(CohortError):
        DukeCohortSource().load(_cfg(tables, phase_order="explicit"))  # legacy profile keeps row order only


def test_clinical_reader_tokens_and_age(tables):
    clinical = read_clinical_workbook(tables["clinical_features_file"])
    assert len(clinical) == 40 and clinical["patient_id"].iloc[0] == "Breast_MRI_001"
    assert clinical["tumor_laterality"].isna().any()                          # "NC" tokens -> NaN
    assert (clinical["age_at_diagnosis_years"] > 0).all()


def test_metadata_csv_equivalent(tables, tmp_path):
    xlsx = read_metadata_digest(tables["metadata_file"])
    csv = tmp_path / "meta.csv"
    tables["series"].to_csv(csv, index=False)
    from_csv = read_metadata_digest(csv)
    assert from_csv[[PATIENT_ID, STUDY_ID, "series_uid", "series_description"]].equals(
        xlsx[[PATIENT_ID, STUDY_ID, "series_uid", "series_description"]])


# --------------------------------------------------------------------------- tabular groups
def test_default_clinical_groups_and_sensitive_opt_in(tables):
    cohort = DukeCohortSource().load(_cfg(tables))
    assert cohort.continuous_columns == ["age_at_diagnosis_years"]
    assert "race_ethnicity" not in cohort.categorical_columns and "race_ethnicity" not in cohort.frame
    assert len(cohort.categorical_columns) == 7
    sensitive = DukeCohortSource().load(_cfg(tables, include_sensitive=True))
    assert sensitive.categorical_columns[-1] == "race_ethnicity"


def test_image_only_branch_skips_tabular(tables):
    cohort = DukeCohortSource().load(_cfg(tables, branches=("image",), feature_groups=("clinical", "kinetic")))
    assert cohort.continuous_columns == [] and cohort.categorical_columns == []


def test_imaging_features_merge_duplicates_and_missing(tables, tmp_path):
    studies = tables["studies"][[PATIENT_ID, STUDY_ID]]
    imaging = pd.read_excel(tables["imaging_features_file"], sheet_name="Imaging Features")
    merged, columns, report = merge_imaging_features(studies, tables["imaging_features_file"], ["kinetic", "morphology"])
    assert all(c.startswith(("kinetic_", "morphology_")) for c in columns) and len(merged) == len(studies)
    dup = pd.concat([imaging, imaging.iloc[[0]]], ignore_index=True)
    with pytest.raises(ImagingFeatureError, match="duplicate"):
        merge_imaging_features(studies, "x.csv", ["kinetic"], table=dup)
    missing = imaging[imaging["Patient ID"] != studies[PATIENT_ID].iloc[0]]
    with pytest.raises(ImagingFeatureError, match="missing for"):
        merge_imaging_features(studies, "x.csv", ["kinetic"], table=missing)
    merged, columns, _ = merge_imaging_features(studies, "x.csv", ["kinetic"], table=missing, require_complete_match=False)
    assert (~merged["imaging_features_matched"]).sum() == (studies[PATIENT_ID] == studies[PATIENT_ID].iloc[0]).sum()
    renamed = imaging.rename(columns={"Patient ID": "PID"})
    merged, _, _ = merge_imaging_features(studies, "x.csv", ["heterogeneity"], table=renamed, source_patient_id_column="PID")
    assert merged["imaging_features_matched"].all()
    with pytest.raises(ImagingFeatureError, match="missing 'Patient ID'"):
        merge_imaging_features(studies, "x.csv", ["kinetic"], table=renamed)


def test_allow_missing_imaging_features_via_cohort(tables, tmp_path):
    imaging = pd.read_excel(tables["imaging_features_file"], sheet_name="Imaging Features")
    first = tables["studies"][PATIENT_ID].iloc[0]
    path = tmp_path / "imaging.csv"
    imaging[imaging["Patient ID"] != first].to_csv(path, index=False)
    groups = ("clinical", "kinetic")
    with pytest.raises(ImagingFeatureError):
        DukeCohortSource().load(_cfg(tables, imaging_features_file=str(path), feature_groups=groups))
    cohort = DukeCohortSource().load(_cfg(tables, imaging_features_file=str(path), feature_groups=groups,
                                          allow_missing_imaging_features=True))
    assert cohort.audit.loc[cohort.audit[PATIENT_ID] == first, "imaging_features_matched"].eq(False).all()


# --------------------------------------------------------------------------- policy
def test_policy_rejects_forbidden_and_unknown():
    policy = load_policy()
    assert policy.version == "1.0"
    policy.check(["age_at_diagnosis_years", "kinetic_Peak_SER_tumor", "density_consensus"])
    for column in ("oncotype_score", "label", "ER", "PR", "Nottingham grade", "staging_tumor_size", "my_label_copy",
                   "recurrence_event", "neoadjuvant_chemotherapy", "kinetic_oncotype_proxy"):
        with pytest.raises(PredictorPolicyError):
            policy.check([column])
    with pytest.raises(PredictorPolicyError, match="not in the allowed list"):
        policy.check(["some_new_column"])
    with pytest.raises(PredictorPolicyError, match="include_sensitive"):
        policy.check(["race_ethnicity"])
    policy.check(["race_ethnicity"], include_sensitive=True)


def test_policy_override_file_and_data_leakage(tables, tmp_path):
    import json
    from mribreast.data.predictor_policy import DEFAULT_POLICY_FILE
    from importlib import resources

    payload = json.loads(resources.files("mribreast.data").joinpath(DEFAULT_POLICY_FILE).read_text(encoding="utf-8"))
    payload["version"] = "1.1"
    payload["prediction_times"]["mri"]["clinical"]["columns"].remove("menopause")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PredictorPolicyError, match="menopause"):
        DukeCohortSource().load(_cfg(tables, allowed_predictors_file=str(path)))
    frame = pd.DataFrame({LABEL: [0, 1, 0, 1, 1, 0], "kinetic_x": [0, 1, 0, 1, 1, 0], "kinetic_y": [3, 1, 2, 5, 1, 0]})
    with pytest.raises(PredictorPolicyError, match="kinetic_x == label"):
        check_label_leakage(frame, ["kinetic_x", "kinetic_y"])
    check_label_leakage(frame, ["kinetic_y"])


# --------------------------------------------------------------------------- density
def test_density_adapter_real_schema(tables, tmp_path):
    raw = pd.read_excel(tables["density_file"], sheet_name="Assessments")
    assert list(raw.columns) == ["Subject_ID", "Radiologist A", "Radiologist B", "Radiologist C", DENSITY_LEGEND]
    assessments = read_density_assessments(tables["density_file"])
    assert set(assessments.columns) == {PATIENT_ID, "Radiologist A", "Radiologist B", "Radiologist C"}
    features = density_features(pd.DataFrame({PATIENT_ID: ["p1", "p2", "p3"], "Radiologist A": ["a", "b", "a"],
                                              "Radiologist B": ["a", "c", np.nan], "Radiologist C": ["b", "d", np.nan]}))
    assert features["density_consensus"].tolist() == ["a", "c", "a"]           # majority; median when no majority
    assert features["density_ordinal_mean"].round(4).tolist() == [1.3333, 3.0, 1.0]
    studies = pd.DataFrame({PATIENT_ID: [assessments[PATIENT_ID].iloc[0], "nobody"], STUDY_ID: ["s1", "s2"]})
    merged, cont, cat, report = merge_density(studies, tables["density_file"])
    assert cont == ["density_ordinal_mean"] and cat == ["density_consensus"]
    assert report["studies_with_density"] == 1 and np.isnan(merged.loc[1, "density_ordinal_mean"])
    bad = raw.copy()
    bad.loc[0, "Radiologist A"] = "e"
    bad.to_excel(tmp_path / "bad.xlsx", sheet_name="Assessments", index=False)
    with pytest.raises(ValueError, match="unknown density"):
        read_density_assessments(tmp_path / "bad.xlsx")


def test_density_group_off_by_default_and_opt_in(tables):
    cohort = DukeCohortSource().load(_cfg(tables))
    assert not any(c.startswith("density_") for c in cohort.frame.columns)
    cohort = DukeCohortSource().load(_cfg(tables, feature_groups=("clinical", "density")))
    assert "density_consensus" in cohort.categorical_columns and "density_ordinal_mean" in cohort.continuous_columns
    assert any("density coverage" in n for n in cohort.notes)
