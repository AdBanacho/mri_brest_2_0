"""MAMA-MIA import/mapping job with a fake Synapse client (no network, no credentials)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from mribreast.prepare import cli, mama_mia


class FakeSynapse:
    def __init__(self, table: pd.DataFrame):
        self.table, self.calls = table, []

    def get(self, entity, downloadLocation=None):  # noqa: N803
        self.calls.append((entity, downloadLocation))
        path = f"{downloadLocation}/mama_mia_dataset_info.csv"
        self.table.to_csv(path, index=False)
        return SimpleNamespace(path=path)


@pytest.fixture()
def inputs(tmp_path):
    duke = pd.DataFrame({"Patient ID": ["P1", "P1", "P2", "P3", "P3"],
                         "Study Instance UID": ["S1", "S1", "S2", "S3", "S3"],
                         "Series Instance UID": ["u1", "u2", "u3", "u4", "u5"]})
    duke.to_csv(tmp_path / "meta.csv", index=False)
    pd.DataFrame({"patient_id": ["P1", "P2", "P3"], "oncotype_score": [10, None, 40]}).to_csv(
        tmp_path / "clinical.csv", index=False)
    mama = pd.DataFrame({"patient_id": ["DUKE_001", "DUKE_002", "ISPY_1"], "tcia_series_uid": [" u2", "u3", "zz"],
                         "dataset": ["DUKE", "DUKE", "ISPY1"]})
    return tmp_path, mama


def test_download_uses_injected_client(tmp_path, inputs):
    _, mama = inputs
    client = FakeSynapse(mama)
    path = mama_mia.download_resource(client, tmp_path / "res")
    assert client.calls == [(mama_mia.LEGACY_ENTITY_ID, str(tmp_path / "res"))]
    table = mama_mia.read_mama_mia_table(path)
    assert table["series_uid"].tolist() == ["u2", "u3", "zz"]


def test_mapping_reproduces_legacy_intersection(inputs):
    tmp_path, mama = inputs
    mama_path = tmp_path / "mama.csv"
    mama.to_csv(mama_path, index=False)
    from mribreast.prepare.nbia import read_nbia_metadata, read_outcome_patients

    duke = read_nbia_metadata(tmp_path / "meta.csv")
    outcome = read_outcome_patients(tmp_path / "clinical.csv")
    table = mama_mia.read_mama_mia_table(mama_path)
    mapping, summary = mama_mia.map_to_duke(table, duke, outcome, keep_columns=["dataset"])
    # legacy: len(set(duke series of patients with numeric score) & set(mama uids))
    legacy = len(set(duke.loc[duke.patient_id.isin(outcome), "series_uid"]) & set(table["series_uid"]))
    assert summary["legacy_intersection"] == legacy == 1
    assert summary == {"duke_series": 5, "mama_mia_series": 3, "mama_mia_series_in_duke": 2,
                       "mama_mia_series_not_in_duke": 1, "duke_patients_with_mama_mia": 2,
                       "legacy_intersection": 1, "outcome_patients_with_mama_mia": 1}
    assert mapping.loc[mapping.series_uid == "u2", "dataset"].item() == "DUKE"
    assert list(mapping.columns[:5]) == list(mama_mia.MAPPING_COLUMNS)


def test_duplicates_and_missing_column_rejected(tmp_path):
    pd.DataFrame({"tcia_series_uid": ["a", "a"]}).to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        mama_mia.read_mama_mia_table(tmp_path / "dup.csv")
    pd.DataFrame({"uid": ["a"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="tcia_series_uid"):
        mama_mia.read_mama_mia_table(tmp_path / "bad.csv")


def test_cli_map(inputs):
    tmp_path, mama = inputs
    mama.to_csv(tmp_path / "mama.csv", index=False)
    rc = cli.main(["mama-mia-map", "--table", str(tmp_path / "mama.csv"), "--metadata_file", str(tmp_path / "meta.csv"),
                   "--clinical_features_file", str(tmp_path / "clinical.csv"), "--output_dir", str(tmp_path / "out")])
    assert rc == 0
    summary = json.loads((tmp_path / "out" / "mama_mia_summary.json").read_text(encoding="utf-8"))
    assert summary["legacy_intersection"] == 1
    assert len(pd.read_csv(tmp_path / "out" / "mama_mia_duke_mapping.csv")) == 5


def test_cli_nbia_dry_run(inputs, capsys):
    tmp_path, _ = inputs
    rc = cli.main(["nbia-download", "--metadata_file", str(tmp_path / "meta.csv"), "--clinical_features_file",
                   str(tmp_path / "clinical.csv"), "--cohort", "outcome", "--dest", str(tmp_path / "dl"), "--dry_run"])
    assert rc == 0 and "2 studies (cohort=outcome)" in capsys.readouterr().out
    assert not (tmp_path / "dl").exists()
    assert cli.main(["nbia-download", "--metadata_file", str(tmp_path / "meta.csv"), "--dest", "x"]) == 2
