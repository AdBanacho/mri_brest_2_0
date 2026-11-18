"""mribreast.prepare.nbia with a fake client and a local HTTP server (no internet)."""

from __future__ import annotations

import io
import json
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from mribreast.prepare import nbia


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


class FakeClient:
    def __init__(self, series: dict[str, list[dict]], zips: dict[str, bytes]):
        self.series, self.zips, self.calls = series, zips, []

    def get_series(self, collection, study_uid):
        self.calls.append(("series", collection, study_uid))
        if study_uid == "boom":
            raise ConnectionError("listing failed")
        return self.series.get(study_uid, [])

    def get_series_zip(self, series_uid):
        self.calls.append(("zip", series_uid))
        return self.zips[series_uid]


def test_prepare_imports_no_optional_dependency():
    import subprocess

    code = ("import sys, mribreast.prepare.nbia, mribreast.prepare.dicom_to_nifti, mribreast.prepare.mama_mia, "
            "mribreast.prepare.cli, mribreast.legacy.importer, mribreast.legacy.checkpoints; "
            "print(sorted(m for m in ('pydicom','SimpleITK','synapseclient','requests','torch','dicom2nifti') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_metadata_and_cohort_selection(tmp_path):
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"Patient ID": ["P2", "P1", "P1", "P3"], "Study Instance UID": ["S2", "S1", "S1", "S3"],
                  "Series Instance UID": ["a", "b", "c", "d"]}).to_csv(meta, index=False)
    clinical = tmp_path / "clinical.csv"
    pd.DataFrame({"patient_id": ["P1", "P2", "P3"], "oncotype_score": [12, "NA", 30]}).to_csv(clinical, index=False)
    metadata = nbia.read_nbia_metadata(meta)
    outcome = nbia.read_outcome_patients(clinical)
    assert outcome == {"P1", "P3"}
    assert nbia.select_study_uids(metadata, "all") == ["S2", "S1", "S3"]
    assert nbia.select_study_uids(metadata, "outcome", outcome) == ["S1", "S3"]
    with pytest.raises(ValueError):
        nbia.select_study_uids(metadata, "outcome")


def test_download_resume_and_failures(tmp_path):
    series = {"S1": [{"SeriesInstanceUID": "1.2.1", "ImageCount": 2, "Modality": "MR", "PatientID": "P1"},
                     {"SeriesInstanceUID": "1.2.2", "ImageCount": 3}],
              "S2": [{"SeriesInstanceUID": "1.2.3"}]}
    zips = {"1.2.1": _zip({"1-1.dcm": b"a", "1-2.dcm": b"b"}), "1.2.2": _zip({"x.dcm": b"a"}),
            "1.2.3": _zip({"../evil.dcm": b"x"})}
    client = FakeClient(series, zips)
    manifest = nbia.download_studies(client, ["S1", "S2", "boom", "S9"], tmp_path / "dl")
    status = dict(zip(manifest["series_uid"].where(manifest["series_uid"] != "", manifest["study_uid"]),
                      manifest["status"]))
    assert status == {"1.2.1": "downloaded", "1.2.2": "failed", "1.2.3": "failed", "boom": "failed_listing",
                      "S9": "no_series"}
    assert (tmp_path / "dl" / "1.2.1" / "1-2.dcm").read_bytes() == b"b"
    assert nbia.is_complete(tmp_path / "dl" / "1.2.1")
    assert not (tmp_path / "dl" / "1.2.2").exists() and not (tmp_path / "evil.dcm").exists()
    assert not [p for p in (tmp_path / "dl").iterdir() if p.name.startswith(".")]  # temp dirs cleaned
    again = nbia.download_studies(client, ["S1"], tmp_path / "dl")
    assert again.loc[again.series_uid == "1.2.1", "status"].item() == "skipped_existing"
    assert list(manifest.columns) == list(nbia.MANIFEST_COLUMNS)
    path = nbia.write_manifest(manifest, tmp_path / "dl" / "download_manifest.csv")
    assert len(pd.read_csv(path)) == len(manifest)


class _Handler(BaseHTTPRequestHandler):
    requests_seen: list = []

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        _Handler.requests_seen.append((url.path, query))
        if url.path.endswith("/getSeries"):
            body = json.dumps([{"SeriesInstanceUID": "9.9", "StudyInstanceUID": query["StudyInstanceUID"],
                                "ImageCount": 1}]).encode()
        elif url.path.endswith("/getImage"):
            body = _zip({"one.dcm": b"dicom"})
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


def test_rest_client_against_local_server(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = nbia.RestNbiaClient(base_url=f"http://127.0.0.1:{server.server_address[1]}/nbia-api/services/v1",
                                     timeout=10)
        manifest = nbia.download_studies(client, ["1.2.840.1"], tmp_path / "dl")
    finally:
        server.shutdown()
        server.server_close()
    assert manifest["status"].tolist() == ["downloaded"]
    assert (tmp_path / "dl" / "9.9" / "one.dcm").read_bytes() == b"dicom"
    paths = [p for p, _ in _Handler.requests_seen]
    assert paths == ["/nbia-api/services/v1/getSeries", "/nbia-api/services/v1/getImage"]
    assert _Handler.requests_seen[0][1] == {"Collection": nbia.DUKE_COLLECTION, "StudyInstanceUID": "1.2.840.1",
                                            "format": "json"}


def test_injected_http_get_is_used():
    seen = []

    def fake_get(url, params, timeout):
        seen.append((url, dict(params), timeout))
        return b"[]"

    assert nbia.RestNbiaClient(base_url="https://example.invalid/v1", http_get=fake_get, timeout=3).get_series("C", "S") == []
    assert seen == [("https://example.invalid/v1/getSeries", {"Collection": "C", "StudyInstanceUID": "S",
                                                               "format": "json"}, 3)]
