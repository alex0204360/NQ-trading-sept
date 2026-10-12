"""Optional completed-run checks, plus a synthetic publication integrity guard."""
import json
import os
from pathlib import Path
import pytest
from tools.mnq_oracle_report import publish
from tools.verify_mnq_oracle import verify


def test_publication_rejects_changed_manifest_file(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    (source/"results.json").write_text("{}")
    (source/"manifest.json").write_text(json.dumps({"complete": True, "files_sha256": {"results.json": "0"*64}}))
    with pytest.raises(ValueError, match="checksum"):
        publish(source, tmp_path/"public")


@pytest.mark.skipif(not os.environ.get("MNQ_PRIVATE_RUN"), reason="Requires a completed private MNQ run, never NQ holdout data")
def test_completed_run_accounting_and_publication_allowlist(tmp_path):
    source = Path(os.environ["MNQ_PRIVATE_RUN"])
    result = verify(source)
    assert result["verified"] and result["scenario_count"] == 24
    publish(source, tmp_path/"public")
    files = {file.name for file in (tmp_path/"public").iterdir()}
    assert files == {"results.json", "development.json", "selection.json", "audit.json", "environment.json", "private_artifact_hashes.json", "dashboard.html"}
    page = (tmp_path/"public"/"dashboard.html").read_text(encoding="utf-8")
    assert "No trade" in page and "future information" in page
    assert "NONCAUSAL" in page
    for forbidden in ("<script src=", "fetch(", "requestAnimationFrame", "entry_price,exit_price", "p_short,p_wait,p_long"):
        assert forbidden not in page
