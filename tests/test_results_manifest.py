from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from cliffordnet_cl20.results import ManifestError, render_results_table, validate_manifest


@pytest.fixture
def manifest():
    return yaml.safe_load(Path("results/manifest.yaml").read_text())


def test_manifest_is_valid_and_renders_separate_families(manifest):
    validate_manifest(manifest, root=Path.cwd())
    rendered = render_results_table(manifest)
    assert "Official CliffordNet" in rendered
    assert "Cl(2,0) Roll & Shift" in rendered
    assert "Geometric Shuffle" in rendered
    assert "82.46" in rendered


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row.update(status="claimed"), "status"),
    (lambda row: row.update(evidence=[]), "evidence"),
    (lambda row: row.update(evidence=["/home/private/run.json"]), "relative"),
    (lambda row: row.update(parameters=123), "parameter"),
])
def test_invalid_rows_are_rejected(manifest, mutation, match):
    broken = deepcopy(manifest)
    mutation(broken["results"][0])
    with pytest.raises(ManifestError, match=match):
        validate_manifest(broken, root=Path.cwd())


def test_duplicate_identity_is_rejected(manifest):
    broken = deepcopy(manifest)
    broken["results"].append(deepcopy(broken["results"][0]))
    with pytest.raises(ManifestError, match="duplicate"):
        validate_manifest(broken, root=Path.cwd())


def test_reproduced_requires_complete_run_record(manifest, tmp_path):
    broken = deepcopy(manifest)
    row = broken["results"][0]
    row["status"] = "reproduced"
    row["evidence"] = ["run.json"]
    (tmp_path / "run.json").write_text('{"status":"incomplete"}')
    with pytest.raises(ManifestError, match="complete"):
        validate_manifest(broken, root=tmp_path)
