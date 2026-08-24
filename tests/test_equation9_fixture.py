from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

import form_factors_beads as ff

FIXTURE = Path(__file__).parent / "data" / "equation9_fixture"


@pytest.mark.integration
def test_equation9_cross_repository_fixture(tmp_path):
    manifest = yaml.safe_load((FIXTURE / "manifest.yaml").read_text())
    for relative, expected_hash in manifest["files"].items():
        assert hashlib.sha256((FIXTURE / relative).read_bytes()).hexdigest() == expected_hash

    config, _ = ff.load_config(FIXTURE / "config.yaml")
    result = ff.calculate_molecule(
        {
            "name": "equation9_fixture",
            "topology": FIXTURE / "structure.pdb",
            "trajectory": None,
            "mapping": FIXTURE / "mapping.map",
        },
        config,
        Path("/"),
    )
    ff.write_decomposition_outputs(tmp_path, result, force=False)

    expected_files = sorted((FIXTURE / "expected").iterdir())
    assert expected_files
    for expected in expected_files:
        generated = tmp_path / expected.name
        assert generated.is_file(), expected.name
        assert generated.read_bytes() == expected.read_bytes(), expected.name

    atomic_h = ff.parse_contextual_parameter_file(
        tmp_path / "equation9_fixture_sans_h_atomic_parameters.inp"
    )["parameters"]
    atomic_d = ff.parse_contextual_parameter_file(
        tmp_path / "equation9_fixture_sans_d_atomic_parameters.inp"
    )["parameters"]
    assert atomic_h[0] != atomic_d[0]
    assert atomic_h[1] == atomic_d[1]

    reference = yaml.safe_load((FIXTURE / "reference.yaml").read_text())["expected"]
    assert result.decomposition is not None
    assert list(result.decomposition["exchange"]["exchangeable_hydrogens"]) == reference[
        "exchangeable_hydrogens"
    ]
    assert [
        bead
        for bead, eligible in result.decomposition["exchange"]["eligible_beads"].items()
        if eligible
    ] == reference["exchangeable_beads"]
    comparison = result.decomposition["lcpo"]["exposure_comparison"]
    assert comparison["mismatch_fraction"] == pytest.approx(
        reference["exposure_comparison"]["mismatch_fraction"]
    )
    assert comparison["confusion_matrix"] == reference["exposure_comparison"][
        "confusion_matrix"
    ]
    for bead, values in reference["exposure_comparison"]["per_bead"].items():
        assert comparison["per_bead"][bead]["hybrid_mean_fraction"] == pytest.approx(
            values["hybrid_mean_fraction"]
        )
        assert comparison["per_bead"][bead]["cg_mean_fraction"] == pytest.approx(
            values["cg_mean_fraction"]
        )
    for method, threshold in reference["reconstruction_max_abs_error"].items():
        largest = max(
            values["max_abs_error"]
            for values in result.decomposition["methods"][method]["reconstruction"].values()
        )
        assert largest <= threshold
