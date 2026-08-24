from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

import form_factors_beads as ff

FIXTURE = Path(__file__).parent / "data" / "equation9_fixture"
FIT_RTOL = 1.0e-9
FIT_ATOL = 1.0e-11


def _assert_parameter_file_close(generated: Path, expected: Path) -> None:
    actual = ff.parse_contextual_parameter_file(generated)
    reference = ff.parse_contextual_parameter_file(expected)
    assert {key: value for key, value in actual.items() if key != "parameters"} == {
        key: value for key, value in reference.items() if key != "parameters"
    }
    np.testing.assert_allclose(
        actual["parameters"],
        reference["parameters"],
        rtol=FIT_RTOL,
        atol=FIT_ATOL,
    )


def _parse_inline_file(path: Path) -> dict[str, tuple[float, ...]]:
    assignments = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split("=", 1)
        assignments[key] = tuple(float(item) for item in value.split(","))
    return assignments


def _assert_terms_csv_close(generated: Path, expected: Path) -> None:
    with generated.open(newline="") as handle:
        actual = list(csv.DictReader(handle))
    with expected.open(newline="") as handle:
        reference = list(csv.DictReader(handle))
    assert len(actual) == len(reference)
    labels = ("method", "term", "group")
    values = ("q_A^-1", "value", "fit", "residual")
    for actual_row, reference_row in zip(actual, reference, strict=True):
        assert tuple(actual_row[key] for key in labels) == tuple(
            reference_row[key] for key in labels
        )
        np.testing.assert_allclose(
            [float(actual_row[key]) for key in values],
            [float(reference_row[key]) for key in values],
            rtol=1e-12,
            atol=1e-12,
        )


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
        text = expected.read_text()
        if "METHOD=" in text and "TERM=" in text:
            _assert_parameter_file_close(generated, expected)
        elif expected.name.endswith("_equation9_inline.inp"):
            actual = _parse_inline_file(generated)
            reference = _parse_inline_file(expected)
            assert actual.keys() == reference.keys()
            for key in actual:
                np.testing.assert_allclose(
                    actual[key],
                    reference[key],
                    rtol=FIT_RTOL,
                    atol=FIT_ATOL,
                )
        elif expected.name.endswith("_equation9_terms.csv"):
            _assert_terms_csv_close(generated, expected)
        else:
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
    assert comparison["accessible_fraction_cutoff"] == pytest.approx(
        reference["exposure_comparison"]["accessible_fraction_cutoff"]
    )
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
