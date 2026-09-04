from __future__ import annotations

import copy
import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

import form_factors_beads as ff

DATA_DIR = Path(__file__).with_name("data")
BEADS = tuple(f"B{i}" for i in range(15))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curve_reference() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with (DATA_DIR / "tx9_get_ff_beads_saxs_curves.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    q = np.asarray([float(row["q_A^-1"]) for row in rows])
    curves = {
        bead: np.asarray([float(row[bead]) for row in rows]) for bead in BEADS
    }
    return q, curves


def load_coefficient_reference() -> dict[str, np.ndarray]:
    with (DATA_DIR / "tx9_known_good_saxs_coefficients.csv").open() as handle:
        return {
            row["group"]: np.asarray([float(row[f"c{i}"]) for i in range(7)])
            for row in csv.DictReader(handle)
        }


@pytest.fixture(scope="module")
def tx9_regression():
    metadata = yaml.safe_load((DATA_DIR / "tx9_reference_metadata.yaml").read_text())
    paths = {
        key: DATA_DIR / details["path"] for key, details in metadata["inputs"].items()
    }
    for key, path in paths.items():
        assert sha256(path) == metadata["inputs"][key]["sha256"], (
            f"TX9 {key} does not match the reference dataset: {path}"
        )
    assert sha256(DATA_DIR / "tx9_get_ff_beads_saxs_curves.csv") == metadata[
        "provenance"
    ]["curves_sha256"]
    assert sha256(DATA_DIR / "tx9_known_good_saxs_coefficients.csv") == metadata[
        "provenance"
    ]["coefficients_sha256"]

    config = copy.deepcopy(ff.DEFAULT_CONFIG)
    config["trajectory"]["stride"] = metadata["calculation"]["stride"]
    solvated_density = metadata["calculation"]["solvated_electron_density"]
    config["saxs"]["electron_density_overrides"] = {
        bead: solvated_density for bead in metadata["calculation"]["solvated_beads"]
    }
    config["sign"]["crossovers"] = {
        "saxs": metadata["calculation"]["sign_crossovers_A^-1"]
    }
    config["sign"]["widths"] = {
        "saxs": metadata["calculation"]["sign_widths_A^-1"]
    }
    result = ff.calculate_molecule(
        {
            "name": "tx9",
            "topology": paths["topology"],
            "trajectory": paths["trajectory"],
            "mapping": paths["mapping"],
        },
        config,
        DATA_DIR,
    )
    reference_q, reference_curves = load_curve_reference()
    return result, metadata, reference_q, reference_curves


@pytest.mark.integration
def test_tx9_reference_dataset_and_sampling(tx9_regression):
    result, metadata, reference_q, _ = tx9_regression
    assert result.frames == metadata["calculation"]["sampled_frames"]
    assert result.bead_order == list(BEADS)
    np.testing.assert_allclose(result.q, reference_q, rtol=0.0, atol=1e-14)


@pytest.mark.integration
@pytest.mark.parametrize("bead", BEADS)
def test_tx9_saxs_bead_curves_match_get_ff_beads(tx9_regression, bead):
    result, metadata, _, reference_curves = tx9_regression
    actual = result.methods["saxs"]["bead_curves"][bead]
    expected = reference_curves[bead]
    difference = actual - expected
    tolerances = metadata["tolerances"]
    assert abs(difference[0]) <= tolerances["raw_curve_f0_abs"]
    assert np.max(np.abs(difference)) <= tolerances["raw_curve_max_abs"]
    assert np.sqrt(np.mean(difference**2)) <= tolerances["raw_curve_rmse"]


@pytest.mark.integration
def test_tx9_low_q_polynomials_match_known_good_scattering(tx9_regression):
    result, metadata, _, _ = tx9_regression
    reference = load_coefficient_reference()
    calculated = result.methods["saxs"]["coefficients"]
    canonical = {
        "B0": calculated["B0"],
        "B1": calculated["B1"],
        "B2/B4": (calculated["B2"] + calculated["B4"]) / 2.0,
        "B3": calculated["B3"],
        "B5": calculated["B5"],
        "B6/B7": calculated["B6"],
        "B8/B9/B10/B11/B12/B13": calculated["B8"],
        "B14": calculated["B14"],
    }
    low_q = result.q <= 0.75
    tolerances = metadata["tolerances"]
    for group, coefficients in canonical.items():
        expected = np.polynomial.polynomial.polyval(result.q, reference[group])
        actual = np.polynomial.polynomial.polyval(result.q, coefficients)
        difference = actual[low_q] - expected[low_q]
        assert np.max(np.abs(difference)) <= tolerances["polynomial_low_q_max_abs"], group
        assert np.sqrt(np.mean(difference**2)) <= tolerances[
            "polynomial_low_q_rmse"
        ], group
