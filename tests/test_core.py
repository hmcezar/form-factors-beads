from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import form_factors_beads as ff
from form_factors_beads import core
from form_factors_beads.core import _prepare_output


def write_mapping(path: Path, text: str | None = None) -> Path:
    path.write_text(
        text
        or """[ to ]
martini

[ martini ]
B0 B1

[ atoms ]
1 C1 B0 B1
2 H1 B0
3 O1 B1
"""
    )
    return path


def write_bonded_pdb(path: Path) -> Path:
    path.write_text(
        """ATOM      1  C1  MOL A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  H1  MOL A   1       1.090   0.000   0.000  1.00  0.00           H
ATOM      3  O1  MOL A   1      -1.230   0.000   0.000  1.00  0.00           O
CONECT    1    2    3
CONECT    2    1
CONECT    3    1
END
"""
    )
    return path


def test_cgbuilder_mapping_supports_shared_atoms_and_comments(tmp_path):
    mapping = ff.read_cgbuilder_mapping(write_mapping(tmp_path / "shared.map"))
    assert mapping.bead_order == ("B0", "B1")
    assert mapping.atom_beads[1] == ("B0", "B1")
    assert mapping.bead_atoms == {"B0": (1, 2), "B1": (1, 3)}


@pytest.mark.parametrize(
    "text, message",
    [
        ("[ martini ]\nB0\n", "expected"),
        (
            "[ martini ]\nB0\n[ atoms ]\n1 C1 missing\n",
            "unknown bead",
        ),
        (
            "[ martini ]\nB0\n[ atoms ]\n2 C1 B0\n",
            "contiguous",
        ),
    ],
)
def test_mapping_validation_is_explicit(tmp_path, text, message):
    path = write_mapping(tmp_path / "bad.map", text)
    with pytest.raises(ValueError, match=message):
        ff.read_cgbuilder_mapping(path)


def test_shared_atom_f0_is_conserved():
    q = np.array([0.0, 0.1])
    full = ff.sans_factors(["C", "H", "O"], np.ones(3), q)[0].sum()
    left = ff.sans_factors(["C", "H"], np.array([0.5, 1.0]), q)[0].sum()
    right = ff.sans_factors(["C", "O"], np.array([0.5, 1.0]), q)[0].sum()
    assert left + right == pytest.approx(full)


def test_neutron_isotopes_are_supported():
    assert ff.neutron_length("D") > 0
    assert ff.neutron_length("H") < 0
    assert ff.neutron_length("13C") != pytest.approx(ff.neutron_length("C"))
    assert ff.neutron_length("C-13") == pytest.approx(ff.neutron_length("13C"))


def test_constrained_fit_has_exact_f0_and_prioritizes_low_q():
    q = np.arange(0.0, 2.0, 0.01)
    values = np.exp(-(q**2)) + 0.03 * np.sin(8 * q)
    weighted, metrics = ff.fit_polynomial(q, values, 6, 0.75, 5.0, True)
    unweighted, _ = ff.fit_polynomial(q, values, 6, 0.75, 1.0, True)
    assert weighted[0] == values[0]
    assert metrics["f0_polynomial"] == values[0]
    low = q <= 0.75
    weighted_error = np.polynomial.polynomial.polyval(q, weighted) - values
    unweighted_error = np.polynomial.polynomial.polyval(q, unweighted) - values
    assert np.mean(weighted_error[low] ** 2) < np.mean(unweighted_error[low] ** 2)


def test_q_grid_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="start at 0"):
        ff.q_grid({"minimum": 0.1, "maximum": 2.0, "step": 0.01})
    with pytest.raises(ValueError, match="too short"):
        ff.q_grid({"minimum": 0.0, "maximum": 0.05, "step": 0.01})


def test_sign_restoration_preserves_f0_and_explicit_crossing():
    q = np.array([0.0, 0.5, 1.0])
    amplitude = np.array([-2.0, 1.0, 2.0])
    signed, crossover, note = ff.restore_amplitude_sign(amplitude, q, 0.5, 0.1)
    assert signed[0] == pytest.approx(-2.0)
    assert signed[1] == pytest.approx(0.0)
    assert signed[2] > 0.0
    assert crossover == 0.5
    assert note is None


def test_positive_contrast_does_not_invent_a_sign_crossing():
    q = np.array([0.0, 0.5, 1.0])
    signed, crossover, note = ff.restore_amplitude_sign(
        np.array([2.0, 1.0, 0.5]), q, None, 0.1
    )
    np.testing.assert_array_equal(signed, [2.0, 1.0, 0.5])
    assert crossover is None
    assert note is None


def test_xray_factor_requires_a_displaced_volume():
    with pytest.raises(ValueError, match="No solvent-displaced SAXS volume"):
        ff.xray_factors(
            ["P"],
            np.ones(1),
            np.array([0.0, 0.1]),
            density=0.334,
            volumes={"C": 16.44},
        )


def test_isotope_override_cannot_change_the_element():
    with pytest.raises(ValueError, match="changes atom 1 element"):
        core.isotope_labels(["C"], {1: "N-15"})


def test_graph_identity_distinguishes_connectivity():
    mapping = ff.MappingData(
        bead_order=("linear", "branched"),
        bead_atoms={"linear": (1, 2, 3, 4), "branched": (5, 6, 7, 8)},
        atom_names={i: f"C{i}" for i in range(1, 9)},
        atom_beads={i: (("linear",) if i < 5 else ("branched",)) for i in range(1, 9)},
        atom_count=8,
    )
    bonds = {
        (1, 2): "1",
        (2, 3): "1",
        (3, 4): "1",
        (5, 6): "1",
        (5, 7): "1",
        (5, 8): "1",
    }
    groups = ff.group_equivalent_beads(
        mapping,
        ["C"] * 8,
        ["C"] * 8,
        bonds,
        {"solvent_electron_density": 0.334, "electron_density_overrides": {}},
        {"force_groups": {}, "force_split": []},
    )
    assert list(groups.values()) == [["linear"], ["branched"]]


def test_saxs_contrast_splits_otherwise_identical_graphs():
    mapping = ff.MappingData(
        bead_order=("B0", "B1"),
        bead_atoms={"B0": (1, 2), "B1": (3, 4)},
        atom_names={1: "C1", 2: "H1", 3: "C2", 4: "H2"},
        atom_beads={1: ("B0",), 2: ("B0",), 3: ("B1",), 4: ("B1",)},
        atom_count=4,
    )
    bonds = {(1, 2): "1", (3, 4): "1"}
    common = {
        "solvent_electron_density": 0.334,
        "electron_density_overrides": {},
    }
    identity = {"force_groups": {}, "force_split": []}
    merged = ff.group_equivalent_beads(
        mapping, ["C", "H", "C", "H"], ["C", "H", "C", "H"], bonds, common, identity
    )
    assert list(merged.values()) == [["B0", "B1"]]

    contrasted = copy.deepcopy(common)
    contrasted["electron_density_overrides"] = {"B1": 0.214}
    split = ff.group_equivalent_beads(
        mapping,
        ["C", "H", "C", "H"],
        ["C", "H", "C", "H"],
        bonds,
        contrasted,
        identity,
    )
    assert list(split.values()) == [["B0"], ["B1"]]


def test_config_deep_merge_keeps_unspecified_defaults(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("fit:\n  low_q_weight: 9.0\n")
    config, base = ff.load_config(path)
    assert config["fit"]["low_q_weight"] == 9.0
    assert config["fit"]["degree"] == 6
    assert config["saxs"]["solvent_electron_density"] == pytest.approx(0.334)
    assert base == tmp_path


def test_atomic_volume_data_is_packaged():
    volumes, provenance = ff.load_atomic_volumes({})
    assert volumes["H"] == pytest.approx(5.15)
    assert volumes["C"] == pytest.approx(16.44)
    assert "Fraser" in provenance["O"]["source"]


def test_output_directory_is_non_clobbering(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    existing = output / "existing.txt"
    existing.write_text("keep")
    with pytest.raises(FileExistsError):
        _prepare_output(output, force=False)
    assert existing.read_text() == "keep"


def test_minimal_end_to_end_calculation_and_outputs(tmp_path):
    topology = write_bonded_pdb(tmp_path / "molecule.pdb")
    mapping = write_mapping(tmp_path / "molecule.map")
    config = copy.deepcopy(ff.DEFAULT_CONFIG)
    config["q"] = {"minimum": 0.0, "maximum": 0.2, "step": 0.02}
    config["fit"]["degree"] = 3
    result = ff.calculate_molecule(
        {
            "name": "mini",
            "topology": topology,
            "trajectory": None,
            "mapping": mapping,
        },
        config,
        tmp_path,
    )
    assert result.frames == 1
    assert result.occurrences == 1
    assert set(result.methods) == {"saxs", "sans"}
    assert all(values[0] == pytest.approx(result.methods["saxs"]["metrics"][group]["f0_true"])
               for group, values in result.methods["saxs"]["coefficients"].items())

    output = tmp_path / "results"
    _prepare_output(output, force=False)
    ff.write_result(output, result, config, force=False)
    assert (output / "mini_saxs_parameters.inp").is_file()
    assert (output / "mini_sans_parameters.inp").is_file()
    assert (output / "mini_sans_scattering_lengths.inp").is_file()
    assert (output / "mini_form_factor_fits.png").stat().st_size > 0
    assert (output / "mini_report.yaml").is_file()
