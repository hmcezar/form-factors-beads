from __future__ import annotations

import copy
from argparse import Namespace
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


def test_equation9_terms_reconstruct_the_legacy_saxs_factor():
    q = np.array([0.0, 0.05, 0.1, 0.2])
    positions = np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]])
    weights = np.array([1.0, 0.5])
    volumes = {"C": 16.44, "O": 9.13}
    atomic, solvent = ff.xray_component_factors(["C", "O"], weights, q, volumes)
    terms = ff.debye_component_terms(positions, q, atomic, solvent, box=None)
    density = 0.334
    legacy = ff.debye_amplitude(
        positions, q, atomic - density * solvent, box=None
    )
    np.testing.assert_allclose(ff.equation9_radicand(terms, density), legacy**2)
    assert terms["atomic"][0] == pytest.approx(atomic[0].sum() ** 2)
    assert terms["solvent"][0] == pytest.approx(solvent[0].sum() ** 2)
    assert terms["mixed"][0] == pytest.approx(
        2.0 * atomic[0].sum() * solvent[0].sum()
    )


def test_equation9_radicand_clamps_roundoff_but_rejects_material_negatives():
    tiny = {
        "atomic": np.array([1.0]),
        "solvent": np.array([0.0]),
        "mixed": np.array([1.0 + 5.0e-13]),
    }
    assert ff.equation9_radicand(tiny, 1.0)[0] == 0.0
    material = {**tiny, "mixed": np.array([2.0])}
    with pytest.raises(ValueError, match="radicand is negative"):
        ff.equation9_radicand(material, 1.0)


def test_exchangeable_hydrogens_use_bonds_and_overrides():
    elements = ["O", "H", "C", "H", "N", "H"]
    bonds = {(1, 2): "1", (3, 4): "1", (5, 6): "1"}
    assert ff.detect_exchangeable_hydrogens(elements, bonds) == (2, 6)
    assert ff.detect_exchangeable_hydrogens(elements, bonds, include=[4], exclude=[2]) == (
        4,
        6,
    )
    with pytest.raises(ValueError, match="not hydrogen"):
        ff.detect_exchangeable_hydrogens(elements, bonds, include=[3])


def test_lcpo_exact_fallback_and_hydrogen_assignments():
    database, provenance = ff.load_lcpo_database()
    assert len(database) > 1000
    assert provenance["probe_radius_A"] == pytest.approx(1.4)
    with pytest.warns(UserWarning, match="automatic LCPO element-fallback.*C=1"):
        assignments, summary = ff.assign_lcpo_parameters(
            ["CA", "C1", "H1"],
            ["ALA", "MOL", "MOL"],
            ["C", "C", "H"],
            {"allow_element_fallback": True},
        )
    assert assignments[0].source == "plumed_exact"
    assert assignments[1].source == "element_fallback"
    assert assignments[2].values == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert summary["assignment_counts"] == {
        "plumed_exact": 1,
        "element_fallback": 1,
        "hydrogen_excluded": 1,
    }


def test_lcpo_element_fallbacks_cover_supported_elements_and_warn():
    elements = ["C", "N", "O", "P", "S", "F", "Mg"]
    with pytest.warns(UserWarning, match=r"C=1.*F=1.*Mg=1.*N=1.*O=1.*P=1.*S=1"):
        assignments, summary = ff.assign_lcpo_parameters(
            [f"X{index}" for index in range(len(elements))],
            ["MOL"] * len(elements),
            elements,
            {"allow_element_fallback": True},
        )
    assert [assignment.source for assignment in assignments] == [
        "element_fallback"
    ] * len(elements)
    assert assignments[-2].values == pytest.approx(
        (1.47, 0.68563, -0.18680, -0.00135573, 0.00023743)
    )
    assert assignments[-1].values == pytest.approx(
        (1.18, 0.49392, -0.16038, -0.00015512, 0.00016453)
    )
    assert summary["assignment_counts"] == {"element_fallback": len(elements)}


def test_lcpo_isolated_area_and_pbc_safe_virtual_center():
    assignment = ff.LCPOAssignment(
        (1.7, 0.56482, -0.19608, -0.0010219, 0.0002658),
        "test",
        "C",
    )
    areas, isolated = ff.lcpo_areas(np.zeros((1, 3)), [assignment], box=None)
    assert areas[0] == pytest.approx(isolated[0])
    box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    center = ff.virtual_center_positions(
        np.array([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        ["B"],
        {"B": [1, 2]},
        {"B": [0.5, 0.5]},
        box,
    )
    assert center[0, 0] == pytest.approx(10.0)


def test_cg_radius_is_derived_from_shared_atom_adjusted_volume():
    carbon = ff.LCPOAssignment(
        (1.7, 0.56482, -0.19608, -0.0010219, 0.0002658),
        "test",
        "C",
    )
    common = (
        ["B"],
        {"B": [1, 2]},
        {1: ["B"], 2: ["B"]},
        ["C", "C"],
        {"C": 16.44},
        [carbon, carbon],
    )
    derived, details = ff.derive_cg_lcpo_parameters(
        *common,
        {"probe_radius_A": 1.4},
    )
    expected_radius = (3.0 * 2.0 * 16.44 / (4.0 * np.pi)) ** (1.0 / 3.0)
    assert derived["B"].values[0] == pytest.approx(expected_radius)
    assert details["B"]["volume_equivalent_radius_A"] == pytest.approx(expected_radius)
    assert details["B"]["split_displaced_volume_A3"] == pytest.approx(32.88)


def test_contextual_parameter_parser_rejects_bad_contract(tmp_path):
    path = tmp_path / "bad.inp"
    path.write_text(
        "FORMAT_VERSION=1\nMETHOD=SANS\nTERM=ATOMIC\n"
        "Q_MIN=0\nQ_MAX=0.2\nQ_STEP=0.1\nPARAMETERS2=1\n"
    )
    with pytest.raises(ValueError, match="incompatible"):
        ff.parse_contextual_parameter_file(path)
    path.write_text(
        "FORMAT_VERSION=1\nMETHOD=SAXS\nTERM=ATOMIC\n"
        "Q_MIN=0\nQ_MAX=0.2\nQ_STEP=0.1\nPARAMETERS2=1\n"
    )
    with pytest.raises(ValueError, match="contiguous"):
        ff.parse_contextual_parameter_file(path)


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


def test_documented_example_config_loads():
    path = Path(__file__).resolve().parents[1] / "examples" / "example.yaml"
    config, base = ff.load_config(path)
    assert base == path.parent
    assert config == ff.DEFAULT_CONFIG


@pytest.mark.parametrize("option", ["topology", "trajectory", "mapping", "name"])
def test_project_mode_rejects_cli_molecule_arguments(option):
    values = {"topology": None, "trajectory": None, "mapping": None, "name": None}
    values[option] = "command-line-value"
    args = Namespace(**values)
    config = {"molecules": [{"name": "from-yaml", "mapping": "molecule.map"}]}
    with pytest.raises(ValueError, match="CLI molecule arguments cannot be combined"):
        core._molecule_specs(args, config)


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
    for method in result.methods.values():
        for magnitude in method["positive_magnitude_curves"].values():
            assert np.all(magnitude >= 0.0)

    output = tmp_path / "results"
    _prepare_output(output, force=False)
    ff.write_result(output, result, config, force=False)
    assert (output / "mini_saxs_parameters.inp").is_file()
    assert (output / "mini_sans_parameters.inp").is_file()
    assert (output / "mini_sans_scattering_lengths.inp").is_file()
    assert (output / "mini_form_factor_fits.png").stat().st_size > 0
    assert (output / "mini_report.yaml").is_file()


def test_minimal_equation9_decomposition_and_outputs(tmp_path):
    topology = write_bonded_pdb(tmp_path / "molecule.pdb")
    mapping = write_mapping(tmp_path / "molecule.map")
    config = copy.deepcopy(ff.DEFAULT_CONFIG)
    config["q"] = {"minimum": 0.0, "maximum": 0.2, "step": 0.02}
    config["fit"]["degree"] = 3
    config["decomposition"]["enabled"] = True
    result = ff.calculate_molecule(
        {"name": "mini", "topology": topology, "trajectory": None, "mapping": mapping},
        config,
        tmp_path,
    )
    assert result.decomposition is not None
    assert set(result.decomposition["methods"]["saxs"]["terms"]) == {
        "atomic",
        "solvent",
        "mixed",
    }
    assert set(result.decomposition["methods"]["sans"]["terms"]) == {
        "atomic_h",
        "mixed_h",
        "atomic_d",
        "mixed_d",
        "solvent",
    }
    assert result.decomposition["mapping"]["bead_atoms"]["B0"] == (1, 2)

    output = tmp_path / "decomposed"
    _prepare_output(output, force=False)
    ff.write_result(output, result, config, force=False)
    atomic_file = output / "mini_saxs_atomic_parameters.inp"
    assert atomic_file.is_file()
    text = atomic_file.read_text()
    assert "FORMAT_VERSION=1" in text
    assert "METHOD=SAXS" in text
    assert "TERM=ATOMIC" in text
    assert text.count("PARAMETERS") == 2
    parsed = ff.parse_contextual_parameter_file(atomic_file)
    assert parsed["method"] == "SAXS"
    assert parsed["term"] == "ATOMIC"
    assert len(parsed["parameters"]) == 2
    assert (output / "mini_sans_d_mixed_parameters.inp").is_file()
    assert (output / "mini_hybrid_mapping.inp").is_file()
    assert (output / "mini_hybrid_lcpo_parameters.inp").is_file()
    assert (output / "mini_cg_lcpo_parameters.inp").is_file()
    assert (output / "mini_sans_exchangeable_beads.inp").is_file()
    assert (output / "mini_saxs_equation9_inline.inp").is_file()
    assert (output / "mini_sans_equation9_inline.inp").is_file()
    assert (output / "mini_equation9_terms.csv").is_file()
    for method, terms in {
        "saxs": ("atomic", "solvent", "mixed"),
        "sans": ("atomic_h", "mixed_h", "atomic_d", "mixed_d", "solvent"),
    }.items():
        for term in terms:
            assert (output / f"mini_{method}_{term}_fit.png").stat().st_size > 0
    assert (output / "mini_lcpo_exposure_samples.csv").is_file()
    assert (output / "mini_lcpo_exposure_correlation.png").stat().st_size > 0
    comparison = result.decomposition["lcpo"]["exposure_comparison"]
    assert comparison["sample_count"] == 2
    assert sum(comparison["confusion_matrix"].values()) == 2

    project = tmp_path / "project"
    _prepare_output(project, force=False)
    core.write_project_outputs(
        project,
        [result],
        [{"name": "mini", "count": 2}],
        config,
        force=False,
    )
    project_atomic = (project / "project_saxs_atomic_parameters.inp").read_text()
    assert project_atomic.count("PARAMETERS") == 4
    project_mapping = (project / "project_hybrid_mapping.inp").read_text()
    assert "BEAD_ATOMS1=1,2" in project_mapping
    assert "BEAD_ATOMS3=4,5" in project_mapping
    project_hybrid = (project / "project_hybrid_lcpo_parameters.inp").read_text()
    assert project_hybrid.count("LCPO_PARAMETERS") == 6


@pytest.mark.parametrize(
    ("removed_key", "value"),
    [
        ("cg_radius_model", "volume"),
        ("cg_radius_scale", 1.0),
        ("cg_radius_fit", {}),
    ],
)
def test_removed_lcpo_settings_are_rejected(tmp_path, removed_key, value):
    topology = write_bonded_pdb(tmp_path / "molecule.pdb")
    mapping = write_mapping(tmp_path / "molecule.map")
    config = copy.deepcopy(ff.DEFAULT_CONFIG)
    config["q"] = {"minimum": 0.0, "maximum": 0.2, "step": 0.02}
    config["fit"]["degree"] = 3
    config["decomposition"]["enabled"] = True
    config["decomposition"]["lcpo"][removed_key] = value
    with pytest.raises(ValueError, match="unsupported decomposition.lcpo settings"):
        ff.calculate_molecule(
            {"name": "mini", "topology": topology, "trajectory": None, "mapping": mapping},
            config,
            tmp_path,
        )
