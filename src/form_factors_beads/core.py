#!/usr/bin/env python3
"""Generate SAXS and SANS coarse-grained bead form factors.

The module accepts CGBuilder mappings, identifies topologically equivalent
beads, averages Debye form-factor magnitudes over a trajectory, performs a
low-q weighted polynomial fit, and writes PLUMED parameter fragments.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import math
import re
import sys
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import numpy as np
import periodictable as pt
import yaml
from matplotlib.lines import Line2D
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.guesser.default_guesser import DefaultGuesser
from MDAnalysis.lib.distances import distance_array
from networkx.algorithms import isomorphism
from periodictable import cromermann

from .decomposition import (
    LCPOAssignment,
    aggregate_hybrid_accessible_fractions,
    assign_lcpo_parameters,
    debye_component_terms,
    derive_cg_lcpo_parameters,
    detect_exchangeable_hydrogens,
    equation9_radicand,
    lcpo_areas,
    normalized_mass_weights,
    virtual_center_positions,
)

PACKAGE_DIR = Path(__file__).resolve().parent
ATOMIC_VOLUME_FILE = PACKAGE_DIR / "data" / "atomic_volumes.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "q": {"minimum": 0.0, "maximum": 2.0, "step": 0.01},
    "trajectory": {
        "selection": "all",
        "start": 0,
        "stop": None,
        "stride": 1,
        "guess_bonds": False,
    },
    "fit": {
        "degree": 6,
        "enforce_f0": True,
        "low_q_max": 0.75,
        "low_q_weight": 5.0,
    },
    "saxs": {
        "enabled": True,
        "solvent_electron_density": 0.334,
        "electron_density_overrides": {},
        "excluded_volume_overrides": {},
    },
    "sans": {
        "polynomial": True,
        "one_parameter": True,
        "isotope_overrides": {},
    },
    "identity": {"force_groups": {}, "force_split": []},
    "sign": {"default_width": 0.03, "crossovers": {}, "widths": {}},
    "plot": {"show_positive_magnitude": True},
    "element_overrides": {},
    "decomposition": {
        "enabled": False,
        "representations": ["hybrid", "cg"],
        "lcpo": {
            "allow_element_fallback": True,
            "atom_overrides": {},
            "element_overrides": {},
            "bead_overrides": {},
            "probe_radius_A": 1.4,
        },
        "diagnostics": {"accessible_fraction_cutoff": 0.2},
        "exchange": {"include": [], "exclude": []},
    },
}


@dataclass(frozen=True)
class MappingData:
    bead_order: tuple[str, ...]
    bead_atoms: Mapping[str, tuple[int, ...]]
    atom_names: Mapping[int, str]
    atom_beads: Mapping[int, tuple[str, ...]]
    atom_count: int


@dataclass
class MoleculeResult:
    name: str
    bead_order: list[str]
    groups: OrderedDict[str, list[str]]
    frames: int
    occurrences: int
    q: np.ndarray
    methods: dict[str, dict[str, Any]]
    sans_one_parameter: dict[str, float]
    warnings: list[str]
    provenance: dict[str, Any]
    decomposition: dict[str, Any] | None = None


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | None) -> tuple[dict[str, Any], Path]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    base = Path.cwd()
    if path is not None:
        path = path.resolve()
        with path.open() as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("The YAML root must be a mapping")
        _deep_update(config, loaded)
        base = path.parent
    return config, base


def _strip_comment(line: str) -> str:
    return re.split(r"[;#]", line, maxsplit=1)[0].strip()


def read_cgbuilder_mapping(path: str | Path) -> MappingData:
    """Read a CGBuilder map, including atoms shared by multiple beads."""
    sections: dict[str, list[list[str]]] = {}
    section: str | None = None
    with Path(path).open() as handle:
        for raw in handle:
            line = _strip_comment(raw)
            if not line:
                continue
            match = re.fullmatch(r"\[\s*([^]]+?)\s*\]", line)
            if match:
                section = match.group(1).strip().lower()
                sections.setdefault(section, [])
                continue
            if section is not None:
                sections[section].append(line.split())

    if "martini" not in sections or "atoms" not in sections:
        raise ValueError(f"{path}: expected [ martini ] and [ atoms ] sections")
    bead_order = tuple(token for row in sections["martini"] for token in row)
    if not bead_order or len(bead_order) != len(set(bead_order)):
        raise ValueError(f"{path}: bead names must be present and unique")

    bead_atoms: dict[str, list[int]] = {bead: [] for bead in bead_order}
    atom_names: dict[int, str] = {}
    atom_beads: dict[int, tuple[str, ...]] = {}
    for row in sections["atoms"]:
        if len(row) < 3:
            raise ValueError(f"{path}: malformed [ atoms ] row: {' '.join(row)}")
        try:
            index = int(row[0])
        except ValueError as exc:
            raise ValueError(f"{path}: invalid atom index {row[0]!r}") from exc
        if index < 1 or index in atom_names:
            raise ValueError(f"{path}: duplicate or invalid atom index {index}")
        memberships = tuple(dict.fromkeys(row[2:]))
        unknown = [bead for bead in memberships if bead not in bead_atoms]
        if unknown:
            raise ValueError(f"{path}: unknown bead(s) {unknown} for atom {index}")
        atom_names[index] = row[1]
        atom_beads[index] = memberships
        for bead in memberships:
            bead_atoms[bead].append(index)

    empty = [bead for bead, atoms in bead_atoms.items() if not atoms]
    if empty:
        raise ValueError(f"{path}: beads without atoms: {', '.join(empty)}")
    indices = sorted(atom_names)
    if indices != list(range(1, max(indices) + 1)):
        raise ValueError(f"{path}: atom indices must be contiguous and 1-based")
    return MappingData(
        bead_order=bead_order,
        bead_atoms={key: tuple(value) for key, value in bead_atoms.items()},
        atom_names=atom_names,
        atom_beads=atom_beads,
        atom_count=max(indices),
    )


def _normalize_element(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.upper() in {"D", "T"}:
        return value.upper()
    match = re.fullmatch(r"([A-Za-z]{1,2})", value)
    if not match:
        return ""
    symbol = match.group(1).capitalize()
    try:
        pt.elements.symbol(symbol)
    except ValueError:
        return ""
    return symbol


def resolve_elements(
    universe: mda.Universe,
    selected_indices: np.ndarray,
    atom_count: int,
    overrides: Mapping[Any, Any],
) -> list[str]:
    first = universe.atoms[selected_indices[:atom_count]]
    values: list[str]
    try:
        values = [_normalize_element(str(value)) for value in first.elements]
    except NoDataError:
        guessed = DefaultGuesser(universe).guess_types(first.names)
        values = [_normalize_element(str(value)) for value in guessed]
    normalized_overrides = {int(key): str(value) for key, value in overrides.items()}
    for atom_index, element in normalized_overrides.items():
        if not 1 <= atom_index <= atom_count:
            raise ValueError(f"element override index {atom_index} is outside the mapping")
        values[atom_index - 1] = _normalize_element(element)
    unresolved = [i + 1 for i, value in enumerate(values) if not value]
    if unresolved:
        details = ", ".join(f"{i}:{first.names[i-1]}" for i in unresolved)
        raise ValueError(f"Could not resolve elements for mapped atoms {details}")
    return values


def _parse_isotope(label: str) -> tuple[str, int | None]:
    label = str(label).strip()
    if label == "D":
        return "H", 2
    if label == "T":
        return "H", 3
    match = re.fullmatch(r"(\d+)([A-Za-z]{1,2})", label)
    if not match:
        match = re.fullmatch(r"([A-Za-z]{1,2})-?(\d+)", label)
        if match:
            return _normalize_element(match.group(1)), int(match.group(2))
    elif match:
        return _normalize_element(match.group(2)), int(match.group(1))
    element = _normalize_element(label)
    if not element:
        raise ValueError(f"Unknown element/isotope label {label!r}")
    return element, None


def isotope_labels(elements: list[str], overrides: Mapping[Any, Any]) -> list[str]:
    labels = list(elements)
    for key, value in overrides.items():
        index = int(key)
        if not 1 <= index <= len(labels):
            raise ValueError(f"isotope override index {index} is outside the mapping")
        element, mass = _parse_isotope(str(value))
        if element != ("H" if elements[index - 1] in {"D", "T"} else elements[index - 1]):
            raise ValueError(
                f"isotope override {value!r} changes atom {index} element "
                f"from {elements[index - 1]} to {element}"
            )
        labels[index - 1] = f"{mass}{element}" if mass is not None else element
    return labels


def neutron_length(label: str) -> float:
    element, mass = _parse_isotope(label)
    atom = pt.elements.symbol(element)
    if mass is not None:
        atom = atom[mass]
    value = atom.neutron.b_c
    if value is None:
        raise ValueError(f"No coherent neutron scattering length for {label}")
    return float(value)


def base_element(label: str) -> str:
    return _parse_isotope(label)[0]


def load_atomic_volumes(overrides: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    with ATOMIC_VOLUME_FILE.open() as handle:
        data = yaml.safe_load(handle)
    entries = data["volumes_A3"]
    volumes = {symbol: float(entry["value"]) for symbol, entry in entries.items()}
    provenance = copy.deepcopy(entries)
    for symbol, value in overrides.items():
        normalized = _normalize_element(str(symbol))
        if not normalized:
            raise ValueError(f"Invalid excluded-volume element {symbol!r}")
        volumes[normalized] = float(value)
        provenance[normalized] = {
            "value": float(value),
            "source": "user configuration",
            "convention": "user_override",
        }
    return volumes, provenance


def q_grid(settings: Mapping[str, Any]) -> np.ndarray:
    q_min = float(settings["minimum"])
    q_max = float(settings["maximum"])
    step = float(settings["step"])
    if q_min != 0.0 or q_max <= q_min or step <= 0.0:
        raise ValueError("q grid must start at 0 and have positive maximum and step")
    count = int(math.floor((q_max - q_min) / step + 1e-12))
    if count < 7:
        raise ValueError("q grid is too short for a sixth-order fit")
    return q_min + step * np.arange(count, dtype=float)


def valid_box(dimensions: Any) -> np.ndarray | None:
    if dimensions is None:
        return None
    dimensions = np.asarray(dimensions, dtype=float)
    if dimensions.size < 3 or np.any(dimensions[:3] <= 0.0):
        return None
    return dimensions


def _selected_occurrences(
    universe: mda.Universe, selection: str, atom_count: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    selected = universe.select_atoms(selection)
    if len(selected) == 0:
        raise ValueError(f"trajectory selection {selection!r} contains no atoms")
    if len(selected) % atom_count:
        raise ValueError(
            f"selection contains {len(selected)} atoms, which is not a multiple of "
            f"the {atom_count}-atom mapping template"
        )
    indices = np.asarray(selected.indices, dtype=int)
    occurrences = [indices[i : i + atom_count] for i in range(0, len(indices), atom_count)]
    return indices, occurrences


def _bond_table(
    universe: mda.Universe,
    occurrence: np.ndarray,
    guess_bonds: bool,
) -> dict[tuple[int, int], str]:
    try:
        bonds = universe.bonds
    except NoDataError:
        if not guess_bonds:
            raise ValueError(
                "The topology has no bonds. Supply a bonded topology or explicitly "
                "set trajectory.guess_bonds: true."
            ) from None
        warnings.warn(
            "Guessing bonds from the first trajectory frame", RuntimeWarning, stacklevel=2
        )
        universe.atoms.guess_bonds()
        bonds = universe.bonds
    reverse = {global_index: local + 1 for local, global_index in enumerate(occurrence)}
    table: dict[tuple[int, int], str] = {}
    for bond in bonds:
        a, b = (int(value) for value in bond.indices)
        if a not in reverse or b not in reverse:
            continue
        local_a, local_b = reverse[a], reverse[b]
        order = getattr(bond, "order", None)
        rendered = "1" if order in {None, "", 0} else str(order)
        table[tuple(sorted((local_a, local_b)))] = rendered
    if not table:
        raise ValueError("No bonds were found within the first mapped molecule")
    return table


def _bead_graph(
    bead: str,
    mapping: MappingData,
    elements: list[str],
    isotopes: list[str],
    bonds: Mapping[tuple[int, int], str],
    density: float,
) -> nx.Graph:
    graph = nx.Graph(density=f"{density:.12g}")
    members = set(mapping.bead_atoms[bead])
    for atom in sorted(members):
        fraction = 1.0 / len(mapping.atom_beads[atom])
        label = f"atom|{elements[atom-1]}|{isotopes[atom-1]}|{fraction:.12g}"
        graph.add_node(("atom", atom), label=label)
    for (a, b), order in bonds.items():
        if a in members and b in members:
            graph.add_edge(("atom", a), ("atom", b), order=order)
        elif (a in members) != (b in members):
            inside, outside = (a, b) if a in members else (b, a)
            boundary = ("boundary", inside, outside)
            label = f"boundary|{elements[outside-1]}|{isotopes[outside-1]}"
            graph.add_node(boundary, label=label)
            graph.add_edge(("atom", inside), boundary, order=order)
    return graph


def _graphs_equal(left: nx.Graph, right: nx.Graph) -> bool:
    if left.graph.get("density") != right.graph.get("density"):
        return False
    return nx.is_isomorphic(
        left,
        right,
        node_match=isomorphism.categorical_node_match("label", ""),
        edge_match=isomorphism.categorical_edge_match("order", "1"),
    )


def group_equivalent_beads(
    mapping: MappingData,
    elements: list[str],
    isotopes: list[str],
    bonds: Mapping[tuple[int, int], str],
    sax_settings: Mapping[str, Any],
    identity_settings: Mapping[str, Any],
) -> OrderedDict[str, list[str]]:
    default_density = float(sax_settings["solvent_electron_density"])
    raw_overrides = {
        str(k): float(v) for k, v in sax_settings["electron_density_overrides"].items()
    }
    overrides = {
        key: value for key, value in raw_overrides.items() if not key.startswith("group:")
    }
    unknown_beads = set(overrides) - set(mapping.bead_order)
    if unknown_beads:
        raise ValueError(
            "saxs.electron_density_overrides contains unknown beads: "
            + ", ".join(sorted(unknown_beads))
        )
    graphs: dict[str, nx.Graph] = {}
    hashes: dict[str, str] = {}
    for bead in mapping.bead_order:
        graph = _bead_graph(
            bead,
            mapping,
            elements,
            isotopes,
            bonds,
            overrides.get(bead, default_density),
        )
        graphs[bead] = graph
        hashes[bead] = (
            nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", edge_attr="order")
            + "|"
            + graph.graph["density"]
        )
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for bead in mapping.bead_order:
        match = next(
            (
                name
                for name, members in groups.items()
                if hashes[bead] == hashes[members[0]]
                and _graphs_equal(graphs[bead], graphs[members[0]])
            ),
            None,
        )
        groups.setdefault(match or bead, []).append(bead)

    # A group override uses the automatic identity group's representative name,
    # e.g. ``group:B8``.  Resolve it after the first graph pass, then regroup so
    # the contrast becomes part of the final identity.
    group_overrides = {
        key.removeprefix("group:"): value
        for key, value in raw_overrides.items()
        if key.startswith("group:")
    }
    if group_overrides:
        unknown_groups = set(group_overrides) - set(groups)
        if unknown_groups:
            raise ValueError(
                "SAXS density overrides contain unknown automatic groups: "
                + ", ".join(sorted(unknown_groups))
            )
        for group, density in group_overrides.items():
            for bead in groups[group]:
                overrides[bead] = density
        graphs.clear()
        hashes.clear()
        groups.clear()
        for bead in mapping.bead_order:
            graph = _bead_graph(
                bead,
                mapping,
                elements,
                isotopes,
                bonds,
                overrides.get(bead, default_density),
            )
            graphs[bead] = graph
            hashes[bead] = (
                nx.weisfeiler_lehman_graph_hash(
                    graph, node_attr="label", edge_attr="order"
                )
                + "|"
                + graph.graph["density"]
            )
        for bead in mapping.bead_order:
            match = next(
                (
                    name
                    for name, members in groups.items()
                    if hashes[bead] == hashes[members[0]]
                    and _graphs_equal(graphs[bead], graphs[members[0]])
                ),
                None,
            )
            groups.setdefault(match or bead, []).append(bead)

    force_split = {str(value) for value in identity_settings.get("force_split", [])}
    for bead in force_split:
        if bead not in mapping.bead_order:
            raise ValueError(f"identity.force_split contains unknown bead {bead}")
        for group, members in list(groups.items()):
            if bead in members and len(members) > 1:
                members.remove(bead)
                groups[bead] = [bead]
                if not members:
                    del groups[group]
                break

    force_groups = identity_settings.get("force_groups", {})
    for requested_name, requested_members in force_groups.items():
        members = [str(value) for value in requested_members]
        unknown = set(members) - set(mapping.bead_order)
        if unknown:
            raise ValueError(f"forced group {requested_name} has unknown beads {sorted(unknown)}")
        for group in list(groups):
            groups[group] = [bead for bead in groups[group] if bead not in members]
            if not groups[group]:
                del groups[group]
        groups[str(requested_name)] = members

    ordered = OrderedDict()
    for bead in mapping.bead_order:
        for name, members in groups.items():
            if bead in members and name not in ordered:
                ordered[name] = members
    return ordered


def _validate_exchange_grouping(
    groups: Mapping[str, list[str]],
    mapping: MappingData,
    exchangeable_hydrogens: tuple[int, ...],
) -> None:
    """Reject forced groups whose members have distinct exchanged end states."""
    exchanged = set(exchangeable_hydrogens)
    signatures = {
        bead: tuple(
            sorted(
                1.0 / len(mapping.atom_beads[index])
                for index in mapping.bead_atoms[bead]
                if index in exchanged
            )
        )
        for bead in mapping.bead_order
    }
    for group, members in groups.items():
        if len({signatures[bead] for bead in members}) > 1:
            raise ValueError(
                f"identity group {group} combines beads with inconsistent "
                "exchangeable-hydrogen end states; split the group or revise "
                "decomposition.exchange overrides"
            )


def xray_component_factors(
    labels: list[str],
    weights: np.ndarray,
    q: np.ndarray,
    volumes: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return density-independent SAXS atomic and displaced-volume factors."""
    atomic = np.empty((q.size, len(labels)), dtype=float)
    solvent = np.empty((q.size, len(labels)), dtype=float)
    missing = sorted(
        {base_element(label) for label in labels if base_element(label) not in volumes}
    )
    if missing:
        raise ValueError(
            "No solvent-displaced SAXS volume for "
            + ", ".join(missing)
            + "; provide saxs.excluded_volume_overrides"
        )
    for index, label in enumerate(labels):
        element = base_element(label)
        free_atom = np.asarray(cromermann.fxrayatq(element, q), dtype=float)
        volume = volumes[element]
        displaced_volume = volume * np.exp(
            -(q**2) * volume ** (2.0 / 3.0) / (4.0 * np.pi)
        )
        atomic[:, index] = weights[index] * free_atom
        solvent[:, index] = weights[index] * displaced_volume
    return atomic, solvent


def xray_factors(
    labels: list[str],
    weights: np.ndarray,
    q: np.ndarray,
    density: float,
    volumes: Mapping[str, float],
) -> np.ndarray:
    atomic, solvent = xray_component_factors(labels, weights, q, volumes)
    return atomic - density * solvent


def sans_factors(labels: list[str], weights: np.ndarray, q: np.ndarray) -> np.ndarray:
    values = weights * np.asarray([neutron_length(label) for label in labels])
    return np.broadcast_to(values, (q.size, values.size)).copy()


def sans_component_factors(
    labels: list[str],
    weights: np.ndarray,
    q: np.ndarray,
    volumes: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return SANS atomic scattering lengths and displaced-volume factors."""
    atomic = sans_factors(labels, weights, q)
    missing = sorted(
        {base_element(label) for label in labels if base_element(label) not in volumes}
    )
    if missing:
        raise ValueError(
            "No solvent-displaced SANS volume for "
            + ", ".join(missing)
            + "; provide saxs.excluded_volume_overrides"
        )
    solvent = np.empty_like(atomic)
    for index, label in enumerate(labels):
        volume = volumes[base_element(label)]
        solvent[:, index] = weights[index] * volume * np.exp(
            -(q**2) * volume ** (2.0 / 3.0) / (4.0 * np.pi)
        )
    return atomic, solvent


def debye_amplitude(
    positions: np.ndarray,
    q: np.ndarray,
    atomic_factors: np.ndarray,
    box: np.ndarray | None,
) -> np.ndarray:
    distances = distance_array(positions, positions, box=box)
    sinc_qr = np.sinc(q[:, None, None] * distances[None, :, :] / np.pi)
    intensity = np.einsum(
        "qi,qij,qj->q", atomic_factors, sinc_qr, atomic_factors, optimize=True
    )
    amplitude = np.sqrt(np.clip(intensity, 0.0, None))
    amplitude[0] = atomic_factors[0].sum()
    return amplitude


def _nested_method_value(settings: Mapping[str, Any], method: str, name: str) -> Any:
    values = settings.get(method, {})
    if not isinstance(values, Mapping):
        return None
    return values.get(name)


def restore_amplitude_sign(
    amplitude: np.ndarray,
    q: np.ndarray,
    crossover: float | None,
    width: float,
) -> tuple[np.ndarray, float | None, str | None]:
    magnitude = np.abs(np.asarray(amplitude, dtype=float))
    if amplitude[0] >= 0.0:
        return magnitude, None, None
    if crossover is None:
        index = 1 + int(np.argmin(magnitude[1:]))
        crossover = float(q[index])
        ambiguous = index == len(q) - 1 or magnitude[index] > 0.1 * abs(amplitude[0])
        note = "automatic sign crossover is weak or lies at the q boundary" if ambiguous else None
    else:
        note = None
    if not q[0] < crossover <= q[-1] or width <= 0.0:
        raise ValueError("sign crossover must lie inside the q grid and have positive width")
    transition = np.tanh((q - crossover) / (2.0 * width))
    q0_scale = abs(transition[0])
    if q0_scale <= np.finfo(float).eps:
        raise ValueError("sign crossover cannot coincide with q=0")
    transition[q <= crossover] /= q0_scale
    return transition * magnitude, crossover, note


def fit_polynomial(
    q: np.ndarray,
    values: np.ndarray,
    degree: int,
    low_q_max: float,
    low_q_weight: float,
    enforce_f0: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    if degree < 1 or q.size <= degree:
        raise ValueError("polynomial degree must be positive and smaller than q-grid size")
    weights = np.ones_like(q)
    weights[q <= low_q_max] = low_q_weight
    unconstrained = np.polynomial.polynomial.polyfit(q, values, degree, w=weights)
    if enforce_f0:
        design = q[:, None] ** np.arange(1, degree + 1)[None, :]
        lhs = design * weights[:, None]
        rhs = (values - values[0]) * weights
        tail, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        coefficients = np.concatenate(([values[0]], tail))
    else:
        coefficients = unconstrained
    fitted = np.polynomial.polynomial.polyval(q, coefficients)
    residual = fitted - values
    low = q <= low_q_max
    metrics = {
        "f0_true": float(values[0]),
        "f0_polynomial": float(coefficients[0]),
        "f0_unconstrained": float(unconstrained[0]),
        "f0_unconstrained_difference": float(unconstrained[0] - values[0]),
        "rmse_all": float(np.sqrt(np.mean(residual**2))),
        "rmse_low_q": float(np.sqrt(np.mean(residual[low] ** 2))),
        "max_abs_error_low_q": float(np.max(np.abs(residual[low]))),
    }
    return coefficients, metrics


def _fit_decomposed_terms(
    sums: Mapping[str, Mapping[str, np.ndarray]],
    sample_count: int,
    groups: Mapping[str, list[str]],
    q: np.ndarray,
    fit_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Average and fit every Equation-9 component without applying contrast."""
    result: dict[str, Any] = {"terms": {}}
    for term, bead_sums in sums.items():
        bead_curves = {
            bead: np.asarray(values, dtype=float) / sample_count
            for bead, values in bead_sums.items()
        }
        group_curves: OrderedDict[str, np.ndarray] = OrderedDict()
        coefficients: OrderedDict[str, np.ndarray] = OrderedDict()
        metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for group, members in groups.items():
            curve = np.mean([bead_curves[bead] for bead in members], axis=0)
            fitted, fit_metrics = fit_polynomial(
                q,
                curve,
                int(fit_settings["degree"]),
                float(fit_settings["low_q_max"]),
                float(fit_settings["low_q_weight"]),
                bool(fit_settings.get("enforce_f0", True)),
            )
            group_curves[group] = curve
            coefficients[group] = fitted
            metrics[group] = {**fit_metrics, "members": list(members)}
        result["terms"][term] = {
            "bead_curves": bead_curves,
            "group_curves": group_curves,
            "coefficients": coefficients,
            "bead_coefficients": {
                bead: coefficients[group]
                for group, members in groups.items()
                for bead in members
            },
            "metrics": metrics,
        }
    return result


def _decomposition_reconstruction_diagnostics(
    method: str,
    decomposition: dict[str, Any],
    legacy_curves: Mapping[str, np.ndarray],
    groups: Mapping[str, list[str]],
    densities: Mapping[str, float],
    q: np.ndarray,
) -> dict[str, Any]:
    """Validate raw and exported-polynomial Equation-9 reconstructions."""
    diagnostics: dict[str, Any] = {}
    terms = decomposition["terms"]
    for bead, legacy in legacy_curves.items():
        if method == "saxs":
            selected = {
                name: terms[name]["bead_curves"][bead]
                for name in ("atomic", "solvent", "mixed")
            }
        else:
            selected = {
                "atomic": terms["atomic_h"]["bead_curves"][bead],
                "solvent": terms["solvent"]["bead_curves"][bead],
                "mixed": terms["mixed_h"]["bead_curves"][bead],
            }
        density = float(densities[bead])
        reconstructed = np.sqrt(equation9_radicand(selected, density))
        difference = reconstructed - np.abs(legacy)
        fitted_selected = {
            name: np.polynomial.polynomial.polyval(
                q,
                terms[term_name]["bead_coefficients"][bead],
            )
            for name, term_name in (
                (("atomic", "atomic"), ("solvent", "solvent"), ("mixed", "mixed"))
                if method == "saxs"
                else (
                    ("atomic", "atomic_h"),
                    ("solvent", "solvent"),
                    ("mixed", "mixed_h"),
                )
            )
        }
        try:
            fitted_reconstructed = np.sqrt(
                equation9_radicand(fitted_selected, density)
            )
        except ValueError as exc:
            raise ValueError(
                f"{method.upper()} bead {bead} exported polynomial {exc}"
            ) from exc
        fitted_difference = fitted_reconstructed - np.abs(legacy)
        diagnostics[bead] = {
            "reference_density": density,
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "max_abs_error": float(np.max(np.abs(difference))),
            "fitted_rmse": float(np.sqrt(np.mean(fitted_difference**2))),
            "fitted_max_abs_error": float(np.max(np.abs(fitted_difference))),
        }
    decomposition["reconstruction"] = diagnostics
    decomposition["group_members"] = copy.deepcopy(dict(groups))
    return diagnostics


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Any, base: Path) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def calculate_molecule(
    molecule: Mapping[str, Any], config: Mapping[str, Any], base: Path
) -> MoleculeResult:
    # Project entries may override any global scientific setting without
    # duplicating the rest of the YAML configuration.
    effective_config = copy.deepcopy(dict(config))
    effective_config.pop("molecules", None)
    _deep_update(effective_config, molecule.get("settings", {}))
    config = effective_config
    name = str(molecule.get("name") or Path(str(molecule["mapping"])).stem)
    topology = _resolve_path(molecule.get("topology"), base)
    trajectory = _resolve_path(molecule.get("trajectory"), base)
    mapping_path = _resolve_path(molecule.get("mapping"), base)
    if topology is None or mapping_path is None:
        raise ValueError(f"{name}: topology and mapping are required")
    for path in (topology, mapping_path, trajectory):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    mapping = read_cgbuilder_mapping(mapping_path)
    universe = mda.Universe(str(topology), *([str(trajectory)] if trajectory else []))
    trajectory_settings = config["trajectory"]
    start = int(trajectory_settings.get("start", 0))
    stop_value = trajectory_settings.get("stop")
    stop = None if stop_value is None else int(stop_value)
    stride = int(trajectory_settings.get("stride", 1))
    if (
        start < 0
        or start >= len(universe.trajectory)
        or (stop is not None and stop <= start)
        or stride < 1
    ):
        raise ValueError("invalid trajectory start/stop/stride")
    selected_indices, occurrences = _selected_occurrences(
        universe, str(trajectory_settings["selection"]), mapping.atom_count
    )
    template_global_atoms = {
        local_index: int(global_index) + 1
        for local_index, global_index in enumerate(occurrences[0], start=1)
    }
    elements = resolve_elements(
        universe, selected_indices, mapping.atom_count, config.get("element_overrides", {})
    )
    isotopes = isotope_labels(elements, config["sans"].get("isotope_overrides", {}))
    bonds = _bond_table(
        universe, occurrences[0], bool(trajectory_settings.get("guess_bonds", False))
    )
    for occurrence_number, occurrence in enumerate(occurrences[1:], start=2):
        occurrence_elements = resolve_elements(
            universe,
            occurrence,
            mapping.atom_count,
            config.get("element_overrides", {}),
        )
        if occurrence_elements != elements:
            raise ValueError(
                f"{name}: selected molecule occurrence {occurrence_number} does not "
                "match the mapping template's element order"
            )
        occurrence_bonds = _bond_table(
            universe,
            occurrence,
            bool(trajectory_settings.get("guess_bonds", False)),
        )
        if occurrence_bonds != bonds:
            raise ValueError(
                f"{name}: selected molecule occurrence {occurrence_number} does not "
                "match the mapping template's bonded topology"
            )
    groups = group_equivalent_beads(
        mapping, elements, isotopes, bonds, config["saxs"], config["identity"]
    )
    q = q_grid(config["q"])
    volumes, volume_provenance = load_atomic_volumes(
        config["saxs"].get("excluded_volume_overrides", {})
    )

    decomposition_enabled = bool(config.get("decomposition", {}).get("enabled", False))
    decomposition_payload: dict[str, Any] | None = None
    exchangeable_hydrogens: tuple[int, ...] = ()
    exchanged_isotopes = list(isotopes)
    lcpo_assignments: list[LCPOAssignment] = []
    if decomposition_enabled:
        # Initialize geometry from the first frame included by the configured slice.
        universe.trajectory[start]
        decomposition_settings = config["decomposition"]
        representations = [
            str(value).lower() for value in decomposition_settings["representations"]
        ]
        if not representations or set(representations) - {"hybrid", "cg"}:
            raise ValueError("decomposition.representations must contain hybrid and/or cg")
        template_atoms = universe.atoms[occurrences[0]]
        atom_names = [str(value) for value in template_atoms.names]
        residue_names = [str(value) for value in template_atoms.resnames]
        lcpo_settings = decomposition_settings["lcpo"]
        allowed_lcpo_settings = {
            "allow_element_fallback",
            "atom_overrides",
            "element_overrides",
            "bead_overrides",
            "probe_radius_A",
        }
        unknown_lcpo_settings = sorted(set(lcpo_settings) - allowed_lcpo_settings)
        if unknown_lcpo_settings:
            raise ValueError(
                "unsupported decomposition.lcpo settings: "
                f"{unknown_lcpo_settings}"
            )
        diagnostic_settings = decomposition_settings.get("diagnostics", {})
        unknown_diagnostic_settings = sorted(
            set(diagnostic_settings) - {"accessible_fraction_cutoff"}
        )
        if unknown_diagnostic_settings:
            raise ValueError(
                "unsupported decomposition.diagnostics settings: "
                f"{unknown_diagnostic_settings}"
            )
        lcpo_assignments, lcpo_provenance = assign_lcpo_parameters(
            atom_names, residue_names, elements, lcpo_settings
        )
        exchange_settings = decomposition_settings.get("exchange", {})
        exchangeable_hydrogens = detect_exchangeable_hydrogens(
            elements,
            bonds,
            exchange_settings.get("include", []),
            exchange_settings.get("exclude", []),
        )
        for atom_index in exchangeable_hydrogens:
            exchanged_isotopes[atom_index - 1] = "D"
        # Exchange overrides are part of the decomposed SANS identity. Rebuild
        # automatic groups so distinct H/D end states are never averaged.
        groups = group_equivalent_beads(
            mapping,
            elements,
            exchanged_isotopes,
            bonds,
            config["saxs"],
            config["identity"],
        )
        _validate_exchange_grouping(groups, mapping, exchangeable_hydrogens)
        try:
            template_masses = np.asarray(template_atoms.masses, dtype=float)
        except NoDataError:
            template_masses = np.asarray(
                [float(pt.elements.symbol(element).mass) for element in elements], dtype=float
            )
        mapping_weights = {
            bead: normalized_mass_weights(
                [template_masses[index - 1] for index in mapping.bead_atoms[bead]]
            )
            for bead in mapping.bead_order
        }
        cg_lcpo, cg_lcpo_details = derive_cg_lcpo_parameters(
            mapping.bead_order,
            mapping.bead_atoms,
            mapping.atom_beads,
            elements,
            volumes,
            lcpo_assignments,
            lcpo_settings,
        )
        decomposition_payload = {
            "representations": representations,
            "mapping": {
                "atom_count": mapping.atom_count,
                "bead_atoms": {
                    bead: tuple(
                        template_global_atoms[index]
                        for index in mapping.bead_atoms[bead]
                    )
                    for bead in mapping.bead_order
                },
                "template_bead_atoms": copy.deepcopy(dict(mapping.bead_atoms)),
                "bead_weights": mapping_weights,
                "index_convention": "global one-based topology atom serials",
            },
            "lcpo": {
                "hybrid": lcpo_assignments,
                "cg": cg_lcpo,
                "provenance": lcpo_provenance,
                "cg_derivation": cg_lcpo_details,
            },
            "exchange": {
                "exchangeable_hydrogens": exchangeable_hydrogens,
                "eligible_beads": {
                    bead: any(
                        index in exchangeable_hydrogens for index in mapping.bead_atoms[bead]
                    )
                    for bead in mapping.bead_order
                },
            },
            "methods": {},
        }

    fractions = {
        bead: np.asarray(
            [1.0 / len(mapping.atom_beads[index]) for index in mapping.bead_atoms[bead]],
            dtype=float,
        )
        for bead in mapping.bead_order
    }
    density_default = float(config["saxs"]["solvent_electron_density"])
    density_overrides = {
        str(key): float(value)
        for key, value in config["saxs"].get("electron_density_overrides", {}).items()
    }
    factors: dict[str, dict[str, np.ndarray]] = {"saxs": {}, "sans": {}}
    component_factors: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "saxs": {},
        "sans_h": {},
        "sans_d": {},
    }
    sans_one = {}
    for bead in mapping.bead_order:
        local = mapping.bead_atoms[bead]
        bead_labels = [isotopes[index - 1] for index in local]
        bead_exchanged_labels = [exchanged_isotopes[index - 1] for index in local]
        if config["saxs"].get("enabled", True):
            factors["saxs"][bead] = xray_factors(
                bead_labels,
                fractions[bead],
                q,
                density_overrides.get(bead, density_default),
                volumes,
            )
            if decomposition_enabled:
                atomic, solvent = xray_component_factors(
                    bead_labels, fractions[bead], q, volumes
                )
                component_factors["saxs"][bead] = {
                    "atomic": atomic,
                    "solvent": solvent,
                }
        factors["sans"][bead] = sans_factors(
            bead_labels, fractions[bead], q
        )
        if decomposition_enabled:
            atomic_h, solvent = sans_component_factors(
                bead_labels, fractions[bead], q, volumes
            )
            atomic_d, _ = sans_component_factors(
                bead_exchanged_labels, fractions[bead], q, volumes
            )
            component_factors["sans_h"][bead] = {
                "atomic": atomic_h,
                "solvent": solvent,
            }
            component_factors["sans_d"][bead] = {
                "atomic": atomic_d,
                "solvent": solvent,
            }
        sans_one[bead] = float(factors["sans"][bead][0].sum())

    enabled_methods = []
    if config["saxs"].get("enabled", True):
        enabled_methods.append("saxs")
    if config["sans"].get("polynomial", True):
        enabled_methods.append("sans")
    sums = {
        method: {bead: np.zeros_like(q) for bead in mapping.bead_order}
        for method in enabled_methods
    }
    decomposed_sums: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    exposure_samples: dict[str, dict[str, list[float]]] = {
        bead: {"hybrid": [], "cg": []} for bead in mapping.bead_order
    }
    if decomposition_enabled:
        if config["saxs"].get("enabled", True):
            decomposed_sums["saxs"] = {
                term: {bead: np.zeros_like(q) for bead in mapping.bead_order}
                for term in ("atomic", "solvent", "mixed")
            }
        if config["sans"].get("polynomial", True):
            decomposed_sums["sans"] = {
                term: {bead: np.zeros_like(q) for bead in mapping.bead_order}
                for term in ("atomic_h", "mixed_h", "atomic_d", "mixed_d", "solvent")
            }
    frames = 0
    for timestep in universe.trajectory[start:stop:stride]:
        frames += 1
        box = valid_box(timestep.dimensions)
        for occurrence in occurrences:
            if decomposition_payload is not None:
                atom_positions = np.asarray(universe.atoms[occurrence].positions, dtype=float)
                probe = float(config["decomposition"]["lcpo"]["probe_radius_A"])
                atom_areas, atom_isolated = lcpo_areas(
                    atom_positions, lcpo_assignments, box, probe
                )
                hybrid_fractions = aggregate_hybrid_accessible_fractions(
                    atom_areas,
                    atom_isolated,
                    mapping.bead_order,
                    mapping.bead_atoms,
                    mapping.atom_beads,
                )
                centers = virtual_center_positions(
                    atom_positions,
                    mapping.bead_order,
                    mapping.bead_atoms,
                    decomposition_payload["mapping"]["bead_weights"],
                    box,
                )
                cg_assignments = [
                    decomposition_payload["lcpo"]["cg"][bead]
                    for bead in mapping.bead_order
                ]
                cg_areas, cg_isolated = lcpo_areas(
                    centers, cg_assignments, box, probe
                )
                for bead_index, bead in enumerate(mapping.bead_order):
                    cg_fraction = (
                        0.0
                        if cg_isolated[bead_index] <= 0.0
                        else cg_areas[bead_index] / cg_isolated[bead_index]
                    )
                    exposure_samples[bead]["hybrid"].append(hybrid_fractions[bead])
                    exposure_samples[bead]["cg"].append(float(cg_fraction))
            for bead in mapping.bead_order:
                local = np.asarray(mapping.bead_atoms[bead], dtype=int) - 1
                positions = universe.atoms[occurrence[local]].positions
                for method in enabled_methods:
                    sums[method][bead] += debye_amplitude(
                        positions, q, factors[method][bead], box
                    )
                if "saxs" in decomposed_sums:
                    components = component_factors["saxs"][bead]
                    terms = debye_component_terms(
                        positions,
                        q,
                        components["atomic"],
                        components["solvent"],
                        box,
                    )
                    for term, values in terms.items():
                        decomposed_sums["saxs"][term][bead] += values
                if "sans" in decomposed_sums:
                    components_h = component_factors["sans_h"][bead]
                    components_d = component_factors["sans_d"][bead]
                    terms_h = debye_component_terms(
                        positions,
                        q,
                        components_h["atomic"],
                        components_h["solvent"],
                        box,
                    )
                    terms_d = debye_component_terms(
                        positions,
                        q,
                        components_d["atomic"],
                        components_d["solvent"],
                        box,
                    )
                    decomposed_sums["sans"]["atomic_h"][bead] += terms_h["atomic"]
                    decomposed_sums["sans"]["mixed_h"][bead] += terms_h["mixed"]
                    decomposed_sums["sans"]["atomic_d"][bead] += terms_d["atomic"]
                    decomposed_sums["sans"]["mixed_d"][bead] += terms_d["mixed"]
                    decomposed_sums["sans"]["solvent"][bead] += terms_h["solvent"]
    if frames == 0:
        raise ValueError(f"{name}: no trajectory frames were sampled")
    sample_count = frames * len(occurrences)

    method_results: dict[str, dict[str, Any]] = {}
    run_warnings: list[str] = []
    for method in enabled_methods:
        bead_curves = {bead: sums[method][bead] / sample_count for bead in mapping.bead_order}
        group_curves: OrderedDict[str, np.ndarray] = OrderedDict()
        positive_magnitude_curves: OrderedDict[str, np.ndarray] = OrderedDict()
        coefficients: OrderedDict[str, np.ndarray] = OrderedDict()
        metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
        crossovers: OrderedDict[str, float | None] = OrderedDict()
        for group, members in groups.items():
            curve = np.mean([bead_curves[bead] for bead in members], axis=0)
            explicit = _nested_method_value(config["sign"].get("crossovers", {}), method, group)
            if explicit is None:
                explicit = _nested_method_value(
                    config["sign"].get("crossovers", {}), method, members[0]
                )
            width_value = _nested_method_value(config["sign"].get("widths", {}), method, group)
            if width_value is None:
                width_value = _nested_method_value(
                    config["sign"].get("widths", {}), method, members[0]
                )
            width = float(width_value or config["sign"]["default_width"])
            signed, crossover, note = restore_amplitude_sign(
                curve, q, None if explicit is None else float(explicit), width
            )
            if note:
                run_warnings.append(f"{method}/{group}: {note}")
            coeff, fit_metrics = fit_polynomial(
                q,
                signed,
                int(config["fit"]["degree"]),
                float(config["fit"]["low_q_max"]),
                float(config["fit"]["low_q_weight"]),
                bool(config["fit"].get("enforce_f0", True)),
            )
            group_curves[group] = signed
            positive_magnitude_curves[group] = np.abs(curve)
            coefficients[group] = coeff
            metrics[group] = {**fit_metrics, "members": list(members), "sign_width": width}
            crossovers[group] = crossover
        bead_coefficients = {
            bead: coefficients[group]
            for group, members in groups.items()
            for bead in members
        }
        method_results[method] = {
            "bead_curves": bead_curves,
            "group_curves": group_curves,
            "positive_magnitude_curves": positive_magnitude_curves,
            "coefficients": coefficients,
            "bead_coefficients": bead_coefficients,
            "metrics": metrics,
            "crossovers": crossovers,
        }

    if decomposition_payload is not None:
        decomposition_payload["q"] = {
            "minimum_A^-1": float(q[0]),
            "maximum_A^-1": float(q[-1]),
            "step_A^-1": float(q[1] - q[0]),
        }
        cutoff = float(
            config["decomposition"]["diagnostics"]["accessible_fraction_cutoff"]
        )
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError(
                "decomposition.diagnostics.accessible_fraction_cutoff must be in [0, 1]"
            )
        hybrid_all = np.asarray(
            [value for bead in mapping.bead_order for value in exposure_samples[bead]["hybrid"]]
        )
        cg_all = np.asarray(
            [value for bead in mapping.bead_order for value in exposure_samples[bead]["cg"]]
        )
        hybrid_exposed = hybrid_all >= cutoff
        cg_exposed = cg_all >= cutoff
        correlation = (
            float(np.corrcoef(hybrid_all, cg_all)[0, 1])
            if hybrid_all.size > 1
            and np.std(hybrid_all) > np.finfo(float).eps
            and np.std(cg_all) > np.finfo(float).eps
            else None
        )
        per_bead_exposure = {
            bead: {
                "hybrid_mean_fraction": float(np.mean(exposure_samples[bead]["hybrid"])),
                "cg_mean_fraction": float(np.mean(exposure_samples[bead]["cg"])),
                "mean_absolute_difference": float(
                    np.mean(
                        np.abs(
                            np.asarray(exposure_samples[bead]["hybrid"])
                            - np.asarray(exposure_samples[bead]["cg"])
                        )
                    )
                ),
            }
            for bead in mapping.bead_order
        }
        decomposition_payload["lcpo"]["exposure_comparison"] = {
            "accessible_fraction_cutoff": cutoff,
            "exposed_when": "accessible_fraction >= cutoff",
            "sample_count": int(hybrid_all.size),
            "pearson_correlation": correlation,
            "mismatch_fraction": float(np.mean(hybrid_exposed != cg_exposed)),
            "confusion_matrix": {
                "both_exposed": int(np.sum(hybrid_exposed & cg_exposed)),
                "hybrid_only": int(np.sum(hybrid_exposed & ~cg_exposed)),
                "cg_only": int(np.sum(~hybrid_exposed & cg_exposed)),
                "both_buried": int(np.sum(~hybrid_exposed & ~cg_exposed)),
            },
            "per_bead": per_bead_exposure,
            "largest_mean_mismatches": sorted(
                per_bead_exposure,
                key=lambda bead: per_bead_exposure[bead]["mean_absolute_difference"],
                reverse=True,
            )[:10],
        }
        decomposition_payload["exposure_samples"] = exposure_samples
        for method, term_sums in decomposed_sums.items():
            fitted = _fit_decomposed_terms(
                term_sums, sample_count, groups, q, config["fit"]
            )
            if method == "saxs":
                densities = {
                    bead: density_overrides.get(bead, density_default)
                    for bead in mapping.bead_order
                }
            else:
                # The legacy SANS custom-parameter path has no displaced-solvent
                # contrast, so rho=0 is the exact reconstruction reference.
                densities = {bead: 0.0 for bead in mapping.bead_order}
            _decomposition_reconstruction_diagnostics(
                method,
                fitted,
                method_results[method]["bead_curves"],
                groups,
                densities,
                q,
            )
            fitted["reference_density"] = densities
            fitted["reference_sign_crossovers_A^-1"] = copy.deepcopy(
                method_results[method]["crossovers"]
            )
            decomposition_payload["methods"][method] = fitted

    provenance = {
        "topology": {"path": str(topology), "sha256": _file_sha256(topology)},
        "trajectory": {
            "path": None if trajectory is None else str(trajectory),
            "sha256": _file_sha256(trajectory),
        },
        "mapping": {"path": str(mapping_path), "sha256": _file_sha256(mapping_path)},
        "atomic_volumes": volume_provenance,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("MDAnalysis", "networkx", "numpy", "periodictable", "PyYAML")
        },
        "frame_selection": {"start": start, "stop": stop, "stride": stride},
        "shared_atom_policy": "equal amplitude split among mapped beads",
    }
    return MoleculeResult(
        name=name,
        bead_order=list(mapping.bead_order),
        groups=groups,
        frames=frames,
        occurrences=len(occurrences),
        q=q,
        methods=method_results,
        sans_one_parameter=sans_one,
        warnings=run_warnings,
        provenance=provenance,
        decomposition=decomposition_payload,
    )


def _prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        existing = list(path.iterdir())
        if existing and not force:
            raise FileExistsError(
                f"output directory is not empty: {path}; choose a new directory or use --force"
            )
    else:
        path.mkdir(parents=True)


def _write_text(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    path.write_text(text)


def _render_coefficients(values: Iterable[float]) -> str:
    return ",".join(f"{float(value):.16g}" for value in values)


def _contextual_parameter_text(
    result: MoleculeResult,
    method: str,
    term: str,
) -> str:
    if result.decomposition is None:
        raise ValueError("decomposed parameters were not calculated")
    q_meta = result.decomposition["q"]
    term_data = result.decomposition["methods"][method]["terms"][term.lower()]
    rows = [term_data["bead_coefficients"][bead] for bead in result.bead_order]
    return _contextual_parameter_rows_text(method, term, q_meta, rows)


def _contextual_parameter_rows_text(
    method: str,
    term: str,
    q_meta: Mapping[str, float],
    rows: Iterable[Iterable[float]],
) -> str:
    lines = [
        "FORMAT_VERSION=1",
        f"METHOD={method.upper()}",
        f"TERM={term.upper()}",
        f"Q_MIN={q_meta['minimum_A^-1']:.16g}",
        f"Q_MAX={q_meta['maximum_A^-1']:.16g}",
        f"Q_STEP={q_meta['step_A^-1']:.16g}",
    ]
    for index, values in enumerate(rows, start=1):
        lines.append(f"PARAMETERS{index}={_render_coefficients(values)}")
    return "\n".join(lines) + "\n"


def _lcpo_parameter_text(
    assignments: Iterable[LCPOAssignment], representation: str
) -> str:
    lines = ["FORMAT_VERSION=1", f"REPRESENTATION={representation.upper()}"]
    for index, assignment in enumerate(assignments, start=1):
        lines.append(f"LCPO_PARAMETERS{index}={_render_coefficients(assignment.values)}")
    return "\n".join(lines) + "\n"


def _mapping_text(result: MoleculeResult) -> str:
    if result.decomposition is None:
        raise ValueError("decomposed mapping was not calculated")
    mapping = result.decomposition["mapping"]
    lines = ["FORMAT_VERSION=1"]
    for index, bead in enumerate(result.bead_order, start=1):
        atoms = ",".join(str(value) for value in mapping["bead_atoms"][bead])
        weights = _render_coefficients(mapping["bead_weights"][bead])
        lines.append(f"BEAD_ATOMS{index}={atoms}")
        lines.append(f"BEAD_WEIGHTS{index}={weights}")
    return "\n".join(lines) + "\n"


def _inline_decomposition_text(result: MoleculeResult, method: str) -> str:
    if result.decomposition is None:
        raise ValueError("decomposed parameters were not calculated")
    q_meta = result.decomposition["q"]
    lines = [
        f"# {result.name}: verbose {method.upper()} Equation-9 keywords",
        f"PARAMETERS_Q_MIN={q_meta['minimum_A^-1']:.16g}",
        f"PARAMETERS_Q_MAX={q_meta['maximum_A^-1']:.16g}",
        f"PARAMETERS_Q_STEP={q_meta['step_A^-1']:.16g}",
    ]
    keywords = (
        {
            "atomic": "ATOMIC_PARAMETERS",
            "solvent": "SOLVENT_PARAMETERS",
            "mixed": "MIXED_PARAMETERS",
        }
        if method == "saxs"
        else {
            "atomic_h": "ATOMIC_PARAMETERS",
            "mixed_h": "MIXED_PARAMETERS",
            "solvent": "SOLVENT_PARAMETERS",
            "atomic_d": "DEUTERATED_ATOMIC_PARAMETERS",
            "mixed_d": "DEUTERATED_MIXED_PARAMETERS",
        }
    )
    for term, keyword in keywords.items():
        term_data = result.decomposition["methods"][method]["terms"][term]
        for index, bead in enumerate(result.bead_order, start=1):
            lines.append(
                f"{keyword}{index}="
                + _render_coefficients(term_data["bead_coefficients"][bead])
            )
    return "\n".join(lines) + "\n"


def write_decomposition_outputs(
    output: Path, result: MoleculeResult, force: bool
) -> None:
    """Write all version-1 Equation-9, mapping, LCPO, and diagnostic files."""
    if result.decomposition is None:
        return
    prefix = output / result.name
    file_terms: list[tuple[str, str, str]] = []
    if "saxs" in result.decomposition["methods"]:
        file_terms.extend(
            ("saxs", term, f"{prefix}_saxs_{term}_parameters.inp")
            for term in ("atomic", "solvent", "mixed")
        )
    if "sans" in result.decomposition["methods"]:
        file_terms.extend(
            [
                ("sans", "atomic_h", f"{prefix}_sans_h_atomic_parameters.inp"),
                ("sans", "mixed_h", f"{prefix}_sans_h_mixed_parameters.inp"),
                ("sans", "atomic_d", f"{prefix}_sans_d_atomic_parameters.inp"),
                ("sans", "mixed_d", f"{prefix}_sans_d_mixed_parameters.inp"),
                ("sans", "solvent", f"{prefix}_sans_solvent_parameters.inp"),
            ]
        )
    for method, term, filename in file_terms:
        _write_text(
            Path(filename),
            _contextual_parameter_text(result, method, term),
            force,
        )
    for method in result.decomposition["methods"]:
        _write_text(
            Path(f"{prefix}_{method}_equation9_inline.inp"),
            _inline_decomposition_text(result, method),
            force,
        )

    representations = result.decomposition["representations"]
    if "hybrid" in representations:
        _write_text(Path(f"{prefix}_hybrid_mapping.inp"), _mapping_text(result), force)
        _write_text(
            Path(f"{prefix}_hybrid_lcpo_parameters.inp"),
            _lcpo_parameter_text(result.decomposition["lcpo"]["hybrid"], "hybrid"),
            force,
        )
    if "cg" in representations:
        assignments = [
            result.decomposition["lcpo"]["cg"][bead] for bead in result.bead_order
        ]
        _write_text(
            Path(f"{prefix}_cg_lcpo_parameters.inp"),
            _lcpo_parameter_text(assignments, "cg"),
            force,
        )

    eligible = result.decomposition["exchange"]["eligible_beads"]
    eligible_indices = [
        str(index)
        for index, bead in enumerate(result.bead_order, start=1)
        if eligible[bead]
    ]
    exchange_lines = [
        "FORMAT_VERSION=1",
        "EXCHANGEABLE_BEADS=" + ",".join(eligible_indices),
    ]
    _write_text(
        Path(f"{prefix}_sans_exchangeable_beads.inp"),
        "\n".join(exchange_lines) + "\n",
        force,
    )

    diagnostics_path = Path(f"{prefix}_equation9_terms.csv")
    if diagnostics_path.exists() and not force:
        raise FileExistsError(diagnostics_path)
    with diagnostics_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["method", "term", "group", "q_A^-1", "value", "fit", "residual"])
        for method, method_data in result.decomposition["methods"].items():
            for term, term_data in method_data["terms"].items():
                for group, curve in term_data["group_curves"].items():
                    fitted = np.polynomial.polynomial.polyval(
                        result.q, term_data["coefficients"][group]
                    )
                    for q_value, value, fit_value in zip(
                        result.q, curve, fitted, strict=True
                    ):
                        writer.writerow(
                            [
                                method,
                                term,
                                group,
                                q_value,
                                value,
                                fit_value,
                                fit_value - value,
                            ]
                        )


def write_parameter_fragment(
    path: Path, result: MoleculeResult, mode: str, force: bool, start_index: int = 1
) -> int:
    lines = [f"# {result.name}: {mode}; one mapped molecule"]
    index = start_index
    for bead in result.bead_order:
        if mode == "sans_one_parameter":
            lines.append(f"  SCATLEN{index}={result.sans_one_parameter[bead]:.16g}")
        else:
            method = mode.split("_", 1)[0]
            values = result.methods[method]["bead_coefficients"][bead]
            lines.append(f"  PARAMETERS{index}={_render_coefficients(values)}")
        index += 1
    _write_text(path, "\n".join(lines) + "\n", force)
    return index


def write_curves(path: Path, result: MoleculeResult, method: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        curves = result.methods[method]["group_curves"]
        writer.writerow(["q_A^-1", *curves])
        for row, q_value in enumerate(result.q):
            writer.writerow([q_value, *[values[row] for values in curves.values()]])


def write_coefficients(path: Path, result: MoleculeResult, method: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    degree = len(next(iter(result.methods[method]["coefficients"].values()))) - 1
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", *[f"c{i}" for i in range(degree + 1)]])
        for group, values in result.methods[method]["coefficients"].items():
            writer.writerow([group, *values])


def plot_fits(
    path: Path,
    result: MoleculeResult,
    low_q_max: float,
    force: bool,
    show_positive_magnitude: bool = True,
) -> None:
    methods = list(result.methods)
    figure, axes = plt.subplots(2, len(methods), figsize=(8 * len(methods), 9), squeeze=False)
    for column, method in enumerate(methods):
        top, bottom = axes[0, column], axes[1, column]
        for group, curve in result.methods[method]["group_curves"].items():
            coefficients = result.methods[method]["coefficients"][group]
            fitted = np.polynomial.polynomial.polyval(result.q, coefficients)
            line = top.plot(result.q, curve, label=group)[0]
            if show_positive_magnitude:
                positive = result.methods[method]["positive_magnitude_curves"][group]
                top.plot(
                    result.q,
                    positive,
                    ":",
                    color=line.get_color(),
                    alpha=0.75,
                    zorder=line.get_zorder() - 1,
                )
            top.plot(result.q, fitted, "--", color=line.get_color(), alpha=0.85)
            bottom.plot(result.q, fitted - curve, color=line.get_color(), label=group)
        for axis in (top, bottom):
            axis.axvspan(0.0, low_q_max, color="black", alpha=0.04)
            axis.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
            axis.set_xlabel(r"$q$ ($\AA^{-1}$)")
        top.set_title(f"{result.name} {method.upper()}")
        top.set_ylabel("bead form-factor amplitude")
        style_handles = [
            Line2D([], [], color="0.25", linestyle="-", label="signed form factor"),
            Line2D([], [], color="0.25", linestyle="--", label="polynomial fit"),
        ]
        if show_positive_magnitude:
            style_handles.append(
                Line2D([], [], color="0.25", linestyle=":", label="positive magnitude")
            )
        group_handles, group_labels = top.get_legend_handles_labels()
        top.legend(
            handles=[*group_handles, *style_handles],
            labels=[*group_labels, *[handle.get_label() for handle in style_handles]],
            ncols=2,
            fontsize=8,
        )
        bottom.set_ylabel("residuals")
    figure.tight_layout()
    if path.exists() and not force:
        plt.close(figure)
        raise FileExistsError(path)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_decomposition_term_fits(
    output: Path,
    result: MoleculeResult,
    low_q_max: float,
    force: bool,
) -> None:
    """Plot every Equation-9 term and its exported polynomial reconstruction."""
    if result.decomposition is None:
        return
    for method, method_data in result.decomposition["methods"].items():
        for term, term_data in method_data["terms"].items():
            path = output / f"{result.name}_{method}_{term}_fit.png"
            figure, (top, bottom) = plt.subplots(
                2,
                1,
                figsize=(8, 9),
                sharex=True,
                gridspec_kw={"height_ratios": (3, 1)},
            )
            for group, curve in term_data["group_curves"].items():
                coefficients = term_data["coefficients"][group]
                fitted = np.polynomial.polynomial.polyval(result.q, coefficients)
                line = top.plot(result.q, curve, label=group)[0]
                top.plot(
                    result.q,
                    fitted,
                    "--",
                    color=line.get_color(),
                    alpha=0.85,
                )
                bottom.plot(
                    result.q,
                    fitted - curve,
                    color=line.get_color(),
                    label=group,
                )
            for axis in (top, bottom):
                axis.axvspan(0.0, low_q_max, color="black", alpha=0.04)
                axis.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
            top.set_title(
                f"{result.name} {method.upper()} Equation 9: "
                f"{term.replace('_', ' ')}"
            )
            top.set_ylabel("Equation-9 term value")
            group_legend = top.legend(ncols=2, fontsize=8)
            bottom.set_xlabel(r"$q$ ($\AA^{-1}$)")
            bottom.set_ylabel("fit - value")
            style_handles = [
                Line2D([], [], color="0.25", linestyle="-", label="raw term"),
                Line2D([], [], color="0.25", linestyle="--", label="polynomial fit"),
            ]
            top.add_artist(group_legend)
            top.legend(handles=style_handles, loc="lower left")
            figure.tight_layout()
            if path.exists() and not force:
                plt.close(figure)
                raise FileExistsError(path)
            figure.savefig(path, dpi=180)
            plt.close(figure)


def write_exposure_samples(path: Path, result: MoleculeResult, force: bool) -> None:
    """Write every hybrid/native-CG LCPO fraction pair used by the diagnostics."""
    if result.decomposition is None:
        return
    samples = result.decomposition.get("exposure_samples")
    if not samples:
        return
    if path.exists() and not force:
        raise FileExistsError(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "bead", "hybrid_fraction", "cg_fraction"])
        for bead in result.bead_order:
            hybrid = samples[bead]["hybrid"]
            cg = samples[bead]["cg"]
            for sample, (hybrid_value, cg_value) in enumerate(
                zip(hybrid, cg, strict=True)
            ):
                writer.writerow([sample, bead, hybrid_value, cg_value])


def plot_exposure_correlation(path: Path, result: MoleculeResult, force: bool) -> None:
    """Plot all LCPO fraction pairs and the bead-wise means against identity."""
    if result.decomposition is None:
        return
    samples = result.decomposition.get("exposure_samples")
    if not samples:
        return
    hybrid = np.concatenate(
        [np.asarray(samples[bead]["hybrid"], dtype=float) for bead in result.bead_order]
    )
    cg = np.concatenate(
        [np.asarray(samples[bead]["cg"], dtype=float) for bead in result.bead_order]
    )
    comparison = result.decomposition["lcpo"]["exposure_comparison"]
    cutoff = float(comparison["accessible_fraction_cutoff"])
    correlation = comparison["pearson_correlation"]
    mae = float(np.mean(np.abs(cg - hybrid)))
    bias = float(np.mean(cg - hybrid))

    figure, axis = plt.subplots(figsize=(7.2, 6.4))
    axis.scatter(
        hybrid,
        cg,
        s=13,
        alpha=0.22,
        edgecolors="none",
        color="tab:blue",
        label=f"bead/frame observations (n={hybrid.size})",
    )
    hybrid_means = np.asarray(
        [np.mean(samples[bead]["hybrid"]) for bead in result.bead_order]
    )
    cg_means = np.asarray([np.mean(samples[bead]["cg"]) for bead in result.bead_order])
    axis.scatter(
        hybrid_means,
        cg_means,
        s=42,
        marker="D",
        color="tab:orange",
        edgecolors="black",
        linewidths=0.4,
        label="bead means",
        zorder=3,
    )
    axis.plot([0.0, 1.0], [0.0, 1.0], ":", color="0.3", label="perfect agreement")
    axis.axvline(cutoff, linestyle="--", linewidth=0.9, color="0.55")
    axis.axhline(cutoff, linestyle="--", linewidth=0.9, color="0.55")
    correlation_text = "undefined" if correlation is None else f"{correlation:.4f}"
    axis.text(
        0.03,
        0.97,
        f"Pearson r = {correlation_text}\nMAE = {mae:.4f}\nCG-hybrid bias = {bias:+.4f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="hybrid LCPO accessible fraction",
        ylabel="native-CG LCPO accessible fraction",
        title=f"{result.name}: hybrid versus native-CG exposure",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    if path.exists() and not force:
        plt.close(figure)
        raise FileExistsError(path)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def result_report(result: MoleculeResult, config: Mapping[str, Any]) -> dict[str, Any]:
    report = {
        "name": result.name,
        "frames": result.frames,
        "molecule_occurrences": result.occurrences,
        "bead_order": result.bead_order,
        "identical_bead_groups": result.groups,
        "q": config["q"],
        "fit": config["fit"],
        "methods": {
            method: {
                "coefficients_c0_first": data["coefficients"],
                "metrics": data["metrics"],
                "sign_crossovers_A^-1": data["crossovers"],
            }
            for method, data in result.methods.items()
        },
        "sans_one_parameter_fm": result.sans_one_parameter,
        "warnings": result.warnings,
        "provenance": result.provenance,
    }
    if result.decomposition is not None:
        report["equation9_decomposition"] = {
            "format_version": 1,
            "representations": result.decomposition["representations"],
            "q": result.decomposition["q"],
            "mapping": result.decomposition["mapping"],
            "lcpo": result.decomposition["lcpo"],
            "exchange": result.decomposition["exchange"],
            "methods": {
                method: {
                    "terms": {
                        term: {
                            "coefficients_c0_first": data["coefficients"],
                            "metrics": data["metrics"],
                        }
                        for term, data in method_data["terms"].items()
                    },
                    "reference_density": method_data["reference_density"],
                    "reference_reconstruction": method_data["reconstruction"],
                    "reference_sign_crossovers_A^-1": method_data[
                        "reference_sign_crossovers_A^-1"
                    ],
                }
                for method, method_data in result.decomposition["methods"].items()
            },
        }
    return report


def write_result(
    output: Path,
    result: MoleculeResult,
    config: Mapping[str, Any],
    force: bool,
) -> None:
    prefix = output / result.name
    for method in result.methods:
        write_curves(Path(f"{prefix}_{method}_curves.csv"), result, method, force)
        write_coefficients(Path(f"{prefix}_{method}_coefficients.csv"), result, method, force)
        write_parameter_fragment(
            Path(f"{prefix}_{method}_parameters.inp"),
            result,
            f"{method}_polynomial",
            force,
        )
    if config["sans"].get("one_parameter", True):
        write_parameter_fragment(
            Path(f"{prefix}_sans_scattering_lengths.inp"),
            result,
            "sans_one_parameter",
            force,
        )
    write_decomposition_outputs(output, result, force)
    plot_fits(
        Path(f"{prefix}_form_factor_fits.png"),
        result,
        float(config["fit"]["low_q_max"]),
        force,
        bool(config.get("plot", {}).get("show_positive_magnitude", True)),
    )
    if result.decomposition is not None:
        plot_decomposition_term_fits(
            output,
            result,
            float(config["fit"]["low_q_max"]),
            force,
        )
        write_exposure_samples(Path(f"{prefix}_lcpo_exposure_samples.csv"), result, force)
        plot_exposure_correlation(
            Path(f"{prefix}_lcpo_exposure_correlation.png"), result, force
        )
    report = yaml.safe_dump(_plain(result_report(result, config)), sort_keys=False)
    _write_text(Path(f"{prefix}_report.yaml"), report, force)


def write_project_outputs(
    output: Path,
    results: list[MoleculeResult],
    molecules: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    force: bool,
) -> None:
    example = [
        "# Copy the desired generated lines into the corresponding PLUMED SAXS/SANS action.",
        "# Polynomial fragments use PARAMETERSn=c0,c1,...,c6.",
        "# One-parameter SANS fragments use SCATLENn=b.",
    ]
    for result in results:
        example.extend(
            [
                "",
                f"# {result.name}",
                f"# SAXS: {result.name}_saxs_parameters.inp",
                f"# SANS polynomial: {result.name}_sans_parameters.inp",
                f"# SANS one parameter: {result.name}_sans_scattering_lengths.inp",
            ]
        )
        if result.decomposition is not None:
            example.extend(
                [
                    f"# Equation-9 SAXS files: {result.name}_saxs_*_parameters.inp",
                    f"# Equation-9 SANS files: {result.name}_sans_*_parameters.inp",
                    f"# Hybrid mapping/LCPO: {result.name}_hybrid_*.inp",
                    f"# Native-CG LCPO: {result.name}_cg_lcpo_parameters.inp",
                    f"# Verbose keywords: {result.name}_*_equation9_inline.inp",
                ]
            )
    _write_text(output / "plumed_example.inp", "\n".join(example) + "\n", force)

    if len(molecules) == 1 and not any(
        int(molecule.get("count", 1)) != 1 for molecule in molecules
    ):
        return
    modes = ["saxs_polynomial", "sans_polynomial", "sans_one_parameter"]
    by_name = {result.name: result for result in results}
    for mode in modes:
        if mode.startswith("saxs") and not config["saxs"].get("enabled", True):
            continue
        if mode == "sans_polynomial" and not config["sans"].get("polynomial", True):
            continue
        if mode == "sans_one_parameter" and not config["sans"].get("one_parameter", True):
            continue
        lines = [f"# Expanded project fragment: {mode}"]
        index = 1
        for molecule in molecules:
            result = by_name[str(molecule["name"])]
            for _ in range(int(molecule.get("count", 1))):
                for bead in result.bead_order:
                    if mode == "sans_one_parameter":
                        lines.append(f"  SCATLEN{index}={result.sans_one_parameter[bead]:.16g}")
                    else:
                        method = mode.split("_", 1)[0]
                        values = result.methods[method]["bead_coefficients"][bead]
                        lines.append(f"  PARAMETERS{index}={_render_coefficients(values)}")
                    index += 1
        _write_text(output / f"project_{mode}.inp", "\n".join(lines) + "\n", force)

    expanded: list[MoleculeResult] = []
    for molecule in molecules:
        result = by_name[str(molecule["name"])]
        expanded.extend([result] * int(molecule.get("count", 1)))
    if all(result.decomposition is None for result in expanded):
        return
    if any(result.decomposition is None for result in expanded):
        raise ValueError("all project molecules must provide Equation-9 decomposition data")
    decompositions = [result.decomposition for result in expanded]
    assert all(item is not None for item in decompositions)
    first = decompositions[0]
    assert first is not None
    for item in decompositions[1:]:
        assert item is not None
        if item["q"] != first["q"]:
            raise ValueError("project Equation-9 outputs require identical q grids")
        if item["representations"] != first["representations"]:
            raise ValueError("project molecules must export the same LCPO representations")
        if set(item["methods"]) != set(first["methods"]):
            raise ValueError("project molecules must calculate the same scattering methods")

    term_specs: list[tuple[str, str, str]] = []
    if "saxs" in first["methods"]:
        term_specs.extend(
            ("saxs", term, f"project_saxs_{term}_parameters.inp")
            for term in ("atomic", "solvent", "mixed")
        )
    if "sans" in first["methods"]:
        term_specs.extend(
            [
                ("sans", "atomic_h", "project_sans_h_atomic_parameters.inp"),
                ("sans", "mixed_h", "project_sans_h_mixed_parameters.inp"),
                ("sans", "atomic_d", "project_sans_d_atomic_parameters.inp"),
                ("sans", "mixed_d", "project_sans_d_mixed_parameters.inp"),
                ("sans", "solvent", "project_sans_solvent_parameters.inp"),
            ]
        )
    for method, term, filename in term_specs:
        rows = [
            result.decomposition["methods"][method]["terms"][term]["bead_coefficients"][bead]
            for result in expanded
            for bead in result.bead_order
            if result.decomposition is not None
        ]
        _write_text(
            output / filename,
            _contextual_parameter_rows_text(method, term, first["q"], rows),
            force,
        )

    atom_offset = 0
    bead_index = 1
    mapping_lines = ["FORMAT_VERSION=1"]
    hybrid_assignments: list[LCPOAssignment] = []
    cg_assignments: list[LCPOAssignment] = []
    exchangeable_beads: list[str] = []
    for result in expanded:
        assert result.decomposition is not None
        mapping = result.decomposition["mapping"]
        template_bead_atoms = mapping.get("template_bead_atoms", mapping["bead_atoms"])
        for bead in result.bead_order:
            atoms = [atom_offset + int(value) for value in template_bead_atoms[bead]]
            mapping_lines.append(f"BEAD_ATOMS{bead_index}=" + ",".join(map(str, atoms)))
            mapping_lines.append(
                f"BEAD_WEIGHTS{bead_index}="
                + _render_coefficients(mapping["bead_weights"][bead])
            )
            if result.decomposition["exchange"]["eligible_beads"][bead]:
                exchangeable_beads.append(str(bead_index))
            cg_assignments.append(result.decomposition["lcpo"]["cg"][bead])
            bead_index += 1
        hybrid_assignments.extend(result.decomposition["lcpo"]["hybrid"])
        atom_offset += int(mapping["atom_count"])

    if "hybrid" in first["representations"]:
        _write_text(
            output / "project_hybrid_mapping.inp",
            "\n".join(mapping_lines) + "\n",
            force,
        )
        _write_text(
            output / "project_hybrid_lcpo_parameters.inp",
            _lcpo_parameter_text(hybrid_assignments, "hybrid"),
            force,
        )
    if "cg" in first["representations"]:
        _write_text(
            output / "project_cg_lcpo_parameters.inp",
            _lcpo_parameter_text(cg_assignments, "cg"),
            force,
        )
    _write_text(
        output / "project_sans_exchangeable_beads.inp",
        "FORMAT_VERSION=1\nEXCHANGEABLE_BEADS=" + ",".join(exchangeable_beads) + "\n",
        force,
    )


def _molecule_specs(args: argparse.Namespace, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("molecules")
    if configured:
        cli_molecule_values = (args.topology, args.trajectory, args.mapping, args.name)
        if any(value is not None for value in cli_molecule_values):
            raise ValueError("CLI molecule arguments cannot be combined with YAML molecules")
        specs = [dict(item) for item in configured]
    else:
        if args.topology is None or args.mapping is None:
            raise ValueError("--topology and --mapping are required without YAML molecules")
        specs = [
            {
                "name": args.name or Path(args.mapping).stem,
                "topology": args.topology,
                "trajectory": args.trajectory,
                "mapping": args.mapping,
                "count": 1,
            }
        ]
    names = [str(spec.get("name") or Path(str(spec["mapping"])).stem) for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("molecule names must be unique")
    for spec, name in zip(specs, names, strict=True):
        spec["name"] = name
        count = int(spec.get("count", 1))
        if count < 1:
            raise ValueError(f"{name}: count must be positive")
        spec["count"] = count
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", help="bonded atomistic topology")
    parser.add_argument(
        "--trajectory",
        help="atomistic trajectory; topology coordinates are used if omitted",
    )
    parser.add_argument("--mapping", help="CGBuilder mapping")
    parser.add_argument("--name", help="output molecule name")
    parser.add_argument("--config", type=Path, help="YAML settings or project file")
    parser.add_argument("--output", type=Path, help="new output directory")
    parser.add_argument(
        "--force", action="store_true", help="allow replacement of named output files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and report bead groups without writing",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="generate Equation-9 atomic, solvent, mixed, mapping, and LCPO files",
    )
    parser.add_argument(
        "--representation",
        choices=("hybrid", "cg", "both"),
        help="LCPO representation(s) to export; implies --decompose",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config, base = load_config(args.config)
        if args.decompose or args.representation:
            config["decomposition"]["enabled"] = True
        if args.representation:
            config["decomposition"]["representations"] = (
                ["hybrid", "cg"]
                if args.representation == "both"
                else [args.representation]
            )
        molecules = _molecule_specs(args, config)
        results = [calculate_molecule(molecule, config, base) for molecule in molecules]
        for result in results:
            print(
                f"{result.name}: {result.frames} frame(s), {result.occurrences} occurrence(s), "
                f"groups={dict(result.groups)}"
            )
            for note in result.warnings:
                print(f"warning: {note}", file=sys.stderr)
        if args.dry_run:
            print("Dry run complete; no files written.")
            return 0
        if args.output is None:
            raise ValueError("--output is required unless --dry-run is used")
        output = args.output.resolve()
        _prepare_output(output, args.force)
        for result in results:
            write_result(output, result, config, args.force)
        write_project_outputs(output, results, molecules, config, args.force)
        manifest = {
            "configuration": config,
            "molecules": molecules,
            "results": [result_report(result, config) for result in results],
        }
        _write_text(
            output / "form_factors_manifest.yaml",
            yaml.safe_dump(_plain(manifest), sort_keys=False),
            args.force,
        )
        print(f"Wrote outputs to {output}")
        return 0
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
