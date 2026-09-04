#!/usr/bin/env python3
"""Generate SAXS and SANS coarse-grained bead form factors.

The module accepts CGBuilder mappings, identifies topologically equivalent
beads, averages Debye form-factor magnitudes over a trajectory, performs a
low-q weighted polynomial fit, and writes PLUMED parameter fragments.
"""

from __future__ import annotations

import argparse
import collections
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
from dataclasses import dataclass
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
}


@dataclass(frozen=True)
class MappingData:
    # bead_order holds unique keys: the bead name when unique, otherwise a
    # 1-based positional suffix (SC1#1, SC1#2, ...) following [ martini ] order.
    bead_order: tuple[str, ...]
    bead_atoms: Mapping[str, tuple[int, ...]]
    atom_names: Mapping[int, str]
    atom_beads: Mapping[int, tuple[str, ...]]
    atom_count: int
    # Original bead names in [ martini ] order; may repeat.
    bead_names: tuple[str, ...] = ()
    # Global (1-based) topology index of each template atom when the mapping
    # was resolved from an index (.ndx) file; None for map-only templates.
    atom_topology: tuple[int, ...] | None = None
    indexed: bool = False


@dataclass
class MoleculeResult:
    name: str
    bead_order: list[str]
    bead_names: list[str]
    groups: OrderedDict[str, list[str]]
    frames: int
    occurrences: int
    q: np.ndarray
    methods: dict[str, dict[str, Any]]
    sans_one_parameter: dict[str, float]
    warnings: list[str]
    provenance: dict[str, Any]


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


def _unique_bead_keys(names: tuple[str, ...]) -> tuple[str, ...]:
    """Give every bead a unique key: its name, or NAME#k for repeated names."""
    counts = collections.Counter(names)
    duplicated = {name for name, count in counts.items() if count > 1}
    seen: collections.Counter[str] = collections.Counter()
    keys: list[str] = []
    for name in names:
        if name in duplicated:
            seen[name] += 1
            keys.append(f"{name}#{seen[name]}")
        else:
            keys.append(name)
    return tuple(keys)


def _parse_ndx(path: str | Path) -> list[tuple[str, list[int]]]:
    """Read an index file: ordered [ name ] groups of 1-based atom numbers."""
    groups: list[tuple[str, list[int]]] = []
    name: str | None = None
    with Path(path).open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = _strip_comment(raw)
            if not line:
                continue
            match = re.fullmatch(r"\[\s*([^]]+?)\s*\]", line)
            if match:
                name = match.group(1).strip()
                groups.append((name, []))
                continue
            if name is None:
                raise ValueError(f"{path}: data before the first [ group ] header")
            try:
                groups[-1][1].extend(int(token) for token in line.split())
            except ValueError as exc:
                raise ValueError(
                    f"{path}: invalid atom index on line {line_number}"
                ) from exc
    if not groups:
        raise ValueError(f"{path}: no [ group ] sections found")
    empty = [group_name for group_name, atoms in groups if not atoms]
    if empty:
        raise ValueError(f"{path}: empty groups: {', '.join(empty)}")
    return groups


def resolve_bead_key(mapping: MappingData, value: Any) -> str:
    """Resolve a user-facing bead key (name or NAME#k) to the unique key."""
    key = str(value)
    if key in mapping.bead_order:
        return key
    positional = [
        unique
        for name, unique in zip(mapping.bead_names, mapping.bead_order, strict=True)
        if name == key
    ]
    if len(positional) == 1:
        return positional[0]
    if len(positional) > 1:
        raise ValueError(
            f"ambiguous bead name {key!r}; use the positional form such as "
            f"{positional[0]} (valid beads: {', '.join(mapping.bead_order)})"
        )
    raise ValueError(f"unknown bead {key!r}; known beads: {', '.join(mapping.bead_order)}")


def _resolve_bead_key_or_keep(mapping: MappingData, value: Any) -> str:
    """Resolve a bead key, keeping the original for non-bead labels (group names)."""
    try:
        return resolve_bead_key(mapping, value)
    except ValueError:
        return str(value)


def read_cgbuilder_mapping(
    path: str | Path, index_path: str | Path | None = None
) -> MappingData:
    """Read a CGBuilder map, including atoms shared by multiple beads.

    Bead names may repeat (one entry per residue in [ martini ]). When they do,
    ``index_path`` must point to an index file whose [ group ] sections list,
    in [ martini ] order, the 1-based topology atom numbers of every bead
    occurrence; the index is authoritative for the atom-to-bead assignment.
    """
    sections: dict[str, list[list[str]]] = {}
    section: str | None = None
    with Path(path).open() as handle:
        for _line_number, raw in enumerate(handle, start=1):
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
    bead_names = tuple(token for row in sections["martini"] for token in row)
    if not bead_names:
        raise ValueError(f"{path}: bead names must be present and unique")
    bead_order = _unique_bead_keys(bead_names)
    known_names = set(bead_names)

    bead_atoms: dict[str, list[int]]
    atom_names: dict[int, str]
    atom_beads: dict[int, tuple[str, ...]]
    atom_topology: tuple[int, ...] | None

    for row in sections["atoms"]:
        if len(row) < 3:
            raise ValueError(f"{path}: malformed [ atoms ] row: {' '.join(row)}")
        try:
            index = int(row[0])
        except ValueError as exc:
            raise ValueError(f"{path}: invalid atom index {row[0]!r}") from exc
        if index < 1:
            raise ValueError(f"{path}: duplicate or invalid atom index {index}")
        unknown = [bead for bead in row[2:] if bead not in known_names]
        if unknown:
            raise ValueError(f"{path}: unknown bead(s) {unknown} for atom {index}")

    if index_path is not None:
        groups = _parse_ndx(index_path)
        expected_names = list(bead_names)
        actual_names = [group_name for group_name, _ in groups]
        if actual_names != expected_names:
            raise ValueError(
                f"{index_path}: group names {actual_names} do not match the "
                f"[ martini ] bead order {expected_names}"
            )
        union = sorted({atom for _, atoms in groups for atom in atoms})
        rank = {atom: position + 1 for position, atom in enumerate(union)}
        atom_topology = tuple(union)
        bead_atoms = {
            key: tuple(sorted(rank[atom] for atom in atoms))
            for (_, atoms), key in zip(groups, bead_order, strict=True)
        }
        atom_beads = {}
        for (_, atoms), key in zip(groups, bead_order, strict=True):
            for atom in atoms:
                atom_beads.setdefault(rank[atom], []).append(key)
        atom_beads = {atom: tuple(keys) for atom, keys in atom_beads.items()}
        counts = collections.Counter(
            token for row in sections["atoms"] for token in row[2:]
        )
        group_sizes = collections.Counter()
        for group_name, atoms in groups:
            group_sizes[group_name] += len(atoms)
        mismatched = [
            name for name in known_names if counts.get(name, 0) != group_sizes[name]
        ]
        if mismatched:
            raise ValueError(
                f"{index_path}: atom memberships disagree with the mapping for "
                f"{', '.join(sorted(mismatched))}; the [ atoms ] section must "
                "carry one token per bead membership"
            )
        atom_names = {}
    else:
        duplicates = [
            name for name, count in collections.Counter(bead_names).items() if count > 1
        ]
        if duplicates:
            raise ValueError(
                f"{path}: bead names must be present and unique unless an index "
                f"file is supplied (duplicated: {', '.join(duplicates)})"
            )
        bead_atoms = {bead: [] for bead in bead_order}
        atom_names = {}
        atom_beads = {}
        atom_topology = None
        for row in sections["atoms"]:
            index = int(row[0])
            memberships = tuple(dict.fromkeys(row[2:]))
            atom_names[index] = row[1]
            atom_beads[index] = memberships
            for bead in memberships:
                bead_atoms[bead].append(index)

    if index_path is None:
        empty = [bead for bead, atoms in bead_atoms.items() if not atoms]
        if empty:
            raise ValueError(f"{path}: beads without atoms: {', '.join(empty)}")
        indices = sorted(atom_names)
        if indices != list(range(1, max(indices) + 1)):
            raise ValueError(f"{path}: atom indices must be contiguous and 1-based")
    else:
        indices = sorted(atom_beads)
        if indices != list(range(1, len(atom_beads) + 1)):
            raise ValueError(
                f"{index_path}: index atoms must be contiguous and 1-based"
            )
    return MappingData(
        bead_order=bead_order,
        bead_atoms={key: tuple(value) for key, value in bead_atoms.items()},
        atom_names=atom_names,
        atom_beads=atom_beads,
        atom_count=max(indices),
        bead_names=bead_names,
        atom_topology=atom_topology,
        indexed=index_path is not None,
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
            "Guessing bonds from the first trajectory frame",
            RuntimeWarning,
            stacklevel=2,
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


def _resolve_settings_bead_keys(config: Mapping[str, Any], mapping: MappingData) -> dict[str, Any]:
    """Resolve user-facing bead keys in a config copy against a mapping.

    Bead-keyed settings accept either the plain bead name (unique names only)
    or the positional NAME#k form. ``group:`` density overrides are kept as-is
    because their keys name automatic groups, which exist only after identity
    grouping. Sign crossovers and widths may also name whole groups, so their
    unresolvable keys are kept unchanged and ignored, as before.
    """
    resolved = copy.deepcopy(dict(config))
    overrides = resolved.setdefault("saxs", {}).get("electron_density_overrides", {})
    if isinstance(overrides, Mapping):
        resolved["saxs"]["electron_density_overrides"] = {
            **{
                key: value
                for key, value in overrides.items()
                if str(key).startswith("group:")
            },
            **{
                resolve_bead_key(mapping, key): value
                for key, value in overrides.items()
                if not str(key).startswith("group:")
            },
        }
    identity = resolved.setdefault("identity", {})
    force_split = identity.get("force_split", [])
    if isinstance(force_split, list | tuple):
        identity["force_split"] = [resolve_bead_key(mapping, item) for item in force_split]
    force_groups = identity.get("force_groups", {})
    if isinstance(force_groups, Mapping):
        identity["force_groups"] = {
            str(name): [resolve_bead_key(mapping, member) for member in members]
            for name, members in force_groups.items()
        }
    for leaves in (
        resolved.get("sign", {}).get("crossovers", {}),
        resolved.get("sign", {}).get("widths", {}),
    ):
        if isinstance(leaves, Mapping):
            for method, values in leaves.items():
                if isinstance(values, Mapping):
                    leaves[method] = {
                        _resolve_bead_key_or_keep(mapping, key): value
                        for key, value in values.items()
                    }
    return resolved


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


def xray_factors(
    labels: list[str],
    weights: np.ndarray,
    q: np.ndarray,
    density: float,
    volumes: Mapping[str, float],
) -> np.ndarray:
    factors = np.empty((q.size, len(labels)), dtype=float)
    missing = sorted(
        {label for label in labels if base_element(label) not in volumes}
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
        displaced = density * volume * np.exp(
            -(q**2) * volume ** (2.0 / 3.0) / (4.0 * np.pi)
        )
        factors[:, index] = weights[index] * (free_atom - displaced)
    return factors


def sans_factors(labels: list[str], weights: np.ndarray, q: np.ndarray) -> np.ndarray:
    values = weights * np.asarray([neutron_length(label) for label in labels])
    return np.broadcast_to(values, (q.size, values.size)).copy()


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
    index_path = _resolve_path(molecule.get("index"), base)
    if topology is None or mapping_path is None:
        raise ValueError(f"{name}: topology and mapping are required")
    for path in (topology, mapping_path, trajectory, index_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    mapping = read_cgbuilder_mapping(mapping_path, index_path)
    config = _resolve_settings_bead_keys(config, mapping)
    universe = mda.Universe(str(topology), *([str(trajectory)] if trajectory else []))
    trajectory_settings = config["trajectory"]
    selected_indices, occurrences = _selected_occurrences(
        universe, str(trajectory_settings["selection"]), mapping.atom_count
    )
    if mapping.atom_topology is not None:
        expected = np.asarray(mapping.atom_topology, dtype=int) - 1
        if not np.array_equal(np.asarray(occurrences[0]), expected):
            raise ValueError(
                f"{name}: the index-defined template requires the first selected "
                f"occurrence to contain topology atoms {list(mapping.atom_topology)} "
                "in that order; adjust the trajectory selection"
            )
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
    sans_one = {}
    for bead in mapping.bead_order:
        local = mapping.bead_atoms[bead]
        bead_labels = [isotopes[index - 1] for index in local]
        if config["saxs"].get("enabled", True):
            factors["saxs"][bead] = xray_factors(
                bead_labels,
                fractions[bead],
                q,
                density_overrides.get(bead, density_default),
                volumes,
            )
        factors["sans"][bead] = sans_factors(
            bead_labels, fractions[bead], q
        )
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
    frames = 0
    start = int(trajectory_settings.get("start", 0))
    stop_value = trajectory_settings.get("stop")
    stop = None if stop_value is None else int(stop_value)
    stride = int(trajectory_settings.get("stride", 1))
    if start < 0 or (stop is not None and stop <= start) or stride < 1:
        raise ValueError("invalid trajectory start/stop/stride")
    for timestep in universe.trajectory[start:stop:stride]:
        frames += 1
        box = valid_box(timestep.dimensions)
        for occurrence in occurrences:
            for bead in mapping.bead_order:
                local = np.asarray(mapping.bead_atoms[bead], dtype=int) - 1
                positions = universe.atoms[occurrence[local]].positions
                for method in enabled_methods:
                    sums[method][bead] += debye_amplitude(
                        positions, q, factors[method][bead], box
                    )
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

    provenance = {
        "topology": {"path": str(topology), "sha256": _file_sha256(topology)},
        "trajectory": {
            "path": None if trajectory is None else str(trajectory),
            "sha256": _file_sha256(trajectory),
        },
        "mapping": {"path": str(mapping_path), "sha256": _file_sha256(mapping_path)},
        "index": {
            "path": None if index_path is None else str(index_path),
            "sha256": _file_sha256(index_path),
        },
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
        bead_names=list(mapping.bead_names),
        groups=groups,
        frames=frames,
        occurrences=len(occurrences),
        q=q,
        methods=method_results,
        sans_one_parameter=sans_one,
        warnings=run_warnings,
        provenance=provenance,
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


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def result_report(result: MoleculeResult, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": result.name,
        "frames": result.frames,
        "molecule_occurrences": result.occurrences,
        "bead_order": result.bead_order,
        "bead_names": result.bead_names,
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


def write_result(
    output: Path, result: MoleculeResult, config: Mapping[str, Any], force: bool
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
    plot_fits(
        Path(f"{prefix}_form_factor_fits.png"),
        result,
        float(config["fit"]["low_q_max"]),
        force,
        bool(config.get("plot", {}).get("show_positive_magnitude", True)),
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
    _write_text(output / "plumed_example.inp", "\n".join(example) + "\n", force)

    if not any(int(molecule.get("count", 1)) != 1 for molecule in molecules):
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


def _molecule_specs(args: argparse.Namespace, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("molecules")
    if configured:
        cli_molecule_values = (args.topology, args.trajectory, args.mapping, args.name, args.index)
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
                "index": args.index,
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
    parser.add_argument(
        "--index",
        help="index (.ndx) file defining the atom membership of every bead "
        "occurrence; required when [ martini ] bead names repeat",
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config, base = load_config(args.config)
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
