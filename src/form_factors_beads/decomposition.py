"""Equation-9 decomposition, LCPO assignment, and PLUMED export helpers."""

from __future__ import annotations

import copy
import math
import re
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from MDAnalysis.lib.distances import distance_array, minimize_vectors

PACKAGE_DIR = Path(__file__).resolve().parent
LCPO_PARAMETER_FILE = PACKAGE_DIR / "data" / "lcpo_parameters.yaml"

# Pragmatic element fallbacks from the PLUMED database and Amber's LCPO
# implementation. Exact residue/atom assignments always take priority.
GENERIC_LCPO: dict[str, tuple[float, float, float, float, float]] = {
    "C": (1.70, 0.56482, -0.19608, -0.0010219, 0.0002658),
    "N": (1.65, 0.41102, -0.12254, -0.000075448, 0.00011804),
    "O": (1.60, 0.68563, -0.18680, -0.00135573, 0.00023743),
    "S": (1.90, 0.54581, -0.19477, -0.0012873, 0.00029247),
    "P": (1.90, 0.03873, -0.0089339, 0.0000083582, 0.0000030381),
    "F": (1.47, 0.68563, -0.18680, -0.00135573, 0.00023743),
    "Mg": (1.18, 0.49392, -0.16038, -0.00015512, 0.00016453),
}

ALLOWED_TERMS = {
    "SAXS": {"ATOMIC", "SOLVENT", "MIXED"},
    "SANS": {"ATOMIC_H", "MIXED_H", "ATOMIC_D", "MIXED_D", "SOLVENT"},
}


@dataclass(frozen=True)
class LCPOAssignment:
    """One serialized LCPO tuple and the reason it was selected."""

    values: tuple[float, float, float, float, float]
    source: str
    identifier: str


def _lcpo_tuple(value: Sequence[Any], description: str) -> tuple[float, ...]:
    if len(value) != 5:
        raise ValueError(f"{description} must contain radius,p1,p2,p3,p4")
    result = tuple(float(item) for item in value)
    if not np.all(np.isfinite(result)) or result[0] < 0.0:
        raise ValueError(f"{description} contains invalid LCPO values")
    return result


def load_lcpo_database() -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
    """Load the exact residue/atom LCPO tuples extracted from PLUMED."""
    with LCPO_PARAMETER_FILE.open() as handle:
        document = yaml.safe_load(handle)
    parameters = {
        str(key): _lcpo_tuple(value, str(key))
        for key, value in document["parameters"].items()
    }
    provenance = {
        "source": copy.deepcopy(document["source"]),
        "probe_radius_A": float(document["probe_radius_A"]),
        "entry_count": len(parameters),
    }
    return parameters, provenance


def assign_lcpo_parameters(
    atom_names: Sequence[str],
    residue_names: Sequence[str],
    elements: Sequence[str],
    settings: Mapping[str, Any],
) -> tuple[list[LCPOAssignment], dict[str, Any]]:
    """Assign exact PLUMED tuples, explicit overrides, or element fallbacks."""
    if not (len(atom_names) == len(residue_names) == len(elements)):
        raise ValueError("LCPO atom, residue, and element arrays must have equal length")
    database, database_provenance = load_lcpo_database()
    atom_overrides = {int(key): value for key, value in settings.get("atom_overrides", {}).items()}
    element_overrides = {
        str(key).capitalize(): value
        for key, value in settings.get("element_overrides", {}).items()
    }
    unknown_indices = set(atom_overrides) - set(range(1, len(elements) + 1))
    if unknown_indices:
        raise ValueError(
            "LCPO atom override indices are outside the mapping: "
            f"{sorted(unknown_indices)}"
        )

    assignments: list[LCPOAssignment] = []
    fallback_atoms: list[tuple[int, str, str]] = []
    for index, (atom_name, residue_name, element) in enumerate(
        zip(atom_names, residue_names, elements, strict=True), start=1
    ):
        chemical_element = "H" if element in {"D", "T"} else element
        identifier = f"{residue_name}_{atom_name}"
        if index in atom_overrides:
            values = _lcpo_tuple(atom_overrides[index], f"LCPO atom override {index}")
            source = "atom_override"
        elif chemical_element in element_overrides:
            values = _lcpo_tuple(
                element_overrides[chemical_element],
                f"LCPO element override {chemical_element}",
            )
            source = "element_override"
        elif identifier in database:
            values = database[identifier]
            source = "plumed_exact"
        elif chemical_element == "H":
            values = (0.0, 0.0, 0.0, 0.0, 0.0)
            source = "hydrogen_excluded"
        elif (
            settings.get("allow_element_fallback", True)
            and chemical_element in GENERIC_LCPO
        ):
            values = GENERIC_LCPO[chemical_element]
            source = "element_fallback"
            fallback_atoms.append((index, identifier, chemical_element))
        else:
            raise ValueError(
                f"No LCPO parameters for atom {index} ({identifier}, element {element}); "
                "provide decomposition.lcpo.atom_overrides or element_overrides"
            )
        assignments.append(LCPOAssignment(values=values, source=source, identifier=identifier))

    if fallback_atoms:
        element_counts = Counter(element for _, _, element in fallback_atoms)
        counts_text = ", ".join(
            f"{element}={count}" for element, count in sorted(element_counts.items())
        )
        examples = ", ".join(
            f"{index}:{identifier}/{element}"
            for index, identifier, element in fallback_atoms[:5]
        )
        if len(fallback_atoms) > 5:
            examples += ", ..."
        warnings.warn(
            "Using automatic LCPO element-fallback parameters for "
            f"{len(fallback_atoms)} atom(s) ({counts_text}); atoms: {examples}. "
            "These are approximate element defaults, not exact residue/atom "
            "assignments. Use decomposition.lcpo.atom_overrides or "
            "element_overrides when system-specific parameters are available.",
            UserWarning,
            stacklevel=2,
        )

    # Match the terminal adjustments made after PLUMED's exact database lookup.
    if assignments and atom_names[0] == "N" and assignments[0].source == "plumed_exact":
        radius = assignments[0].values[0]
        assignments[0] = LCPOAssignment(
            (radius, 0.73511, -0.22116, -0.00089148, 0.00025230),
            "plumed_exact_n_terminus",
            assignments[0].identifier,
        )
    if assignments and atom_names[-1] == "O" and assignments[-1].source == "plumed_exact":
        radius = assignments[-1].values[0]
        assignments[-1] = LCPOAssignment(
            (radius, 0.88857, -0.33421, -0.0018683, 0.00049372),
            "plumed_exact_c_terminus",
            assignments[-1].identifier,
        )

    counts: dict[str, int] = {}
    for assignment in assignments:
        counts[assignment.source] = counts.get(assignment.source, 0) + 1
    return assignments, {"database": database_provenance, "assignment_counts": counts}


def derive_cg_lcpo_parameters(
    bead_order: Sequence[str],
    bead_atoms: Mapping[str, Sequence[int]],
    atom_memberships: Mapping[int, Sequence[str]],
    elements: Sequence[str],
    atomic_volumes: Mapping[str, float],
    atom_assignments: Sequence[LCPOAssignment],
    settings: Mapping[str, Any],
) -> tuple[dict[str, LCPOAssignment], dict[str, Any]]:
    """Derive pragmatic bead LCPO tuples from displaced volumes and atom tuples."""
    probe = float(settings.get("probe_radius_A", 1.4))
    bead_overrides = settings.get("bead_overrides", {})
    unknown = set(bead_overrides) - set(bead_order)
    if unknown:
        raise ValueError(f"LCPO bead overrides contain unknown beads: {sorted(unknown)}")
    derived: dict[str, LCPOAssignment] = {}
    details: dict[str, Any] = {}
    for bead in bead_order:
        if bead in bead_overrides:
            values = _lcpo_tuple(bead_overrides[bead], f"LCPO bead override {bead}")
            derived[bead] = LCPOAssignment(values, "bead_override", bead)
            details[bead] = {"source": "bead_override"}
            continue
        volume = 0.0
        weighted_coefficients = np.zeros(4, dtype=float)
        area_weight = 0.0
        for atom in bead_atoms[bead]:
            fraction = 1.0 / len(atom_memberships[atom])
            element = elements[atom - 1]
            if element not in atomic_volumes:
                raise ValueError(f"No displaced volume for LCPO bead atom element {element}")
            volume += fraction * float(atomic_volumes[element])
            assignment = atom_assignments[atom - 1]
            radius = assignment.values[0]
            if radius <= 0.0:
                continue
            isolated_area = 4.0 * math.pi * (radius + probe) ** 2
            weight = fraction * isolated_area
            weighted_coefficients += weight * np.asarray(assignment.values[1:])
            area_weight += weight
        if volume <= 0.0 or area_weight <= 0.0:
            raise ValueError(f"Cannot derive a CG LCPO tuple for bead {bead}")
        coefficients = weighted_coefficients / area_weight
        radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        values = (radius, *coefficients.tolist())
        derived[bead] = LCPOAssignment(values, "derived_from_atoms", bead)
        details[bead] = {
            "source": "derived_from_atoms",
            "volume_equivalent_radius_A": radius,
            "split_displaced_volume_A3": volume,
            "isolated_area_weight_A2": area_weight,
        }
    return derived, details


def detect_exchangeable_hydrogens(
    elements: Sequence[str],
    bonds: Mapping[tuple[int, int], str],
    include: Sequence[Any] = (),
    exclude: Sequence[Any] = (),
) -> tuple[int, ...]:
    """Find hydrogens bonded to N, O, or S, with explicit index overrides."""
    neighbors: dict[int, set[int]] = {index: set() for index in range(1, len(elements) + 1)}
    for left, right in bonds:
        neighbors[left].add(right)
        neighbors[right].add(left)
    automatic = {
        index
        for index, element in enumerate(elements, start=1)
        if element == "H"
        and any(elements[neighbor - 1] in {"N", "O", "S"} for neighbor in neighbors[index])
    }
    included = {int(value) for value in include}
    excluded = {int(value) for value in exclude}
    for index in included | excluded:
        if not 1 <= index <= len(elements):
            raise ValueError(f"exchangeable-H override index {index} is outside the mapping")
        if elements[index - 1] != "H":
            raise ValueError(f"exchangeable-H override index {index} is not hydrogen")
    return tuple(sorted((automatic | included) - excluded))


def debye_component_terms(
    positions: np.ndarray,
    q: np.ndarray,
    atomic_factors: np.ndarray,
    solvent_factors: np.ndarray,
    box: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Calculate the atomic, solvent, and mixed terms of Equation 9."""
    distances = distance_array(positions, positions, box=box)
    sinc_qr = np.sinc(q[:, None, None] * distances[None, :, :] / np.pi)
    atomic = np.einsum(
        "qi,qij,qj->q", atomic_factors, sinc_qr, atomic_factors, optimize=True
    )
    solvent = np.einsum(
        "qi,qij,qj->q", solvent_factors, sinc_qr, solvent_factors, optimize=True
    )
    mixed = 2.0 * np.einsum(
        "qi,qij,qj->q", atomic_factors, sinc_qr, solvent_factors, optimize=True
    )
    atomic[0] = atomic_factors[0].sum() ** 2
    solvent[0] = solvent_factors[0].sum() ** 2
    mixed[0] = 2.0 * atomic_factors[0].sum() * solvent_factors[0].sum()
    return {"atomic": atomic, "solvent": solvent, "mixed": mixed}


def equation9_radicand(terms: Mapping[str, np.ndarray], density: float) -> np.ndarray:
    """Reconstruct the squared Equation-9 amplitude for one solvent density."""
    value = (
        np.asarray(terms["atomic"])
        + density**2 * np.asarray(terms["solvent"])
        - density * np.asarray(terms["mixed"])
    )
    scale = np.maximum(
        np.abs(terms["atomic"])
        + density**2 * np.abs(terms["solvent"])
        + abs(density) * np.abs(terms["mixed"]),
        1.0,
    )
    materially_negative = value < -1.0e-12 * scale
    if np.any(materially_negative):
        index = int(np.flatnonzero(materially_negative)[0])
        raise ValueError(f"Equation-9 radicand is negative at q-grid index {index}")
    return np.maximum(value, 0.0)


def parse_contextual_parameter_file(path: str | Path) -> dict[str, Any]:
    """Parse and validate a version-1 Equation-9 coefficient file."""
    metadata: dict[str, str] = {}
    rows: dict[int, tuple[float, ...]] = {}
    with Path(path).open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
            key, value = (part.strip() for part in line.split("=", 1))
            match = re.fullmatch(r"PARAMETERS(\d+)", key)
            if match:
                index = int(match.group(1))
                if index in rows:
                    raise ValueError(f"{path}: duplicate PARAMETERS{index}")
                try:
                    coefficients = tuple(float(item) for item in value.split(","))
                except ValueError as exc:
                    raise ValueError(f"{path}: invalid PARAMETERS{index}") from exc
                if not coefficients or not np.all(np.isfinite(coefficients)):
                    raise ValueError(f"{path}: invalid PARAMETERS{index}")
                rows[index] = coefficients
            else:
                if key in metadata:
                    raise ValueError(f"{path}: duplicate {key}")
                metadata[key] = value
    required = {"FORMAT_VERSION", "METHOD", "TERM", "Q_MIN", "Q_MAX", "Q_STEP"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"{path}: missing metadata {sorted(missing)}")
    if metadata["FORMAT_VERSION"] != "1":
        raise ValueError(f"{path}: unsupported FORMAT_VERSION")
    method = metadata["METHOD"].upper()
    term = metadata["TERM"].upper()
    if method not in ALLOWED_TERMS or term not in ALLOWED_TERMS[method]:
        raise ValueError(f"{path}: incompatible METHOD={method} TERM={term}")
    try:
        q_min, q_max, q_step = (
            float(metadata[key]) for key in ("Q_MIN", "Q_MAX", "Q_STEP")
        )
    except ValueError as exc:
        raise ValueError(f"{path}: invalid q metadata") from exc
    if q_min < 0.0 or q_max < q_min or q_step <= 0.0:
        raise ValueError(f"{path}: invalid q range")
    if not rows or sorted(rows) != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: PARAMETERS rows must be contiguous and one-based")
    return {
        "format_version": 1,
        "method": method,
        "term": term,
        "q_min_A^-1": q_min,
        "q_max_A^-1": q_max,
        "q_step_A^-1": q_step,
        "parameters": [rows[index] for index in range(1, len(rows) + 1)],
    }


def normalized_mass_weights(masses: Sequence[float]) -> tuple[float, ...]:
    values = np.asarray(masses, dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("mapping-center masses must be finite and positive")
    return tuple((values / values.sum()).tolist())


def virtual_center_positions(
    positions: np.ndarray,
    bead_order: Sequence[str],
    bead_atoms: Mapping[str, Sequence[int]],
    bead_weights: Mapping[str, Sequence[float]],
    box: np.ndarray | None,
) -> np.ndarray:
    """Construct PBC-safe weighted bead centers from template-local atoms."""
    centers = np.empty((len(bead_order), 3), dtype=float)
    for bead_index, bead in enumerate(bead_order):
        indices = np.asarray(bead_atoms[bead], dtype=int) - 1
        selected = np.asarray(positions[indices], dtype=float)
        weights = np.asarray(bead_weights[bead], dtype=float)
        if len(selected) != len(weights):
            raise ValueError(f"mapping weights do not match atoms for bead {bead}")
        anchor = selected[0]
        vectors = selected - anchor
        if box is not None:
            vectors = minimize_vectors(vectors, box)
        centers[bead_index] = anchor + np.einsum("i,ij->j", weights, vectors)
    return centers


def lcpo_areas(
    positions: np.ndarray,
    assignments: Sequence[LCPOAssignment],
    box: np.ndarray | None,
    probe_radius_A: float = 1.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the current PLUMED LCPO expression in square ångström."""
    if len(positions) != len(assignments):
        raise ValueError("LCPO positions and parameter tuples must have equal length")
    base_radii = np.asarray([assignment.values[0] for assignment in assignments], dtype=float)
    active = base_radii > 0.0
    radii = np.where(active, base_radii + probe_radius_A, 0.0)
    distances = distance_array(positions, positions, box=box)
    neighbors = [
        [
            j
            for j in range(len(assignments))
            if i != j and active[j] and distances[i, j] < radii[i] + radii[j]
        ]
        if active[i]
        else []
        for i in range(len(assignments))
    ]
    areas = np.zeros(len(assignments), dtype=float)
    isolated = np.zeros(len(assignments), dtype=float)
    for i, assignment in enumerate(assignments):
        if not active[i] or assignment.values[1] <= 0.0:
            continue
        ri = radii[i]
        surface = 4.0 * math.pi * ri**2
        isolated[i] = assignment.values[1] * surface
        area_ij = 0.0
        area_jk = 0.0
        area_ijk = 0.0
        neighbor_set = set(neighbors[i])
        for j in neighbors[i]:
            dij = distances[i, j]
            if dij <= np.finfo(float).eps:
                raise ValueError("LCPO centers cannot occupy the same position")
            rj = radii[j]
            aij = 2.0 * math.pi * ri * (
                ri - dij / 2.0 - (ri**2 - rj**2) / (2.0 * dij)
            )
            ajk = 0.0
            for k in neighbors[j]:
                if k not in neighbor_set:
                    continue
                djk = distances[j, k]
                if djk <= np.finfo(float).eps:
                    raise ValueError("LCPO centers cannot occupy the same position")
                rk = radii[k]
                ajk += 2.0 * math.pi * rj * (
                    rj - djk / 2.0 - (rj**2 - rk**2) / (2.0 * djk)
                )
            area_ijk += aij * ajk
            area_ij += aij
            area_jk += ajk
        p1, p2, p3, p4 = assignment.values[1:]
        areas[i] = max(p1 * surface + p2 * area_ij + p3 * area_jk + p4 * area_ijk, 0.0)
    return areas, isolated


def aggregate_hybrid_accessible_fractions(
    areas: np.ndarray,
    isolated: np.ndarray,
    bead_order: Sequence[str],
    bead_atoms: Mapping[str, Sequence[int]],
    atom_memberships: Mapping[int, Sequence[str]],
) -> dict[str, float]:
    """Aggregate atom LCPO areas to beads, splitting shared atoms equally."""
    fractions: dict[str, float] = {}
    for bead in bead_order:
        numerator = 0.0
        denominator = 0.0
        for atom in bead_atoms[bead]:
            split = 1.0 / len(atom_memberships[atom])
            numerator += split * float(areas[atom - 1])
            denominator += split * float(isolated[atom - 1])
        fractions[bead] = 0.0 if denominator <= 0.0 else numerator / denominator
    return fractions
