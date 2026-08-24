"""Atomistic-to-bead SAXS and SANS form-factor generation."""

from importlib.metadata import PackageNotFoundError, version

from .core import (
    DEFAULT_CONFIG,
    MappingData,
    MoleculeResult,
    calculate_molecule,
    debye_amplitude,
    fit_polynomial,
    group_equivalent_beads,
    load_atomic_volumes,
    load_config,
    neutron_length,
    q_grid,
    read_cgbuilder_mapping,
    restore_amplitude_sign,
    sans_component_factors,
    sans_factors,
    write_decomposition_outputs,
    write_result,
    xray_component_factors,
    xray_factors,
)
from .decomposition import (
    LCPOAssignment,
    aggregate_hybrid_accessible_fractions,
    assign_lcpo_parameters,
    debye_component_terms,
    derive_cg_lcpo_parameters,
    detect_exchangeable_hydrogens,
    equation9_radicand,
    lcpo_areas,
    load_lcpo_database,
    parse_contextual_parameter_file,
    virtual_center_positions,
)

try:
    __version__ = version("form-factors-beads")
except PackageNotFoundError:  # source checkout without installation
    __version__ = "0+unknown"

__all__ = [
    "DEFAULT_CONFIG",
    "LCPOAssignment",
    "MappingData",
    "MoleculeResult",
    "aggregate_hybrid_accessible_fractions",
    "assign_lcpo_parameters",
    "calculate_molecule",
    "debye_amplitude",
    "debye_component_terms",
    "derive_cg_lcpo_parameters",
    "detect_exchangeable_hydrogens",
    "equation9_radicand",
    "fit_polynomial",
    "group_equivalent_beads",
    "load_atomic_volumes",
    "load_config",
    "load_lcpo_database",
    "lcpo_areas",
    "neutron_length",
    "q_grid",
    "parse_contextual_parameter_file",
    "read_cgbuilder_mapping",
    "restore_amplitude_sign",
    "sans_component_factors",
    "sans_factors",
    "write_decomposition_outputs",
    "write_result",
    "virtual_center_positions",
    "xray_component_factors",
    "xray_factors",
]
