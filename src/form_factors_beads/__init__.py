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
    sans_factors,
    write_result,
    xray_factors,
)

try:
    __version__ = version("form-factors-beads")
except PackageNotFoundError:  # source checkout without installation
    __version__ = "0+unknown"

__all__ = [
    "DEFAULT_CONFIG",
    "MappingData",
    "MoleculeResult",
    "calculate_molecule",
    "debye_amplitude",
    "fit_polynomial",
    "group_equivalent_beads",
    "load_atomic_volumes",
    "load_config",
    "neutron_length",
    "q_grid",
    "read_cgbuilder_mapping",
    "restore_amplitude_sign",
    "sans_factors",
    "write_result",
    "xray_factors",
]
