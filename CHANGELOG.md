# Changelog

## Unreleased

- Add per-term Equation-9 fit and residual plots for every exported SAXS and
  SANS coefficient set.
- Write the underlying hybrid/native-CG exposure pairs and an automatic LCPO
  correlation plot for every decomposed calculation.
- Use one native-CG LCPO radius construction based on summed,
  shared-atom-adjusted displaced volume; the accessible-fraction cutoff remains
  diagnostic only.
- Rename the diagnostic threshold to
  `decomposition.diagnostics.accessible_fraction_cutoff` so it cannot be
  confused with PLUMED's absolute-area `SASA_CUTOFF`.
- Extend automatic LCPO element fallbacks to F and Mg using the established
  Amber tuples, and warn whenever any element fallback is selected.
- Add opt-in Equation-9 SAXS/SANS decomposition into separately fitted atomic,
  displaced-solvent, and mixed terms with versioned PLUMED parameter files.
- Export explicit hybrid mappings, exact/fallback atom LCPO tuples, derived
  native-CG LCPO tuples, and hybrid-versus-CG exposure diagnostics.
- Generate SANS protiated/deuterated end states from bond-aware exchangeable-H
  detection with include/exclude overrides.
- Add contextual-file validation, expanded project outputs, scientific unit and
  end-to-end tests, and complete configuration/user documentation.

## 0.1.0 — 2026-08-20

- Package the atomistic-to-bead SAXS/SANS workflow as a Python library and CLI.
- Support CGBuilder mappings, shared atoms, bonded-graph bead identity, SAXS,
  polynomial SANS, and one-parameter SANS outputs.
- Add constrained `F(0)` fitting, low-q residual weighting, PLUMED fragments,
  figures, machine-readable reports, and provenance capture.

- Add unit coverage for scientific edge cases and a provenance-pinned TX9 SAXS
  integration regression against `get_FF_beads.ipynb` outputs.
- Make the TX9 integration regression CI-ready with a self-contained,
  12-frame stratified trajectory fixture and GitHub Actions coverage.
- Prepare the README for public use and license the project under LGPL v3.0 or
  later.
- Rename the fit-difference plot axis to `residuals` and replace the basic YAML
  sample with a comprehensive, commented configuration reference.
- Reject `--name`, like the other single-molecule CLI arguments, when a YAML
  project defines `molecules`.
- Plot the original positive form-factor magnitude by default alongside the
  signed curve and polynomial fit, with a YAML switch to hide it.
