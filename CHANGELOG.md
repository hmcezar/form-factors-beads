# Changelog

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
