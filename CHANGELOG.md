# Changelog

## Unreleased

- Add unit coverage for scientific edge cases and a provenance-pinned TX9 SAXS
  integration regression against `get_FF_beads.ipynb` outputs.
- Make the TX9 integration regression CI-ready with a self-contained,
  12-frame stratified trajectory fixture and GitHub Actions coverage.

## 0.1.0 — 2026-08-20

- Package the atomistic-to-bead SAXS/SANS workflow as a Python library and CLI.
- Support CGBuilder mappings, shared atoms, bonded-graph bead identity, SAXS,
  polynomial SANS, and one-parameter SANS outputs.
- Add constrained `F(0)` fitting, low-q residual weighting, PLUMED fragments,
  figures, machine-readable reports, and provenance capture.
