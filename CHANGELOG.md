# Changelog

## Unreleased

- Allow repeated bead names in CGBuilder `[ martini ]` mappings so one
  amino-acid mapping can cover every residue. Each occurrence becomes a
  distinct positional bead (`NAME#k`); a CGBuilder index (`.ndx`) file must
  be supplied via the new `--index` option (or a molecule `index:` entry)
  and is authoritative for the atom-to-bead assignment.
- Group CSV/report/plot keys, `force_split`/`force_groups`, sign
  crossovers/widths, and SAXS density overrides resolve positional bead
  keys; plain names remain valid when unique.
- Reports record the original `bead_names` order alongside the unique
  `bead_order`, and provenance captures the index file.
- PLUMED fragments keep exactly one `PARAMETERSn`/`SCATLENn` line per
  `[ martini ]` bead, in mapping order.

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
