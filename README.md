# form-factors-beads

Generate coarse-grained bead form factors directly from an atomistic molecular
trajectory and a [CGBuilder](https://jbarnoud.github.io/cgbuilder/) mapping.
The package calculates both SAXS and SANS form factors, discovers chemically
equivalent beads from bonded topology, averages their curves, fits PLUMED-ready
polynomials, and records enough provenance to reproduce the calculation.

## What it produces

For every mapped molecule type, the command writes:

| Output | Contents |
|---|---|
| `*_saxs_parameters.inp` | SAXS `PARAMETERSn=c0,...,c6` fragment |
| `*_sans_parameters.inp` | Geometry-dependent SANS polynomial fragment |
| `*_sans_scattering_lengths.inp` | One-parameter SANS `SCATLENn=b` fragment |
| `*_form_factor_fits.png` | Curves, fits, weighted low-q region, and residuals |
| `*_curves.csv`, `*_coefficients.csv` | Machine-readable curves and coefficients |
| `*_report.yaml` | Groups, errors, crossings, input hashes, versions, and units |

Project configurations can also expand the fragments for a complete mixture in
the requested molecule order.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick start

The topology must contain bonds. The trajectory is optional when the topology
already contains the coordinates to analyze.

```bash
form-factors-beads \
  --topology molecule.pdb \
  --trajectory trajectory.xtc \
  --mapping molecule.map \
  --config examples/basic.yaml \
  --output results/molecule
```

Validate the complete calculation without writing outputs:

```bash
form-factors-beads \
  --topology molecule.pdb \
  --mapping molecule.map \
  --config examples/basic.yaml \
  --dry-run
```

The output directory is non-clobbering: an existing non-empty directory is
rejected unless `--force` is explicitly supplied.

## Mapping and automatic bead identity

The parser reads the CGBuilder `[ martini ]` and `[ atoms ]` sections. An atom
may be assigned to more than one bead. Its scattering amplitude is then split
equally among those memberships, preserving the molecule-wide value of
`F(0)`.

Beads are compared as labeled molecular graphs. Identity includes:

- elements and SANS isotope labels;
- internal bonds and bond orders when available;
- bonded neighbors just outside the mapped bead;
- shared-atom fractions;
- SAXS solvent-contrast class.

A Weisfeiler–Lehman hash is used as a fast prefilter, followed by exact graph
isomorphism. YAML `force_groups` and `force_split` settings are available when
the desired coarse-grained equivalence intentionally differs from atomistic
topology.

## Form factors and polynomial fitting

For each sampled conformation, the package evaluates the orientationally
averaged Debye expression

```text
I(q) = sum_i sum_j f_i(q) f_j(q) sinc(q r_ij)
```

and reconstructs the bead amplitude from `sqrt(I(q))`. SAXS atomic factors and
coherent neutron scattering lengths come from `periodictable`. SAXS includes
the Fraser solvent-displacement correction. The bundled displaced-volume table
contains sourced H, D, C, N, O, and S values; an unsupported element is an
explicit error until its volume is provided in YAML.

Negative-contrast amplitudes require a sign convention because intensity alone
does not determine the sign. By default, the transition is placed at the
minimum magnitude. For sensitive systems, set the crossover explicitly in the
configuration so a one-q-grid-point shift cannot change the fit.

The default polynomial is sixth order and uses fivefold residual weight below
`q = 0.75 A^-1`. Its constant is constrained to the analytically calculated
`F(0)`:

```text
F_SAXS(0) = sum_i w_i [f_i(0) - rho_s v_i]
F_SANS(0) = sum_i w_i b_i
```

The report also records the unconstrained extrapolation and the low-q RMSE and
maximum error.

## YAML configuration

See [`examples/basic.yaml`](examples/basic.yaml) for all common settings. A
project file may define several molecule types:

```yaml
molecules:
  - name: surfactant_a
    topology: structures/a.pdb
    trajectory: trajectories/a.xtc
    mapping: mappings/a.map
    count: 100
  - name: surfactant_b
    topology: structures/b.pdb
    trajectory: trajectories/b.xtc
    mapping: mappings/b.map
    count: 50
```

Paths in a project file are resolved relative to that YAML file. Per-molecule
`settings` may override global fit, contrast, isotope, or sign settings.

## Python API

The computational pieces are importable for notebooks and custom workflows:

```python
from pathlib import Path

from form_factors_beads import calculate_molecule, load_config

config, base = load_config(Path("project.yaml"))
result = calculate_molecule(config["molecules"][0], config, base)
print(result.groups)
print(result.methods["saxs"]["coefficients"])
```

## Development

```bash
python -m pytest
python -m build
```

Tests cover mapping validation, shared-atom conservation, isotope handling,
graph identity, low-q weighting, exact `F(0)`, non-clobbering outputs, packaged
data, and a minimal end-to-end molecular calculation.

The self-contained TX9 scientific regression compares all 15 bead curves and
the grouped low-q polynomials against curated known-good reference data. Its
12-frame stratified trajectory fixture keeps the test fast enough for CI while
preserving the low-q regression accuracy. Run it with:

```bash
python -m pytest -m integration
```

Run only the portable unit suite with:

```bash
python -m pytest -m "not integration"
```

This is an early scientific release. Validate sign crossings, solvent contrast,
and fit residuals for each new molecular family before using generated
parameters in production simulations.

## License

This project is licensed under the GNU Lesser General Public License v3.0 or
later. See [LICENSE](LICENSE) for the complete terms.
