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
| `*_form_factor_fits.png` | Signed curves, positive magnitudes, fits, weighted low-q region, and residuals |
| `*_curves.csv`, `*_coefficients.csv` | Machine-readable curves and coefficients |
| `*_report.yaml` | Groups, errors, crossings, input hashes, versions, and units |

With `--decompose` (or `decomposition.enabled: true`), it additionally writes:

| Output | Contents |
|---|---|
| `*_saxs_{atomic,solvent,mixed}_parameters.inp` | Separate Equation-9 SAXS terms |
| `*_sans_{h_atomic,h_mixed,d_atomic,d_mixed,solvent}_parameters.inp` | SANS H/D end-state terms |
| `*_hybrid_mapping.inp` | Explicit atom membership and normalized mass weights |
| `*_hybrid_lcpo_parameters.inp` | One atom-level LCPO tuple per selected atom |
| `*_cg_lcpo_parameters.inp` | Effective LCPO tuples for native-CG bead coordinates |
| `*_sans_exchangeable_beads.inp` | Beads containing exchangeable hydrogens |
| `*_{saxs,sans}_equation9_inline.inp` | Equivalent verbose direct PLUMED keywords |
| `*_equation9_terms.csv` | Raw term curves, polynomial reconstructions, and residuals |
| `*_{method}_{term}_fit.png` | Raw Equation-9 term, its exported polynomial fit, and fit residuals |
| `*_lcpo_exposure_samples.csv` | Hybrid/native-CG accessible fractions for every bead and sampled structure |
| `*_lcpo_exposure_correlation.png` | Full exposure scatter, bead means, identity line, cutoff, correlation, and errors |

Every term file is contextual and versioned. It declares its method, physical
term, q range, and q step before the contiguous `PARAMETERSn` rows. Project
mode writes corresponding `project_*` files with globally offset mapping atom
serials in the configured molecule/count order.

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
  --config examples/example.yaml \
  --output results/molecule
```

Generate the decomposed PLUMED interface for both hybrid atomistic and native
coarse-grained solvation modes:

```bash
form-factors-beads \
  --topology molecule.pdb \
  --trajectory trajectory.xtc \
  --mapping molecule.map \
  --config examples/example.yaml \
  --decompose \
  --representation both \
  --output results/molecule
```

`--representation hybrid` or `--representation cg` restricts the LCPO files;
the density-independent scattering terms are common to both representations.

Validate the complete calculation without writing outputs:

```bash
form-factors-beads \
  --topology molecule.pdb \
  --mapping molecule.map \
  --config examples/example.yaml \
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

## Equation-9 decomposition

The opt-in decomposition accumulates the three density-independent terms for
each bead and trajectory sample:

```text
A(q) = sum_kl a_k(q) a_l(q) sinc(q r_kl)
S(q) = sum_kl s_k(q) s_l(q) sinc(q r_kl)
M(q) = sum_kl [a_k(q)s_l(q) + s_k(q)a_l(q)] sinc(q r_kl)

F(q,rho)^2 = A(q) + rho^2 S(q) - rho M(q)
```

For SAXS, `a` is in electrons, `s` in cubic ångström, and `rho` in
`e Å^-3`. For SANS, the corresponding atomic unit is femtometres and solvent
SLD is in `fm Å^-3`. Coefficients are always constant-first and q is in
`Å^-1`. The exact q=0 identities are enforced before fitting. The report
compares reconstructed magnitudes with the legacy form-factor path and records
term-specific fit errors and sign-crossing diagnostics.

SANS export contains protiated and exchanged atomic/mixed end states plus one
shared solvent term. Exchangeable hydrogens are detected from bonds as H bound
to N, O, or S; `decomposition.exchange.include` and `exclude` handle unusual
chemistry. The tool does not choose a stochastic exchange realization—that is
a run-time PLUMED responsibility.

## LCPO parameters and resolution modes

Hybrid export uses the residue/atom LCPO table extracted from the current
PLUMED fork. Recognized protein and nucleic-acid atoms therefore retain the
existing radii and coefficients. Hydrogen centers are disabled. Nonstandard
atoms of C, N, O, P, S, F, and Mg use documented element-fallback tuples unless
fallback is disabled. A warning is issued whenever any such fallback is used,
and every assignment source appears in the YAML report. Atom, element, and bead
overrides are available for deliberate parameterization.

The default fallback tuples are `radius,p1,p2,p3,p4` in the existing PLUMED
LCPO units:

| Element | Generic tuple |
|---|---|
| C | `1.70, 0.56482, -0.19608, -0.0010219, 0.0002658` |
| N | `1.65, 0.41102, -0.12254, -0.000075448, 0.00011804` |
| O | `1.60, 0.68563, -0.18680, -0.00135573, 0.00023743` |
| S | `1.90, 0.54581, -0.19477, -0.0012873, 0.00029247` |
| P | `1.90, 0.03873, -0.0089339, 0.0000083582, 0.0000030381` |
| F | `1.47, 0.68563, -0.18680, -0.00135573, 0.00023743` |
| Mg | `1.18, 0.49392, -0.16038, -0.00015512, 0.00016453` |

The F and Mg entries follow Amber's established LCPO implementation. Their
overlap coefficients are borrowed from oxygen atom types rather than fitted as
independent elemental parameterizations, so the same fallback warning applies.
No automatic parameters are assigned to other elements.

Native-CG tuples use the sphere radius corresponding to the bead's summed,
shared-atom-adjusted displaced volume. The four coefficients are
isolated-area-weighted averages of the constituent atom tuples. These are
effective parameters, not universal bead constants. During generation the tool
evaluates the same LCPO expression on atomistic and bead centers and
reports accessible-fraction correlation, the exposed/buried confusion matrix,
the mismatch fraction, and the largest bead discrepancies at
`decomposition.diagnostics.accessible_fraction_cutoff` (default `0.2`). This
cutoff is only a diagnostic threshold; it does not affect generated
coefficients, radii, or form factors. Output radii are base radii in ångström;
the fixed `1.4 Å` water probe is added by PLUMED.

Diagnostic plots draw the signed form factor as a solid line, its polynomial
fit as a dashed line, and the original non-negative `sqrt(I(q))` magnitude as a
dotted line in the same bead-group color. Set
`plot.show_positive_magnitude: false` to hide the dotted reference curves.
When Equation-9 decomposition is enabled, the tool also writes one fit figure
for every exported parameter set (three for SAXS and five for SANS). Each figure
compares the raw atomic, solvent, or mixed term with the polynomial reconstructed
from its seven coefficients and shows `fit - value` in a residual panel.

## YAML configuration

See [`examples/example.yaml`](examples/example.yaml) for a commented reference
covering every configuration key. A
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
graph identity, low-q weighting, exact `F(0)`, Equation-9 reconstruction and
q=0 identities, LCPO assignment and geometry, H/D exchange detection,
versioned-file round trips, project expansion, non-clobbering outputs,
packaged data, and minimal legacy/decomposed end-to-end calculations.

The self-contained TX9 scientific regression compares all 15 bead curves and
the grouped low-q polynomials against curated known-good reference data. Its
12-frame stratified trajectory fixture keeps the test fast enough for CI while
preserving the low-q regression accuracy. Run it with:

```bash
python -m pytest -m integration
```

The same integration marker also covers
`tests/data/equation9_fixture`: a manifest-hashed, nonstandard molecule with a
shared mapping atom and an exchangeable O–H site. Its versioned Equation-9,
mapping, and LCPO files are the tool-owned acceptance fixture intended to be
copied unchanged into the future PLUMED consumer regression.

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
