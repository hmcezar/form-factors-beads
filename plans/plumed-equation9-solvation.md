# Plan: PLUMED Equation-9 scattering and automatic solvation

## Purpose and boundary

Extend the SAXS/SANS PLUMED action to consume decomposed bead form factors, update binary solvation states from LCPO exposure, and evaluate Equation 9 during a simulation. This repository owns parsing, validation, run-time exposure updates, intensity evaluation, derivatives, and force propagation. Parameter derivation and file generation belong to `form-factors-beads` and are planned separately in [tool-equation9-solvation.md](tool-equation9-solvation.md).

The implementation must support both:

- **Hybrid resolution:** `ATOMS` are atomistic simulation atoms; virtual scattering centers are constructed from a mapping, and atom-level LCPO exposure is accumulated onto them.
- **Native coarse graining:** `ATOMS` are the scattering beads; LCPO runs directly on those coordinates.

The existing LCPO calculation is reused for both paths. The new work should extract it into a parameter-driven helper, not introduce a second SASA method.

## Shared compatibility contract (version 1)

This section is normative and is duplicated in the tool plan. A change here requires the matching change there before either implementation is merged.

### Equation and units

For bead `i`, evaluate

```text
F_i(q, rho) = sigma_i(q, rho) sqrt(A_i(q) + rho^2 S_i(q) - rho M_i(q))
```

where `A` is the atomic/self term, `S` the displaced-solvent term, `M` the mixed term, and `rho` the solvent scattering-length density.

- `q` is always in inverse ångström (`Å^-1`). Polynomial coefficients are ordered constant-first: `c0,c1,...`.
- SAXS uses electron units: `A [e^2]`, `S [Å^6]`, `M [e Å^3]`, and `rho [e Å^-3]`.
- SANS uses femtometres: `A [fm^2]`, `S [Å^6]`, `M [fm Å^3]`, and `rho [fm Å^-3]`.
- LCPO tuples use the units of the existing PLUMED LCPO implementation: base radius in ångström, areas in square ångström, and the corresponding coefficient dimensions required by the LCPO expression. PLUMED adds the fixed `1.4 Å` (`0.14 nm`) water probe; serialized radii do not include it.
- Mapping atom identifiers are global, one-based PLUMED atom serials.
- Hybrid accessible fractions are aggregated as conserved accessible area over
  conserved isolated area, splitting shared atoms equally. Mapping mass weights
  locate virtual centers and are not applied a second time to physical areas.

### Contextual coefficient files

Each physical term is a separate text file. Every file contains:

```text
FORMAT_VERSION=1
METHOD=SAXS
TERM=ATOMIC
Q_MIN=0.0
Q_MAX=...
Q_STEP=...
PARAMETERS1=c0,c1,...
PARAMETERS2=c0,c1,...
```

`METHOD` is `SAXS` or `SANS`. Allowed `TERM` values are:

- SAXS: `ATOMIC`, `SOLVENT`, `MIXED`.
- SANS: `ATOMIC_H`, `MIXED_H`, `ATOMIC_D`, `MIXED_D`, `SOLVENT`.

Rows must be contiguous, one-based, sorted, and equal in number to the scattering beads. `Q_MIN`, `Q_MAX`, and `Q_STEP` describe the fitted grid and are mandatory. PLUMED rejects a file used in the wrong context or outside its declared range. No global `0.3 Å^-1` limit is imposed on custom parameters.

Recommended names are:

```text
<system>_saxs_{atomic,solvent,mixed}_parameters.inp
<system>_sans_{h_atomic,h_mixed,d_atomic,d_mixed,solvent}_parameters.inp
```

### LCPO and mapping files

Hybrid LCPO files contain one tuple per atom in the PLUMED `ATOMS` selection:

```text
FORMAT_VERSION=1
REPRESENTATION=HYBRID
LCPO_PARAMETERS1=radius,p1,p2,p3,p4
```

Native-CG LCPO files contain one tuple per bead:

```text
FORMAT_VERSION=1
REPRESENTATION=CG
LCPO_PARAMETERS1=radius,p1,p2,p3,p4
```

Mapping files contain one row per scattering bead:

```text
FORMAT_VERSION=1
BEAD_ATOMS1=17,18,19
BEAD_WEIGHTS1=0.50,0.25,0.25
```

Weights are non-negative and normalized per bead. Shared atoms are permitted. Generated mappings always carry explicit weights; an omitted inline weight list may use normalized atomic masses as a user convenience.

Cardinality invariants are:

- all coefficient term files have the same bead count;
- hybrid mapping rows equal the coefficient-row count, while hybrid LCPO rows equal the atomistic `ATOMS` count;
- native-CG coefficient rows, LCPO rows, and `ATOMS` count are equal.

### Cross-repository acceptance fixture

`form-factors-beads` owns a small deterministic fixture containing coordinates, mapping, all applicable parameter files, and reference intensities/exposure states. A generated copy is consumed by the PLUMED regression suite and records the generator revision plus a content manifest. PLUMED must not maintain a separately hand-derived version of the expected data.

## User interface

### Preserve all legacy paths

Do not change the behavior or defaults of existing `PARAMETERS`, `PARAMETERSFILE`, `ATOMISTIC`, `MARTINI`, or `ONEBEAD` modes. In particular, the current `ONEBEAD` absolute `SASA_CUTOFF` keeps its meaning. New automatic-solvation behavior is selected only when the decomposed interface is used.

### Extend legacy SANS parameters to polynomials

The current SANS implementation parses every inline `PARAMETERSn` and every row in `PARAMETERSFILE` as one scalar, then repeats that value at every q point. Replace both SANS scalar-parsing branches with the vector-based parsing and evaluation already used by SAXS.

Each SANS scattering center accepts any non-empty, constant-first coefficient vector:

```text
PARAMETERS1=c0
PARAMETERS2=c0,c1,...,cN
```

Evaluate the corresponding form factor as

```text
F_i(q) = c0 + c1 q + ... + cN q^N
```

with `q` in `Å^-1`. A single coefficient remains the q-independent scattering-length case and must reproduce the current results. Do not impose a fixed degree or require all scattering centers to use the same coefficient count. For both scalar and polynomial inputs, calculate the normalization from the constant terms as `I(0)=(sum_i c_i0)^2`.

Inline and `PARAMETERSFILE` forms must accept the same values and produce identical results. This requires no syntax change and makes the existing seven-coefficient, sixth-order SANS output from `form-factors-beads` directly consumable. Preserve the existing errors for missing, malformed, empty, unsorted, or incorrectly numbered parameter rows.

### Decomposed coefficients

Support direct values and files:

```text
ATOMIC_PARAMETERSn=...
SOLVENT_PARAMETERSn=...
MIXED_PARAMETERSn=...
ATOMIC_PARAMETERSFILE=...
SOLVENT_PARAMETERSFILE=...
MIXED_PARAMETERSFILE=...
```

For SANS, additionally support:

```text
DEUTERATED_ATOMIC_PARAMETERSn=...
DEUTERATED_MIXED_PARAMETERSn=...
DEUTERATED_ATOMIC_PARAMETERSFILE=...
DEUTERATED_MIXED_PARAMETERSFILE=...
```

The ordinary SANS atomic/mixed inputs are the H state; the deuterated inputs are the D state; solvent parameters are shared. Inside each contextual file, rows remain named `PARAMETERSn` as specified by the contract.

Inline use must declare the fit domain with `PARAMETERS_Q_MIN`, `PARAMETERS_Q_MAX`, and `PARAMETERS_Q_STEP`. Reject partial decompositions, mixed SAXS/SANS metadata, inconsistent q grids, inconsistent polynomial degrees where the evaluator requires equality, and mismatched bead counts.

### LCPO input

Support either:

```text
LCPO_PARAMETERSn=radius,p1,p2,p3,p4
```

or:

```text
LCPO_PARAMETERSFILE=...
```

File and inline forms are mutually exclusive and numerically equivalent. Existing built-in atom/residue lookup remains available for legacy protein/nucleic-acid use. Supplied tuples take precedence and enable nonstandard atomistic systems and native-CG beads.

### Mapping input and mode selection

Support either inline mapping:

```text
BEAD_ATOMSn=17,18,19
BEAD_WEIGHTSn=0.50,0.25,0.25
```

or `MAPPINGFILE=...`. File and inline forms are mutually exclusive.

- If a mapping is supplied, select hybrid mode: construct one virtual scattering center per mapping row.
- If no mapping is supplied, select native-CG mode: each `ATOMS` entry is one scattering center.
- Reject ambiguous combinations rather than inferring a legacy mode.

Inline weights may be omitted to request normalized mass weights. Mapping coordinates use a PBC-safe weighted construction, and forces on a virtual center are distributed to contributing atoms using the same weights.

### Solvation and SANS controls

Add decomposed-mode controls:

```text
SASA_FRACTION_CUTOFF=...
SOLVATION_STRIDE=...
SOLVATION_CORRECTION=...
```

The exposure variable is dimensionless:

```text
x_i = LCPO_area_i / isolated_LCPO_area_i
```

The bead is exposed when `x_i` passes the declared cutoff. Preserve the existing cutoff direction if scientifically equivalent; otherwise name/document it explicitly and cover the boundary in tests. The binary state is updated only every `SOLVATION_STRIDE` steps and remains fixed between updates. No derivative is taken through the discrete state decision.

For SAXS, retain `SOLVDENS` for the bulk electron density and define `SOLVATION_CORRECTION` as the exposed-shell change in compatible units. For SANS, support `DEUTER_CONC` as the water-composition convenience and `SOLVENT_SLD` as a direct override, plus `SOLVATION_SLD_CORRECTION` for the exposed shell. A direct SLD and a composition-derived SLD must be mutually exclusive.

For exchange, add:

```text
EXCHANGE_FRACTION=...
EXCHANGE_SEED=...
EXCHANGEABLE_BEADS=...
```

`EXCHANGE_FRACTION` defaults to `DEUTER_CONC`. Only exposed beads listed as exchangeable are eligible to use the D term set. Selection must be reproducible from the seed and stable between exposure updates; document whether newly exposed beads reuse a bead-identity decision or draw at transition. Prefer a deterministic per-bead hash so the result is independent of MPI layout and history.

## Internal design

### 0. Freeze the existing LCPO behavior before production changes

The first PLUMED commit for this work must contain tests and reference data only. Before changing the SANS parser, extracting LCPO, or adding any new solvation behavior, add a black-box LCPO characterization test under the ISDB regression suite and run it successfully against the untouched implementation.

The baseline fixture must exercise the current `ONEBEAD` LCPO path with deterministic coordinates and built-in parameters, covering an isolated center, a partially overlapping pair, a multi-center overlap, a buried center, and a periodic-boundary overlap. It must also place accessible fractions on both sides of the existing `SASA_CUTOFF` and exercise `SOLVATION_STRIDE`. Freeze all currently observable consequences: SAXS and SANS intensities, exposure-dependent solvation choices, and coordinate derivatives/forces. Record the pre-change PLUMED revision used to generate the references. Establish comparison tolerances from repeated runs of that unchanged revision on supported test toolchains, then freeze those tolerances as well; later work must not loosen them.

These reference files are immutable compatibility evidence: do not regenerate or re-baseline them to make later code pass. Any intentional change to legacy LCPO behavior is outside this feature and requires a separate scientific justification and review.

Because LCPO is currently private and embedded in the SAXS/SANS action, the pre-change test is necessarily black-box. As part of the first production commit, add a focused C++ test under PLUMED's unit-regression harness for the extracted helper. Feed it the same frozen geometries and tuples, and compare per-center LCPO areas, total area, neighbor/PBC behavior, and analytic coordinate derivatives with the legacy reference values. The feature may proceed only when both the pre-change black-box test and the direct helper unit test are green.

### 1. Extract a reusable LCPO engine

Refactor the existing embedded LCPO code into an internal helper that accepts coordinates and `(radius,p1,p2,p3,p4)` tuples. Preserve its formulas, PBC behavior, neighbor-list semantics, fixed `0.14 nm` probe, and analytic coordinate derivatives. First prove that feeding the current built-in tuples through the helper reproduces the old result bitwise or within a tight numerical tolerance.

The helper is resolution agnostic:

- hybrid mode calls it for atomistic centers;
- native-CG mode calls it for bead centers.

This is the main reason no alternative SASA implementation is needed for the extension.

### 2. Implement scattering-center construction

Create a representation layer exposing scattering-center positions and the reverse force map.

- Native CG is a one-to-one identity map.
- Hybrid positions are weighted, PBC-safe virtual centers.
- Shared atoms contribute to multiple centers and receive the sum of their mapped forces.

Keep this layer independent of Equation-9 evaluation so it can be unit tested with simple forces.

### 3. Aggregate hybrid exposure

Run LCPO on the atomistic coordinates and aggregate accessible and isolated
areas separately onto beads, splitting shared atoms equally. The bead fraction
is the aggregated accessible area divided by the aggregated isolated area.
Mapping weights construct coordinates and propagate forces; they do not weight
physical surface areas. Store both the continuous bead fraction and the binary
state for diagnostics.

Native-CG mode directly uses the bead LCPO fraction. The classification interface after this point is identical for both modes.

### 4. Implement Equation-9 evaluation

At each requested q value:

1. evaluate the three polynomials for every bead;
2. select the bulk or exposed-shell solvent density from the cached exposure state;
3. for SANS, select H or D atomic/mixed terms for eligible exchanged beads;
4. form `R=A+rho^2 S-rho M`;
5. apply the serialized sign representation and compute `F=sigma sqrt(R)`;
6. use the resulting bead amplitudes in the existing Debye/intensity and derivative path.

Reject materially negative radicands with bead, q, state, and component values in the error. Clamp only roundoff-sized negatives under a documented relative tolerance. Validate the requested q grid against the metadata before starting the simulation.

Normalization at `F(0)` must be defined consistently with the existing SAXS/SANS action. Adding decomposition must not silently alter scale conventions.

### 5. Handle derivatives deliberately

Differentiate the intensity through scattering-center coordinates and route virtual-center forces to atoms. Include derivatives of continuous LCPO areas only if they enter a continuous energy expression; with the planned binary cached state, do not differentiate the threshold or the state transition. Make this piecewise nature explicit in the manual.

Run finite-difference tests away from LCPO topology changes, the exposure threshold, and solvation-update steps. If the existing implementation has CPU/GPU paths, require parity for the new evaluator or explicitly disable the unsupported path with a clear error.

### 6. Parse and validate the versioned formats

Use a shared parser for contextual coefficient files, plus focused parsers for LCPO and mapping files. Validate version, context, units implied by method, row continuity, numeric finiteness, normalized weights, global atom membership, q-domain agreement, and all cardinality invariants before allocating the run-time evaluator.

Emit a concise startup summary containing representation, atom/bead counts, q domain, solvent-density source, exposure cutoff/stride, exchange settings, and file versions.

### 7. Document the action

Add manual examples for:

- legacy behavior, demonstrating no syntax change;
- hybrid SAXS with three term files, mapping, and atom LCPO file;
- native-CG SAXS with bead LCPO parameters;
- hybrid/native-CG SANS with H/D exchange;
- the verbose all-inline equivalents.

Explain that native-CG LCPO parameters are effective parameters requiring validation against atomistic exposure, not universal bead constants.

## Verification matrix

- Existing `ONEBEAD`, atomistic, MARTINI, and file-based regression tests remain unchanged and pass.
- Before any production edit in this feature, a test-only commit freezes current LCPO results from the untouched implementation and passes in CI.
- The frozen LCPO characterization covers isolated, overlapping, buried, and PBC geometries; both sides of the exposure cutoff; the solvation stride; SAXS/SANS outputs; and forces.
- After extraction, direct C++ unit tests reproduce the frozen per-center areas and analytic derivatives from the old implementation without rewriting the expected values.
- Every subsequent implementation commit must keep both the legacy black-box LCPO characterization and the direct LCPO unit tests green.
- The existing scalar SANS `PARAMETERSFILE` regression remains numerically unchanged, including intensities and forces.
- Inline and file-based scalar SANS parameters are numerically identical.
- Inline and file-based six-coefficient SANS polynomials produce the same q-dependent form factors, intensities, normalization, and forces.
- A seven-coefficient SANS polynomial generated by `form-factors-beads` is accepted and agrees with explicit constant-first evaluation at several q values.
- Mixed non-empty coefficient counts across SANS scattering centers are accepted; empty and malformed vectors are rejected.
- SANS polynomial normalization uses only the constant coefficient from each scattering center.
- The extracted LCPO helper reproduces built-in protein/nucleic-acid results with supplied equivalent tuples.
- Analytical LCPO geometry tests cover isolated, partial-overlap, buried, and PBC-crossing centers.
- Hybrid virtual-center positions and force routing pass finite differences, including shared atoms.
- Native-CG identity mapping and LCPO classification are tested independently.
- File and verbose inline forms produce identical coefficients, states, intensities, and forces.
- Parser failures cover wrong version/context, missing or duplicate rows, q mismatch, atom/bead mismatch, bad weights, non-finite values, and wrong LCPO representation.
- Equation-9 tests cover exact reconstruction, signed amplitudes, sign crossings, tiny-negative clamping, and material-negative failure.
- Binary exposure tests cover cutoff equality, stride caching, restart behavior, and the absence of a derivative through state changes.
- SANS tests cover direct versus composition-derived SLD, deterministic exchange, eligibility restricted to exposed beads, and MPI-layout independence.
- CPU/GPU parity is required where both paths are supported.
- A benchmark records LCPO and scattering overhead versus atom/bead count and `SOLVATION_STRIDE`.
- The generated cross-repository fixture is consumed without modifying its reference outputs.

## Delivery sequence and decision gates

1. **Freeze contract v1 with the tool:** especially sign encoding, headers, units, and validation errors.
2. **Land the LCPO baseline before production edits:** add only the black-box characterization fixture and immutable references, prove them green against the untouched code, and record that revision.
3. **Extract LCPO with no behavior change:** add direct C++ helper unit tests against the frozen cases; both old and new test layers are the gate.
4. **Extend legacy SANS parameter parsing:** keep all LCPO baseline tests green.
5. **Add versioned parsers and native-CG identity centers.**
6. **Add hybrid mapping and force propagation.**
7. **Add Equation-9 SAXS and binary solvation.**
8. **Add SANS solvent SLD and deterministic H/D exchange.**
9. **Consume the tool-generated acceptance fixture.**
10. **Complete documentation, performance measurements, and CPU/GPU checks.**

Open choices to settle during PLUMED-plan refinement are the sign-crossing representation, hybrid atom-to-bead exposure aggregation rule, default accessible-fraction cutoff and comparison direction, restart serialization of cached states, roundoff tolerance for radicands, and whether any accelerator path is in the first release. They must be settled consistently with the tool plan before coding their interfaces.
