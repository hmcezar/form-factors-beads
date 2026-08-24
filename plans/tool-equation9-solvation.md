# Plan: Equation-9 parameter generation and automatic solvation

## Purpose and boundary

Extend `form-factors-beads` so it generates the decomposed scattering terms and solvation metadata needed to evaluate the paper's Equation 9 at run time. This repository owns parameter derivation, fitting, validation, and serialization. It does not own the run-time collective variable, SASA updates, or force propagation; those belong to PLUMED and are planned separately in [plumed-equation9-solvation.md](plumed-equation9-solvation.md).

The implementation must support both:

- **Hybrid resolution:** an atomistic trajectory is mapped to virtual scattering beads. Atomistic LCPO exposure is accumulated onto those beads.
- **Native coarse graining:** the simulation coordinates are already the scattering beads. LCPO is evaluated directly on those bead centers using derived bead parameters.

Both paths intentionally use the same LCPO functional form. Hybrid mode reuses the established atom-level radii and coefficients; native-CG mode supplies effective bead radii and coefficients to the same algorithm.

## Shared compatibility contract (version 1)

This section is normative and is duplicated in the PLUMED plan. A change here requires the matching change there before either implementation is merged.

### Equation and units

For bead `i`, generate polynomial representations of

```text
F_i(q, rho) = sigma_i(q, rho) sqrt(A_i(q) + rho^2 S_i(q) - rho M_i(q))
```

where `A` is the atomic/self term, `S` the displaced-solvent term, `M` the mixed term, and `rho` the solvent scattering-length density.

- `q` is always in inverse ångström (`Å^-1`). Polynomial coefficients are ordered constant-first: `c0,c1,...`.
- SAXS uses electron units: `A [e^2]`, `S [Å^6]`, `M [e Å^3]`, and `rho [e Å^-3]`.
- SANS uses femtometres: `A [fm^2]`, `S [Å^6]`, `M [fm Å^3]`, and `rho [fm Å^-3]`.
- LCPO tuples use the units of the existing PLUMED LCPO implementation: base radius in ångström, areas in square ångström, and the corresponding coefficient dimensions required by the LCPO expression. The fixed water probe is added by PLUMED and is `1.4 Å` (`0.14 nm`); it is not folded into the serialized base radius.
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

Weights are explicit, non-negative, normalized per bead, and have the same cardinality and order as `BEAD_ATOMS`. Shared atoms are permitted. The generator always writes weights explicitly; PLUMED may offer mass weighting as a convenience only for hand-written inline mappings.

Cardinality invariants are:

- all coefficient term files have the same bead count;
- hybrid mapping rows equal the coefficient-row count, while hybrid LCPO rows equal the atomistic `ATOMS` count;
- native-CG coefficient rows, LCPO rows, and PLUMED `ATOMS` count are equal.

The equivalent inline PLUMED interface is defined in the PLUMED plan. Files and inline values must yield identical results.

### Cross-repository acceptance fixture

This tool owns a small, deterministic fixture containing coordinates, mapping, every applicable parameter file, and reference intensities/exposure states. PLUMED consumes a generated copy in its regression tests. The fixture records the generator revision and a content manifest; hand-edited PLUMED copies are not a second source of truth.

## Scientific model

### Term accumulation

For each mapped bead and sampled structure, calculate the Debye-like components before contrast is applied:

```text
A_i(q) = sum_kl a_k(q) a_l(q) sinc(q r_kl)
S_i(q) = sum_kl s_k(q) s_l(q) sinc(q r_kl)
M_i(q) = sum_kl [a_k(q)s_l(q) + s_k(q)a_l(q)] sinc(q r_kl)
```

The atom sums and exact allocation convention must reproduce the existing total bead-factor calculation when reconstructed at the same solvent density. At `q=0`, enforce or explicitly test:

```text
A_i(0) = (sum_k a_k(0))^2
S_i(0) = (sum_k s_k(0))^2
M_i(0) = 2 (sum_k a_k(0))(sum_k s_k(0))
```

Average each component over the selected trajectory frames, then fit `A`, `S`, and `M` independently on the requested q grid. Store enough precision that reconstructing `F` is not dominated by serialization error.

The signed amplitude `sigma` cannot be recovered from a squared term alone. Generate and validate a deterministic sign representation. Prefer deriving the sign continuously from the corresponding unsquared bead amplitude and serializing sign-change metadata if crossings occur; if the first implementation only supports a constant sign per bead, reject fits with a crossing instead of silently taking the positive root.

### SAXS output

Generate the three bulk term sets `ATOMIC`, `SOLVENT`, and `MIXED`. Solvation correction remains a run-time choice in PLUMED: exposed and buried states apply their respective solvent densities to the same decomposed terms.

### SANS output and exchange

Generate a common solvent term plus protiated and exchanged atomic/mixed terms. Detect exchangeable hydrogens as hydrogens bonded to N, O, or S. Provide include/exclude overrides for unusual chemistry and report every selected atom.

The generated H and D term sets describe physical end states, not a stochastic realization. At run time, only exposed beads containing exchangeable sites are eligible for the D set. PLUMED selects eligible beads using `EXCHANGE_FRACTION` (defaulting to `DEUTER_CONC`) and a reproducible seed.

### LCPO assignment for hybrid mode

Use the existing LCPO parameterization wherever a residue/atom tuple is recognized. For nonstandard atoms:

1. infer the element from topology metadata, not only the atom-name spelling;
2. apply documented element defaults for C, N, O, P, S, F, and Mg, issuing a
   warning whenever any fallback is selected;
3. exclude hydrogen centers from LCPO, consistently with the current heavy-atom implementation;
4. allow per-atom and per-element overrides;
5. fail on unknown or ambiguous elements rather than guessing.

The output should state whether each tuple came from an exact built-in assignment, an element fallback, or an override. This keeps the simple fallback usable for molecules such as TX9 without hiding its approximation. F and Mg follow Amber's established borrowed-coefficient LCPO tuples; they remain ordinary `element_fallback` assignments and carry the same warning as C, N, O, P, and S.

### LCPO assignment for native-CG mode

Derive initial bead tuples from their constituent atoms:

- default to the base bead radius of the sphere having the summed displaced atomic volume;
- calculate `p1..p4` as isolated-surface-area-weighted averages of the constituent atom coefficients;
- permit bead-type and individual-bead overrides.

This is a pragmatic transfer of the LCPO model, not a claim that atomistic LCPO coefficients are formally resolution invariant. The generator must therefore compare native-CG and atomistic-hybrid exposure classifications on mapped validation trajectories and expose the mismatch rate before the CG parameters are accepted.

## Implementation work packages

### 1. Introduce explicit domain objects

Add typed representations for q grids, Equation-9 term curves/fits, scattering units, LCPO tuples and provenance, mappings, and the versioned output manifest. Keep computation independent from text serialization so the same objects drive plots, files, and tests.

### 2. Refactor the current form-factor path

Extract the current atom-pair accumulation and bead mapping into reusable functions. Add component accumulation without changing the existing total-factor command or its output. Establish a reconstruction test against the legacy Fraser-style factors before adding new CLI behavior.

### 3. Add component fitting and sign handling

Fit each term independently, retain exact `q=0` constraints, quantify residuals over the full declared range, detect negative fitted radicands under representative solvent densities, and implement the selected sign-crossing representation. Emit a hard diagnostic when a requested polynomial degree is inadequate.

### 4. Add hybrid LCPO parameter generation

Implement exact lookup, element fallback, and override precedence. Emit tuples in the exact order of the exported PLUMED atom selection and generate a provenance table suitable for auditing nonstandard molecules.

### 5. Add native-CG LCPO parameter generation

Derive volume-equivalent bead tuples from the mapping, add override support,
and implement the atomistic-versus-CG exposure comparison. Report correlation
of accessible fractions, the binary confusion matrix at the diagnostic
threshold, and the largest per-bead mismatches.

### 6. Add SANS exchange chemistry

Build bond-aware exchangeable-H detection, overrides, H/D end-state terms, and an eligibility record per bead. Ensure deuterium substitution changes atomic and mixed terms but not the common displaced-solvent term.

### 7. Add serializers and CLI commands

Write one contextual file per term plus the mapping and LCPO files. Suggested CLI organization is a new `decompose` or `plumed-export` operation with explicit `--representation hybrid|cg`, q-grid, polynomial degree, SAXS/SANS, LCPO fallback, override, and output-prefix options. Preserve existing commands and defaults.

### 8. Add diagnostics and provenance

Produce machine-readable and human-readable summaries containing input hashes, frame selection, mapping identity, q range, fit degree, maximum/RMS term error, reconstructed-factor error, sign crossings, LCPO assignment sources, exchange selections, software revision, and generated filenames.

### 9. Publish the acceptance fixture

Generate a tiny fixture covering at least one exposed and one buried bead, a shared mapping atom, a nonstandard element-fallback assignment, and an exchangeable SANS site. Store expected Equation-9 reconstructions at several solvent densities and exposure states.

## Verification matrix

- Unit tests for `sinc`, pair allocation, the three `q=0` identities, units, fitting, serialization precision, and sign crossings.
- Reconstruction of existing SAXS/SANS bead factors at their original solvent density within a declared tolerance.
- Round-trip parsing of every generated file and rejection of missing, duplicated, or non-contiguous rows.
- Exact-assignment and generic-fallback LCPO tests, including TX9-like atom names and explicit overrides.
- Hybrid mapping tests for normalized weights, shared atoms, global serials, and ordering.
- Native-CG radius/coefficient derivation tests and hybrid-versus-CG exposure diagnostics.
- Exchange detection tests for N/O/S-bound H, include/exclude overrides, and H/D end states.
- A regression proving the legacy TX9 workflow is unchanged when decomposition is not requested.
- The cross-repository fixture consumed successfully by PLUMED once both implementations exist.

## Delivery sequence and decision gates

1. **Freeze contract v1:** agree with the PLUMED implementation on headers, units, sign encoding, and errors.
2. **Land decomposition behind a new command:** legacy behavior remains untouched.
3. **Land hybrid LCPO export:** validate exact tuples and the nonstandard element fallback.
4. **Land native-CG tuple derivation:** require exposure-comparison diagnostics, but do not require perfect agreement.
5. **Land SANS exchange end states.**
6. **Generate and pin the acceptance fixture.**
7. **Run the PLUMED consumer regression:** this is the compatibility gate for declaring the feature complete.

Open scientific choices to settle while refining this plan are the exact sign-crossing encoding, default generic-element LCPO tuples, the default accessible-fraction threshold, polynomial error tolerances, and acceptable hybrid-versus-CG exposure mismatch. None should be embedded implicitly in code.
