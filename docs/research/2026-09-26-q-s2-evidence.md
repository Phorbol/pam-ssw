# Q-mode S2 evidence slice

**Scope.** Read-only audit of the archived `lasp` ELF for the S2 parameter setter,
analytic derivative dispatch and Q-vector normalization. This does not implement
S2 or establish default-Q eligibility. It deliberately does not inspect S1 or
S3–S6. ELF virtual addresses below refer to
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.

## Recovered symbols and inputs

`nm -C` identifies `hchemlib::SymfS2::set_para(std::vector<float> const&)` at
`0xb4c580`, and the concrete dispatch target
`AtomDescriptor::cal_symf_deri_S2<SymfS2>(...)` at `0xa25730`. The setter reads
four float slots. Slot 0 is copied as a 32-bit value to object offset `+0xd70`;
slots 1, 2 and 3 are independently converted with `cvttss2si` (truncate toward
zero) and stored at `+0xd50`, `+0xb08` and `+0xd44`, respectively
(`0xb4c580–0xb4c5b1`). The S2 analytic kernel uses `+0xd44` in exponent
arithmetic feeding `power_real<float>` branches at `0xa26611–0xa266e4` and
`0xa26b39–0xa26b75`.
The `+0xd70` field enters a `tanhf` radial envelope/table path at
`0xa2674f–0xa26857`. The derivative family also calls
`QsymfBase::hderi_stress_Q` at `0xa26b21` (callee `0xa26c70`), showing that
Q-symmetry/cell derivative machinery is part of this family.

`SymfS2::sympara()` at `0xb4bb50` prints the same four fields in order:
`+0xd70`, `+0xd50`, `+0xb08`, and `(+0xd44)-1` (`0xb4bcc7–0xb4bd43`). The
native C60 logs include the concrete row `S2 : 2.79456  6  4  8` and the full
long runs contain 19 S2 rows in seed 17093 and 7 in seed 17094. Across those
26 rows the second field is always 6, the third varies over `{2,4,6}`, and the
fourth over `{-3,-1,2,4,8,16}`. The S1 contract calls its second printed field
`neigb_atom`, but it does not establish that this means a neighbor count. The
constant `6` in carbon-only logs cannot distinguish a neighbor-count
parameter from carbon's atomic number or another type key. The S2 paper uses
harmonic degree `L`, and the third field's small even-valued set is consistent
with `+0xb08` being `L`. These semantic names remain interpretations rather
than proven field names: no direct `+0xd50`/`+0xb08` load occurs in the bounded
S2 numeric kernel slice, and the descriptor constructor's Ylm/type setup
mapping was not recovered. `get_gid()` at `0xb5c5f0` points to `+0xd4c`, not
either field, so `+0xd50` is not the descriptor's own gid.

The fourth printed S2 value is specifically `+0xd44 - 1`; meanwhile the
numeric kernel consumes `+0xd44` in its power arithmetic. Thus a log value
`n_printed=8` corresponds to stored `d44=9`. This observed representation does
not by itself establish a mismatch with the paper exponent; it may encode an
exponent offset or the derivative/power-helper convention. The long logs map
the symbolic printed row and setter slots, but they do not close the
setter-to-effective-formula transform. The exact runtime sample is
at `c60-native-long-20260920/seed17093/lasp.out:21184` in the sibling
`c60-vacuum-geometry` evidence tree.

This establishes the setter's actual vector shape, a likely harmonic-degree
role for `+0xb08`, and an integer input used in radial power arithmetic. The
exact mapping of `+0xb08` to the Ylm degree and the interpretation of `+0xd50`
remain unresolved, as does the printed-to-effective exponent transform; the
descriptor construction/serialization inputs and neighbor-type partition are
not fully traced. In particular, `+0xd70`
participates in a radial envelope path, but this slice alone does not establish
its portable parameter name or cutoff convention.

## Mathematical family and normalization

The archived primary paper subset is
[`ptsd-primary-2018-20260926/`](../../research/ga_ssw/evidence/ptsd-primary-2018-20260926/).
Huang et al., *Chemical Science* (2018), DOI
[10.1039/C8SC03427C](https://doi.org/10.1039/C8SC03427C), equation 7 defines
the S2/Q-value two-body descriptor as

```text
S_i = sqrt(sum_{m=-L}^L | sum_{j != i} phi(r_ij) Y_Lm(rhat_ij) |^2),
phi(r) = r^n f_c(r).
```

Using the spherical-harmonic addition theorem, this is equivalently

```text
S_i = sqrt(C_L * sum_{j,k != i} phi_j phi_k P_L(t_jk)),
C_L = (2L+1)/(4*pi),
t_jk = e_j dot e_k,  e_j = r_ij/r_ij.
```

For a nonzero value, its neighbor derivative can be written

```text
g_j = (C_L/S_i) * [
    phi'_j e_j * sum_k phi_k P_L(t_jk)
  + phi_j/r_j * sum_k phi_k P'_L(t_jk) * (e_k - t_jk e_j)
],
g_i = -sum_j g_j.
```

Here `phi'_j = n*r_j^(n-1)*f_c(r_j) + r_j^n*f'_c(r_j)`. The center relation
is for a complete nonperiodic image-free derivative; periodic images must be
mapped and accumulated by their atom/cell indices. The expression follows by
differentiating the paper's equation and applying the addition theorem. The ELF
evidence is consistent with that family (S2 dispatch, `power_real`, radial
`tanhf`, Q stress derivative helper), but this bounded disassembly did not
independently prove every term or normalization constant instruction by
instruction. Do not treat the formula as a bitwise/native-parity claim.

The outer `get_ptsd_mode` path independently normalizes the full `3N`
coordinate derivative by its Euclidean norm and scales the six cell/stress
components by that same norm (`0x717474–0x7176dd`, as recorded in the
[Q-mode contract](2026-09-17-q-mode-contract.md)). It is therefore
`q=v/||v||_2` over atomic entries, with shared scaling for the six additional
components; it is not a combined Cartesian-plus-cell metric. Zero norm behavior
is not recovered.

## Readiness boundary and next probe

The S2 family has a source-grounded mathematical identity, so replacing it with
an arbitrary radial/random descriptor would be unjustified. An independent
S2 primitive is not yet ready to claim native-compatible implementation:
parameter-slot meaning/type partition, exact cutoff function and periodic image
mapping are not fully resolved, and `S_i=0` is nondifferentiable. No epsilon
regularization or silent direction fallback is supported by this evidence.

The minimal falsifiable next probe is a CPU-only three-atom, noncollinear,
nonperiodic fixture with explicit `L`, `n`, and a supplied smooth cutoff. Compare
the addition-theorem value/analytic gradient against central finite differences
away from zero norm, and check translation cancellation. Separately evaluate a
configuration with `S_i=0` to record the implementation's explicit error/zero
policy rather than regularizing it. This tests the independently derived
formula only; native-parameter compatibility still requires a documented S2
input row that reveals the four setter slots. No PES, scheduler job or production
API change is needed for that probe.

## Subsequent independent implementation check (not native parity)

The private S2 primitive is now implemented in research/ga_ssw/s2_reference.py.
CPU1499153 passed9S1/S2tests and6actual-geometry checks on saved C60 defect
and rutile12 environments, using independent SciPy spherical-harmonic values
and actual atomic-coordinate finite differences. Periodic image contributions,
including self-images, were accumulated explicitly in the research check.
This closes this independent formula check only, not native slot mapping,
public walker integration, fullS1–S6or efficiency. See the source evidence
[readout](../../research/ga_ssw/evidence/ptsd-primary-2018-20260926/README.md).
