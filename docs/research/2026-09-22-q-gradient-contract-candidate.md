# Q-gradient contract: bounded candidate review

Date: 2026-09-22. This note records a read-only review from the `eafbc57`
checkout. It does not implement the complete Q controller or change the
walker/public API/checkpoint/default. A later private analytic primitive and
unexecuted checks are recorded below.

## Evidence boundary

The current contract document and the archived ELF support the following
native path:

`gen_randommode` → `hget_ptsd_mode_` → `get_ptsd_mode` → selected
`BrickPTSD::cal_rc_deri` → `AtomDescriptor::cal_symf_deri` → normalize a
length-`3N` direction and six cell/stress components by the same norm.

The runtime S1–S5 census and the S1 setter trace already establish that the
runtime initialization path is active for the archived native runs. The
remaining boundary here is narrower: exact S1 derivative dispatch and cutoff
semantics, plus whether the derivative writes both center and neighbor atoms.
The following new ELF slice addresses that boundary without proposing a Q
implementation.

The supplied local `132.txt` is not the named 2018 PTSD paper. Its
`retrieval.json`/`manifest.json` identify Huang et al., WIREs Computational
Molecular Science (2019), DOI `10.1002/wcms.1415`, acquired as the author-hosted
`132.pdf`. Its PTSD section is still useful for the broad descriptor family,
but it cannot be cited as the requested DOI `10.1039/C8SC03427C`; that 2018
source needs a separate archived text or verified retrieval.

## Formula and contract table

| Layer | Formula/observation | Unit or invariant | Boundary |
|---|---|---|---|
| Paper PTSD family in local text | Up to four-body descriptors from pair distance, three-atom angle, four-atom torsion, combined with power, cosine and spherical functions | Geometry descriptor; no Q direction unit stated | Describes NN descriptors, not the Q caller or selected mode |
| Confirmed S1 string | `S1 = sum_j r_ij^n f_c(r_ij)` | `r` follows structure length; `n` dimensionless | S1 is only one radial family; no complete S1–S6 mapping |
| S1 full atomic derivative | `g_j = phi'(r_ij)e_ij`, `g_i = -sum_j g_j`, with `phi(r)=r^n f_c(r)` and `e_ij=(R_j-R_i)/r_ij` | Translation sum is zero; torque sum is zero for complete nonperiodic pair derivatives | Center-only or missing-neighbor derivatives violate these checks |
| Native Q derivative | `v_(a,alpha) = dG_k/dR_(a,alpha)` from `BrickPTSD::cal_rc_deri` and `cal_symf_deri` | Full `3N` vector; not an NN/PES force | Exact `G_k`, parameters and descriptor selection remain missing |
| Native normalization | `q = v / sqrt(sum_(a,alpha) v_(a,alpha)^2)`; six cell/stress quantities are scaled by the same norm | Shared norm; zero-norm behavior not recovered | Fixed-cell versus cell components and their caller contract remain unresolved |
| Q runtime evidence | Existing forced-Q run: Q changed directions/costs, but four C60 states were mixed and all eight landings missed target cage topology | Evidence of activity, not benefit | Does not justify default switch or full Q implementation |

## New S1 ELF slice

The exact instantiated function is
`AtomDescriptor::cal_symf_deri_radial<SymfS1>` at `0xa24470`. The setter
contract supplies `cutoff=6.48507` at object offset `+0x38`, `neigb_atom=6`
at `+0x18`, and `n=-1` at `+0x0c`; the latter two integers are produced by
float-to-int truncation in `SymfS1::set_para`. The function uses the stored
cutoff in a `distance/cutoff` division (`0xa24c08`) and then enters a `tanh`
cutoff branch (`0xa24c21`). This confirms the cutoff is an active radial
function parameter. The disassembly slice does not by itself justify naming
the complete tanh expression or its constants as a portable default.

The derivative is not center-only. In the neighbor loop, the computed three
components are accumulated into the indexed neighbor output (`0xa24d73`),
while the same components are subtracted from the center output
(`0xa24e01–0xa24e36`). The subsequent parameter-vector terms are also applied
to the center/neighbor pair (`0xa24e3c–0xa24ef0`). Thus the native S1 path
contains both contributions needed for the translation cancellation check.
The slice confirms the radial S1 dispatch and sign pattern; it does not prove
the complete periodic image mapping or all S2–S6 dispatches.

The newly traced cutoff-cache miss gives an instruction-level expression,
without renaming it as a portable physical cutoff convention. With
`r_scaled = distance/cutoff` and `C = 1.0f` loaded at `0xa24cff` from
`0x4a75b1c`, the slice computes

```text
t = tanhf(C - r_scaled)                         # 0xa24bfe–0xa24c21
table_value = t*t*t                              # 0xa24c26–0xa24c3f
table_derivative = -3*t*t*(1 - t*t)/cutoff       # 0xa24c43–0xa24c84
```

The `-3.0` double is loaded at `0xa24c47` from `0x4a75af8`; the `1.0` float is
loaded at `0xa24cff` from `0x4a75b1c`. The branch computes this only when the
cached pair/grid value is not already marked and `cutoff - distance` is above
the epsilon float at `0x4a75b20` (approximately `1e-5`), via
`0xa248c6–0xa248d6`; otherwise the radial contribution is skipped. Cache hits
take the stored values.
This is the observed arithmetic path; native table/image indexing remains
outside the private primitive, which takes explicit relative vectors.

The exponent path is likewise explicit but should not be collapsed into a new
default: `power_real<float>` is called for the positive and negative exponent
branches (`0xa24a95`/`0xa24b0c` and `0xa24f59`/`0xa24fbe`), and the integer `n`
from `+0x0c` enters the coefficient construction at `0xa24b56–0xa24b66`.
The resulting three Cartesian components are then added to the neighbor and
subtracted from the center. This closes the sign and full-atom contribution
for the radial primitive, but not the pair-image index transformation.

## Implementation decision

Added `research/ga_ssw/s1_reference.py`: an independent float64 radial
value/derivative and full center/neighbor gradient for explicitly supplied
relative vectors, integer exponent, cutoff and guard. It does not choose
species, modes or periodic images. The companion tests check a finite
difference, force/torque cancellation on water geometry, and the guard.
These are implementation checks, not a real-system PES search experiment.

Root review corrected an earlier draft that confused adjacent ELF constants
10000/0.5 with the actually loaded 1/-3. The corrected addresses and bytes
were independently checked before writing the isolated instruction probe.
See [probe protocol and resource blocker](../../research/ga_ssw/evidence/s1-cutoff-contract-20260922/README.md).
Only syntax/source checks have completed; both CPU submission attempts were
rejected with `AssocGrpBilling`. No numerical or native-instruction result
is claimed for the new helper or probe.

## Decision

Do not build S1–S6, choose a default Q switch, or add a Q heuristic. The
current useful boundary is the verified S1 radial dispatch/center-neighbor
write pattern; broader descriptor and periodic mapping remain unverified. The
forced-Q result remains a mixed diagnostic rather than evidence for a default
policy.
