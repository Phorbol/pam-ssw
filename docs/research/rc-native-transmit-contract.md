# RC transmit lambda: geometry and force routing, not a force-only switch

2026-09-10. **A consequential recovered distinction:** the release's `krot`
parameter appears in both `divide_force_` and the forward map `rigid_x_r2c_`.
Consequently, copying the force-transmission formula into our existing exact
Jacobian walker would not reproduce its coordinate contract. No public RC module
was changed; no PES, MACE, native main or protection code ran.

## Source definitions and their limit

Local primary paper `225.pdf`/`225.txt`, §2.2.2–2.2.3, and
`ct5c00350_si_001.pdf`/`.txt`, §1 and §8, are under
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/`.
The [publisher record](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00350)
was independently checked; it confirms the paper and SI, but the public record
adds no missing lambda-dependent differential proof.

The paper defines a root pose and child bond rotations, decomposes each force
into a lambda-scaled tangential contribution and a residual transmitted toward
ancestors, and calls the resulting forces energy derivatives (near eq 20).
Those statements alone do not prove equality to the derivative of a particular
finite-angle implementation. SI §1 describes bond alignment plus self-rotation;
it does not resolve the release's additional lambda-dependent inherited rotation
found below. SI Table S2 reports lambda-dependent chignolin search costs
(0.5/0.6/0.7/0.8/0.9 → N50 20308/23557/16496/31338/36858).
That is system-specific empirical evidence, not a universal default calibration.

## 1. Original-instruction force slice: now executable and discriminated

`rigid_f_c2r_` directly calls `divide_force_` at **0x816be6**. The latter's
body-type-2 branch uses the axis triplet at body+0xf8…+0x108, Cartesian force
input and relative position input. The isolated instruction slice
**0x81826c → 0x8184ef**, stopping *before* `crossproduct_`, was executed in
Unicorn using the unmodified uploaded ELF and original constants. Eight fixtures
combine two arbitrary oblique unit axes/positions/forces and krot=0,.5,.7,1.
No original function entry, allocation, recursion, MPI or caller initialization
was executed. This is deliberately an internal-block oracle, not whole-routine
or full-native parity. Every fixture uses a nonzero off-axis distance and avoids
the separately visible near-axis branch.

For unit axis a, relative position v and force F, define

```
w = v - a (a·v)
F_axis = a (a·F)
F_radial = w (w·F)/(w·w)
F_tangent = F - F_axis - F_radial.
```

The measured outputs of this block are:

```
parent-buffer force = F - krot*(F - F_axis)
local torque-input force = krot*F_tangent
nonrotating-force workspace = F_axis + F_radial.
```

Parent-buffer agreement is exact in all eight fixtures; torque-input error is
at most 9.72e-17. The alternative `F - krot*F_tangent`, suggested by applying
the paper's residual formula literally to this buffer, differs by up to
0.4984645203. Thus the distinction is not inferred from an axis-aligned example.
The local torque input is passed with w to `crossproduct_` at 0x8184ef; the
result is subtracted from the body torque accumulator +0x68…+0x78 at
0x818530–0x818540. Sign/order must include this subtraction when reading torques.

**This does not demonstrate lost total force or an algorithm bug.** These are
internal routing buffers; radial constraint forces and root translations can
have separate channels. The oracle does not execute their later use. It does
establish that the paper's simple residual cannot replace this internal block
and be called release parity.

Reproducer: `research/ga_ssw/probe_native_rc_transmit.py` with
`PYTHONPATH=/tmp/pam-ssw-unicorn-probe:.`; evidence and exact ELF SHA256:
`research/ga_ssw/evidence/native-rc-transmit-slice/{probe.py,result.json}`.
Budget: 1,000 instructions and one second per fixture. No code bytes patched.

## 2. Forward geometry also contains krot

In `rigid_x_r2c_`, the krot address is obtained at **0x81d213**. After the
inherited-rotation workspace is passed through `r2vec_` at **0x81db97**, the
routine scales all three returned vector components by **abs(1-krot)**:
0x81dbc8 subtracts krot from the original 1.0 constant; 0x81dbcc applies the
absolute-value mask; 0x81dbda/0x81dbe2 multiply the three components.
`vec2r_` is then called at **0x81dbf5** to rebuild that rotation.
Original constants at 0x4a4e950 and 0x4a4e910 establish 1.0 and the sign-clear
mask. This is static instruction evidence, not an executed whole-map test.

The same entry contains Kabsch calls and the RMASTER/RINH/RLAT composition.
This task does not infer their complete matrix order or shared-endpoint gauge.
For an isolated zero-lattice-change branch, it is still necessary to establish
which inherited rotation is being scaled before claiming a complete analytic map.
The evidence rules out treating krot as exclusively a post-hoc force multiplier.

## 3. The outer coefficient is scheduled, not necessarily constant

In `rigidssw_`, **0x4c3468–0x4c3503** increments the named integer
`rigidssw_$NMIN.0.2`, then sets krot from `krotsave`, `krotfreq`, `krotpower`:

```
s = incremented native NMIN counter
if frequency > 1:
    krot = krotsave + (1-krotsave) * ((s % frequency)/(frequency-1))**power
else:
    krot = krotsave.
```

For the visible nonnegative counter branch, the remainder is the ordinary
integer remainder. The call at 0x4c34ca targets `pow` (0x4920880).
A preceding call is `foundnew_` at 0x4c342f. This does **not** establish whether
NMIN equals accepted basins, completed proposals or another outer event over all
branches: a future controller must recover that counter lifecycle before using
this as its own step schedule.

The actual uploaded XXXII `lasp.in` supplies `rigid.krot 0.7`,
`rigid.krotpower 1.0`, `rigid.krot_freq 10`. The local formula would generate
.7 + .3*(s%10)/9, spanning .7…1, on applicable events; do not describe the whole
run as fixed lambda=.7. `rigid_rotfact` is a separate parameter in the force
routine; it must not be substituted for krot merely because of its name.
Direct calls to the forward and force entries in rigidssw are also visible at
0x4c7380 and 0x4c73e8; no vtable inference is needed for these sites.

## Consequence for our implementation and a minimal implementation contract

Our `RigidChainChart`/`RigidForestChart` forward map uses exact parent-transform
composition and child bond torsions; forces are precisely J.T F, with a scalar
energy and exact finite-angle chain rule. Its chart contains no native krot,
Kabsch inherited-twist adjustment or native counter schedule.

The release supplies a **coupled coordinate/routing policy** that our map omits.
It can change how a root/ancestor proposal spreads into descendant orientation,
and therefore the sampled directions and numerical conditioning. There is no
new Cartesian force oracle or new chemical degree of freedom in this observation.
With freely parameterized torsions, a changed coordinate coupling may chiefly
change proposal geometry rather than expand the molecular configuration set;
that is an inference to test, not a native equivalence proof.

Classification supported now:

- Not simply a coordinate-independent force redistribution: the forward map
  itself depends on krot.
- Not established as a conventional positive-definite metric/preconditioner:
  no recovered M or equality `native force = M^-1 J.T F` exists.
- Not established as nonconservative: that requires testing the native forces
  against the derivative of the **same lambda-dependent forward map**. Testing
  them against our different chart would answer the wrong question.
- The paper's conservation statement is weaker than energy/work consistency;
  neither proves the other for a reduced-coordinate implementation.

The recovered local split and explicit-index schedule are ready as isolated
reference functions, with the oracle fixtures as contracts. They are **not yet
ready for insertion into a conservative SSW optimizer**. Minimal next closure:

1. Freeze a two-body, one-shared-bond nonperiodic topology and krot; recover the
   complete native forward map for that supported branch, including RINH order
   and the mapping between input rotation coordinates and child self-angle.
2. Execute its original instructions on finite but small perturbations and compare
   the complete returned force to finite differences of a known Cartesian scalar
   energy using exactly that forward map. Include radial-only forces to isolate
   the discrepancy above; then include a two-joint chain to test recursive routing.
3. If consistent, implement that chart and derive its own exact Jacobian pullback;
   retain current chart as a baseline. If inconsistent, keep the native force
   policy as a clearly labeled reference, not the gradient supplied to Safe-total.
4. Freeze krot during a live Gaussian continuation/chart. Recover the outer
   NMIN lifecycle separately before implementing the native schedule; changing
   krot while the reference/bias history is live can change the represented
   scalar objective.

No new heuristic threshold, production default or efficiency claim follows from
this closure. The main algorithmic opportunity is a better justified cooperative
coordinate map, not copying a residual-force formula in isolation.

Static evidence: `rc-transmit-evidence/{divide_force.asm,
inherited-rotation.asm,schedule-and-direct-calls.asm}` plus the earlier complete
`rc-native-evidence/rigid_{f_c2r,x_r2c}.asm`.
