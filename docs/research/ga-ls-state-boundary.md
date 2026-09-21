# GA and LS adaptive-state boundary

## Observed Python lifecycle

`paper_reference.run_ssw` constructs a fresh LS runtime for each call. For the
paper mode, `FrozenBondSoftening` and `LSResponseState` are initialized at
`paper_reference.py:227-240`; the response controller is updated only after a
completed outer step at `paper_reference.py:469-472`. Thus the adaptive total
softening strength is inherited across SSW steps inside one call, when that walker begins its
next outer attempt, but not across separate calls.

`paper_ga.run_ga_ssw` calls that walker independently for each initial quick
seed, generation-short seed, and fine seed (`paper_ga.py:337-340`). Each call
therefore starts from the explicit `ls` settings and the current seed geometry.
The GA archive carries geometry, energy, projection and lineage; it carries no
LS response state. This is a restart boundary, rather than an accidental
sharing of mutable LS objects.

## Evidence from the supplied GA boundary

The decompiled Java outer layer launches one `LaspCalculation` task per input
model and parses each task's `all.arc`; its `SSWOpt`/`SSWExplore` calls pass
configured template parameters and do not expose an LS-state object between
tasks. The archived GA interface audit records this process/file boundary and
the separate per-task optimization/search calls in
`docs/research/java-lasp-interface-boundary.md`. The GA/ASE port specification
also defines `run_opt_batch` as a batch of independent candidates and lists
random stream, budget and stage state separately from the physical walker;
it does not establish cross-task LS-state transfer.

The LS response documentation establishes a within-step/within-walker update
law and explicitly says the current paper mode should remain distinct from the
release's adaptive schedule. It does not provide evidence that a response from
one parent or one archived fine landing is physically meaningful for another
parent's frozen bond set. In particular, a new parent can have different bonds,
bond counts and a different current response baseline.

## Conclusion and boundary

There is no evidence-backed, mandatory cross-GA-call inheritance requirement.
The current behavior is best labeled **independent LS restart per walker**, with
adaptive feedback retained across outer steps within that walker. Calling this a
confirmed native parity fact would exceed the Java evidence: the executable
boundary does not expose enough internal state to prove its hidden lifetime.
Conversely, changing the Python controller to share `LSResponseState` would be
a new scientific policy, because the state is tied to a particular frozen bond
table and geometry. Keep the current restart boundary until a declared
cross-parent state definition and matched experiment are supplied. This audit
does not add state transfer or claim that restart is universally optimal.


Root source check (2026-09-12): uploaded decompiled/sgn/lasp/LaspCalculation.java
lines59–88 writes each model to its own input.arc, copies lasp.in/potential and
launches the separate process. app_ssw_ga/SSWGaSupport.java:129–148 creates that
batch and collects all.arc records. No serialized adaptive bond table or LS
response is transferred at these inspected interfaces. Existing task directories
and other hidden native restart mechanisms are not ruled out by this slice.
