# Native VC lifecycle trace — prepared, not executed

Question: do cell-mode random-generation/update calls recur within one native
outer step, or only at a new outer step? Static dispatch evidence cannot settle
the executed frequency. This is original-program control-flow diagnosis only;
it supports no LJ phase ordering, search quality, efficiency or native/Python
trajectory-equivalence claim.

One arm only: input `Run_Type 15`, explicitly present in the uploaded TYPE1-AlOH
lasp.in. Record the actual object run_type; do not assume input-to-object mapping.
Input 16 is deferred and its semantic meaning is unassigned. No binary edits,
state-register writes, date/identity changes or protection bypass are permitted.
If startup refuses execution, preserve it and stop.

Input: reuse `lj-bulk-stacking-readiness-20261007/geometry.py::build_pair((2,2,6))`,
fcc Ar24, all three PBC axes. Rescale its sigma=1 geometry using
sigma=(150830.0777733409/388.36848195153647)^(1/6), recovered from native
ljperipot pair literals in `docs/research/native-lj-stress-producer.md`.
This is a length-scale preparation, not a new PES or an end-to-end qualification
of cutoff/neighbor behavior. Reuse the archived native ARC writer atom rows;
replace its C60-only 50-A cell line with this structure's cellpar. Native initial
optimization, stress and periodic LJ readiness remain runtime gates.

The potential lj keyword comes from TYPE0-LJ75. Its empty lj.pot is not a
parameter table. Input NG/NG_cell=3 are diagnostic caps, shortened from the
TYPE1 values 10/7; ds_atom=.6 and ds_cell=1.6 retain that example's values.
MaxOptstep=50 is a diagnostic cap; ftol/strtol=.05 are native input units,
without importing their meaning into ASE. Ratio_atomcell=2 is explicit and
matches the recovered native initialized value; parser string was verified
at ELF 0x4a49f94. Record the consumed value. Temperature 200 comes from LJ75;
MD disabled, fixed seed 20261012. These values are not claimed universal.

Boundaries: one CPU, Slurm 5 minutes total, native debugger process 120 seconds,
at most two distinct observed nsswstep values at ssw_move entry, input two outer
steps, and 500 trace events. No retry. Existing bounded_process supervisor reaps
all descendants. A timeout, event cap, initial failure, unknown input, absent
cell branch or trace-read error is retained incomplete evidence.

GDB records ssw_move, get_random_mode0 and post-choice point, climb and its
post-dispatch point, update_mode0, gen_randommode, cell/atom pattern calls.
It logs nsswstep, actual run_type, raw lcellmove, step_perssw, ng/ng_cell,
ratio_atomcell, cumulative/per-step call counts and climb serial. The
cell_mode_event_serial is an observation counter, not a claimed native cell
cycle counter: associate post-dispatch, update and generator calls within
each climb_serial and nsswstep. Entry return addresses identify call sites.
Every event also retains raw general registers and up to four unwind frames,
including caller PCs/functions. A current native Gaussian/cell-cycle index
and reverse-communication status have not been mapped in this bounded slice;
neither climb_serial nor cell_mode_event_serial is a substitute for them.
Fortran entry rdi points to a descriptor, whose first pointer is the object.
Generator/pattern observations use the last captured parent object rather than
misinterpreting a local descriptor. Address/pointer read failure stops the run.

Decision/readout: repeated post-climb generator events under the same nsswstep
and cell flag establish within-step fresh random-input calls. One generator
per outer-step with multiple observed cell climbs supports retention at those
observed boundaries. Fewer than two relevant cell climbs, or no generator
coverage, leaves this comparison unresolved; do not call absence proof.
Calling gen_randommode does not establish independent replacement of the
whole direction: persistent-state mixing remains a separate data-flow question.
Repeated events may be reverse-communication returns within one cell cycle.
Unless raw caller/state evidence independently resolves that distinction,
report only same-outer repetition; do not conclude redraw once per cell cycle
or count the trace's climb entries as independent cell cycles. This probe may
therefore narrow the uncertainty without fully resolving cycle frequency.

Review actual lasp.out/allkeys parser echoes, trace.jsonl, trace-stop/error,
process.json and cleanup_survivors before assigning conclusions. Runner status
execution_returned_unqualified deliberately does not mean success. Preserve
all raw native artifacts and failed/censored costs. Default runner invocation
only prepares files; --execute is explicit. Root owns execution and review.

## Preparation fault and one focused correction

CPU1754601 returned supervisor success but GDB startup failed before native
execution: uploaded ELF mode0664 was not executable (child return1, startup126).
No PES or lifecycle events occurred; raw failure and zero cleanup survivors are
retained. Revision2 copies identical ELF bytes into the run directory, verifies
the same hash and sets only the copy to0700. The original archive is untouched.
The runner also propagates a failed child/timeout to its scheduler exit rather
than relying on the cleanup supervisor's exit alone. This is a manually reviewed
preparation correction, not an automatic native retry or a physics change.
One further arm under the same120-second/500-event/two-step bounds is authorized
by the existing bounded diagnostic scope; no additional parameter changes.

## Executed observation and field correction

Run1754624 completed without trace errors/caps or leftover processes. The raw
observer inherited an incorrect field attribution from the previous static
note: run_type is PARA+0x100, and dispatch rbx is PARA, not CSSW object. This
was resolved from raw registers, adjacent update entries, DWARF and exact
instructions without another PES run. Raw trace is preserved; see readout.md
and analyze.py for explicit invalidation/correction. The current trace.py uses
PARA for run_type and r13 descriptor at dispatch; this corrected observer has
not been re-executed. The validated conclusion is repeated random generation
at each of three observed native cell Gaussian stages, not independent redraw.

## Revision 3: distinguish fresh input from history mixing

The first executed trace established generation at each of three native cell
Gaussian stages, but did not record vector inputs. Static follow-up found that
cell pattern .8/.2 are separately enabled additive vector coefficients, not a
categorical draw; update_mode0 also constructs a trajectory-difference vector
and sets a history coefficient at pattern+0x48. This is a distinct unresolved
mechanism, not a request to repeat the frequency test.

Run one additional identical-input arm with the same one-CPU/120-second/500-event
bounds. Only the read-only observer changes: record generator input pattern,
lrigid and actual vector descriptors, then vectors after the history scaling
and two cell-mode additions. Freeze the observer in the run directory. No
physics parameter, native memory or Python algorithm changes are allowed.

Expected distinguishing signal: a nonzero entering vector and nonzero history
coefficient with the observed post-scaling vector consistent with their product
shows actual historical participation at that boundary. Zero coefficient or
zero vector means that boundary does not demonstrate participation, even if the
static code can support it. Invalid descriptor or unexpected control flow is
incomplete evidence, not grounds for automatic retries. Vector snapshots do not
establish the full physical meaning of the local-cell term or search benefit.
If history participates, document the concrete mechanism gap and compare it to
the already recovered Python direction feedback before proposing any public
VC state change. If not, retain the observed conditional boundary and close this
probe. No default mixing coefficient will be inferred as generally optimal.
